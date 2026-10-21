"""Extract WORLD F0 and describe acoustics; never auto-certify lexical accent."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def semitone_drop(before: float, after: float) -> float:
    if not math.isfinite(before) or not math.isfinite(after) or min(before, after) <= 0:
        raise ValueError("F0 values must be finite and voiced")
    return 12 * math.log2(before / after)


def summarise_f0(times, f0, intervals: list[dict] | None = None) -> dict:
    times, f0 = np.asarray(times), np.asarray(f0)
    if (
        times.ndim != 1
        or times.shape != f0.shape
        or not len(times)
        or not np.isfinite(times).all()
        or not np.isfinite(f0).all()
        or (f0 < 0).any()
        or (np.diff(times) <= 0).any()
    ):
        raise ValueError("Invalid F0/time series")
    voiced = f0 > 0
    summary = {
        "voiced_fraction": float(voiced.mean()),
        "median_f0_hz": float(np.median(f0[voiced])) if voiced.any() else None,
        "accent_verified": False,
        "mora_alignment": "unavailable",
    }
    if intervals is None:
        return summary
    moras = []
    previous_end = 0.0
    for interval in intervals:
        start, end = interval["start"], interval["end"]
        if (
            not math.isfinite(start)
            or not math.isfinite(end)
            or start < previous_end
            or end <= start
            or end > times[-1] + 0.005
        ):
            raise ValueError("Mora boundaries must be ordered, disjoint and inside the audio")
        selected = (times >= start) & (times < end)
        measured = selected & voiced
        value = float(np.median(f0[measured])) if measured.sum() >= 3 else None
        moras.append(
            {
                "label": interval["label"],
                "start": start,
                "end": end,
                "voiced_frames": int(measured.sum()),
                "median_f0_hz": value,
            }
        )
        previous_end = end
    summary.update(
        mora_alignment="manual",
        moras=moras,
        adjacent_drops_semitones=[
            None
            if a["median_f0_hz"] is None or b["median_f0_hz"] is None
            else semitone_drop(a["median_f0_hz"], b["median_f0_hz"])
            for a, b in zip(moras, moras[1:], strict=False)
        ],
    )
    return summary


def validate_annotation(annotation: object, wav_sha256: str) -> None:
    if (
        not isinstance(annotation, dict)
        or not isinstance(annotation.get("reviewer"), str)
        or not annotation["reviewer"].strip()
        or annotation.get("transcript_checked") is not True
    ):
        raise ValueError("Manual intervals require reviewer and transcript_checked=true")
    if annotation.get("wav_sha256", wav_sha256) != wav_sha256:
        raise ValueError("Annotation WAV hash mismatch")
    if not isinstance(annotation.get("moras"), list) or not annotation["moras"]:
        raise ValueError("Manual annotation needs non-empty mora intervals")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument(
        "--annotations", type=Path, help="JSON mapping clip IDs to manually checked mora intervals"
    )
    args = parser.parse_args()
    import pyworld
    import soundfile as sf

    from .run import digest

    manifest = json.loads((args.run / "manifest.json").read_text(encoding="utf-8"))
    annotations = (
        {} if args.annotations is None else json.loads(args.annotations.read_text(encoding="utf-8"))
    )
    results = []
    for row in manifest["rows"]:
        if row["status"] != "generated":
            continue
        wav_path = args.run / row["wav"]
        if digest(wav_path) != row["wav_sha256"]:
            raise ValueError("WAV hash mismatch")
        audio, sr = sf.read(wav_path, dtype="float64", always_2d=True)
        if sr < 16000 or not np.isfinite(audio).all():
            raise ValueError("WORLD requires finite audio at >=16 kHz")
        x = np.ascontiguousarray(audio.mean(axis=1))
        f0, times = pyworld.harvest(x, sr, f0_floor=60.0, f0_ceil=600.0, frame_period=5.0)
        annotation = annotations.get(row["id"])
        if annotation is not None:
            validate_annotation(annotation, row["wav_sha256"])
        summary = summarise_f0(times, f0, None if annotation is None else annotation["moras"])
        summary.update(
            pcm_sha256=hashlib.sha256(audio.astype("<f4").tobytes()).hexdigest(),
            id=row["id"],
            method=row["method"],
            nucleus=row["nucleus"],
            rms=float(np.sqrt(np.mean(x**2))),
            peak=float(np.max(np.abs(x))),
            voiced_start_seconds=None if not (f0 > 0).any() else float(times[f0 > 0][0]),
            voiced_end_seconds=None if not (f0 > 0).any() else float(times[f0 > 0][-1]),
            duration_seconds=len(x) / sr,
            pitch_range_semitones=None
            if (f0 > 0).sum() < 2
            else semitone_drop(
                float(np.percentile(f0[f0 > 0], 90)), float(np.percentile(f0[f0 > 0], 10))
            ),
        )
        summary["f0_csv"] = row["id"] + "-f0.csv"
        with (args.run / summary["f0_csv"]).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["time_seconds", "f0_hz", "voiced"])
            writer.writerows(zip(times, f0, (f0 > 0).astype(int), strict=True))
        results.append(summary)
    output = {
        "extractor": f"pyworld {pyworld.__version__} harvest",
        "frame_period_ms": 5,
        "f0_floor_hz": 60,
        "f0_ceil_hz": 600,
        "note": "No equal-duration mora assumption. No lexical accent verdict from global F0. No listening judgement performed by this tool.",
        "results": results,
    }
    (args.run / "analysis.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Analysed {len(results)} WAVs; accent_verified remains false pending manual review")


if __name__ == "__main__":
    main()
