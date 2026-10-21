"""Compare DIO/StoneMask with saved Harvest F0; neither is accent ground truth."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

import numpy as np

from .align import sha256
from .robustness import ROOT, classify, load_f0, screen


def compare_series(saved: np.ndarray, alternative: np.ndarray) -> dict:
    if (
        saved.ndim != 1
        or saved.shape != alternative.shape
        or not len(saved)
        or not np.isfinite(saved).all()
        or not np.isfinite(alternative).all()
        or (saved < 0).any()
        or (alternative < 0).any()
    ):
        raise ValueError("Need finite nonnegative equally sampled F0 series")
    joint = (saved > 0) & (alternative > 0)
    differences = 12 * np.log2(alternative[joint] / saved[joint])
    return {
        "frames": len(saved),
        "joint_voiced_frames": int(joint.sum()),
        "voicing_disagreement_fraction": float(np.mean((saved > 0) != (alternative > 0))),
        "median_absolute_difference_semitones": float(np.median(np.abs(differences)))
        if joint.any()
        else None,
        "p90_absolute_difference_semitones": float(np.percentile(np.abs(differences), 90))
        if joint.any()
        else None,
        "octave_scale_disagreement_frames": int((np.abs(differences) >= 9).sum()),
    }


def run(results: Path) -> dict:
    import pyworld
    import soundfile as sf

    alignment_path = results / "automatic-alignment.json"
    alignment = json.loads(alignment_path.read_text(encoding="utf-8"))
    rows = []
    for row in alignment["rows"]:
        wav = results / row["run"] / (row["id"] + ".wav")
        if sha256(wav) != row["wav_sha256"]:
            raise ValueError("WAV integrity failure")
        samples, sr = sf.read(wav, dtype="float64", always_2d=True)
        samples = np.ascontiguousarray(samples.mean(axis=1))
        csv_path = wav.with_name(row["id"] + "-f0.csv")
        times, saved = load_f0(csv_path, len(samples) / sr)
        replayed, replay_times = pyworld.harvest(
            samples, sr, f0_floor=60, f0_ceil=600, frame_period=5
        )
        if times.shape != replay_times.shape or not np.allclose(
            times, replay_times, rtol=0, atol=1e-9
        ):
            raise ValueError("Harvest replay frame grid differs")
        initial, dio_times = pyworld.dio(samples, sr, f0_floor=60, f0_ceil=600, frame_period=5)
        alternative = pyworld.stonemask(samples, initial, dio_times, sr)
        if times.shape != dio_times.shape or not np.allclose(times, dio_times, rtol=0, atol=1e-9):
            raise ValueError("F0 frame grids differ; interpolation is not assumed")
        intervals = [m for m in row["estimated_moras"] if m["role"] in {"target", "particle"}]
        saved_drops = screen(times, saved, intervals)
        alternative_drops = screen(times, alternative, intervals)
        count = len(intervals) - 1
        rows.append(
            {
                "run": row["run"],
                "id": row["id"],
                "nucleus": row["nucleus"],
                "wav_sha256": row["wav_sha256"],
                "harvest_csv_sha256": sha256(csv_path),
                "frame_comparison": compare_series(saved, alternative),
                "harvest_replay_exact": bool(np.array_equal(saved, replayed)),
                "harvest_replay_max_absolute_difference_hz": float(
                    np.max(np.abs(saved - replayed))
                ),
                "dio_stonemask_f0_hz": alternative.tolist(),
                "time_grid_seconds": {
                    "start": float(dio_times[0]),
                    "step": 0.005,
                    "frames": len(dio_times),
                },
                "harvest_matching_nuclei_raw": [
                    n for n in range(count + 1) if classify(saved_drops, n) is True
                ],
                "dio_stonemask_matching_nuclei_raw": [
                    n for n in range(count + 1) if classify(alternative_drops, n) is True
                ],
                "dio_stonemask_drops_semitones": alternative_drops,
                "original_gated_harvest_screen": row["provisional_shape_agreement"],
                "accent_verified": False,
            }
        )
    intended = [r for r in rows if r["nucleus"] is not None]
    return {
        "schema": "dictionary-prosody-extractor-crosscheck-v1",
        "scope": "Two WORLD estimators on the same WAV and unreviewed CTC intervals; not independent accent validation",
        "settings": {
            "sample_rate": "native WAV",
            "f0_floor": 60,
            "f0_ceil": 600,
            "frame_period_ms": 5,
        },
        "alignment_sha256": sha256(alignment_path),
        "code_sha256": {
            p.name: sha256(p)
            for p in (
                Path(__file__),
                ROOT / "robustness.py",
                ROOT / "analyse.py",
                ROOT / "probes.py",
            )
        },
        "packages": {
            p: importlib.metadata.version(p)
            for p in ("numpy", "pyworld", "soundfile", "setuptools")
        },
        "summary": {
            "clips": len(rows),
            "saved_harvest_replayed_exact_clips": sum(r["harvest_replay_exact"] for r in rows),
            "intended_nucleus_clips": len(intended),
            "raw_harvest_matches": sum(
                r["nucleus"] in r["harvest_matching_nuclei_raw"] for r in intended
            ),
            "raw_dio_stonemask_matches": sum(
                r["nucleus"] in r["dio_stonemask_matching_nuclei_raw"] for r in intended
            ),
            "matching_nucleus_sets_differ": sum(
                r["harvest_matching_nuclei_raw"] != r["dio_stonemask_matching_nuclei_raw"]
                for r in rows
            ),
        },
        "accent_verified": False,
        "human_listening_performed": False,
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must not exist")
    report = run(args.results)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
