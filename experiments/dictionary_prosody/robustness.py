"""Audit saved evidence and stress-test uncalibrated F0 screens, not accent accuracy."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .align import sha256
from .analyse import summarise_f0
from .probes import pitch_labels

ROOT = Path(__file__).parent


def classify(drops: list[float | None], nucleus: int, transition=1.5, tolerance=3.0):
    """Compare every adjacent transition; None means insufficient voiced evidence."""
    labels = pitch_labels(len(drops), nucleus, particle=True)
    if not drops or any(d is None for d in drops):
        return None
    if transition <= 0 or tolerance < 0 or not np.isfinite([transition, tolerance]).all():
        raise ValueError("Invalid thresholds")
    if not np.isfinite(drops).all():
        raise ValueError("Non-finite pitch difference")
    return all(
        drop >= transition
        if a + b == "HL"
        else drop <= -transition
        if a + b == "LH"
        else abs(drop) <= tolerance
        for a, b, drop in zip(labels[:-1], labels[1:], drops, strict=True)
    )


def boundary_variants(intervals: list[dict], radius: float):
    """Independent offsets for all boundaries, including target/particle endpoints.

    The grid is an adversarial sensitivity probe, not a boundary error distribution.
    Shared internal boundaries remain shared; invalid orderings are discarded.
    """
    if not intervals or radius <= 0 or not np.isfinite(radius):
        raise ValueError("Need intervals and positive finite radius")
    for left, right in zip(intervals, intervals[1:], strict=False):
        if abs(left["end"] - right["start"]) > 1e-8:
            raise ValueError("Expected contiguous estimated intervals")
    edges = [intervals[0]["start"]] + [m["end"] for m in intervals]
    for shifts in itertools.product((-radius, 0.0, radius), repeat=len(edges)):
        changed = [edge + shift for edge, shift in zip(edges, shifts, strict=True)]
        if changed[0] < 0 or any(a >= b for a, b in zip(changed, changed[1:], strict=False)):
            continue
        yield [
            unit | {"start": start, "end": end}
            for unit, start, end in zip(intervals, changed, changed[1:], strict=False)
        ]


def load_f0(path: Path, duration: float):
    with path.open(encoding="utf-8") as stream:
        records = list(csv.DictReader(stream))
    times = np.array([float(row["time_seconds"]) for row in records])
    values = np.array([float(row["f0_hz"]) for row in records])
    summarise_f0(times, values)  # Check monotonicity, finiteness and sign.
    if times[0] < 0 or times[-1] > duration + 0.005:
        raise ValueError("F0 times outside WAV")
    if any(int(row["voiced"]) != int(f > 0) for row, f in zip(records, values, strict=True)):
        raise ValueError("F0 voicing column disagrees")
    return times, values


def screen(times, f0, intervals):
    return summarise_f0(times, f0, intervals)["adjacent_drops_semitones"]


def audit(results: Path, alignment: Path) -> dict:
    import soundfile as sf

    aligned = json.loads(alignment.read_text(encoding="utf-8"))
    records = {}
    sources = {str(alignment.relative_to(results)): sha256(alignment)}
    control_groups = []
    for manifest_path in sorted(results.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        analysis_path = manifest_path.with_name("analysis.json")
        analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
        for path in (manifest_path, analysis_path):
            sources[str(path.relative_to(results))] = sha256(path)
        acoustic = {row["id"]: row for row in analysis["results"]}
        groups = {}
        for row in manifest["rows"]:
            if row["status"] != "generated":
                continue
            key = (manifest_path.parent.name, row["id"])
            if key in records:
                raise ValueError("Duplicate run/clip")
            wav = manifest_path.parent / row["wav"]
            audio, sr = sf.read(wav, dtype="float32", always_2d=True)
            pcm_hash = hashlib.sha256(audio.astype("<f4").tobytes()).hexdigest()
            if (
                sha256(wav) != row["wav_sha256"]
                or pcm_hash != acoustic[row["id"]]["pcm_sha256"]
                or not np.isfinite(audio).all()
            ):
                raise ValueError("Saved audio integrity failure")
            f0_path = wav.with_name(row["id"] + "-f0.csv")
            series = load_f0(f0_path, len(audio) / sr)
            sources[str(f0_path.relative_to(results))] = sha256(f0_path)
            if sr != row["sample_rate"] or abs(len(audio) / sr - row["audio_seconds"]) > 1 / sr:
                raise ValueError("WAV rate/duration metadata mismatch")
            records[key] = (row, series)
            condition = (row["spoken_text"], row["caption"], row["seed"], row["repeat"])
            groups.setdefault(condition, []).append((row["id"], pcm_hash))
        for group in groups.values():
            if len(group) > 1:
                control_groups.append(
                    {
                        "run": manifest_path.parent.name,
                        "ids": [x[0] for x in group],
                        "pcm_identical": len({x[1] for x in group}) == 1,
                    }
                )
    aligned_keys = [(row["run"], row["id"]) for row in aligned["rows"]]
    if len(set(aligned_keys)) != len(aligned_keys) or set(aligned_keys) != set(records):
        raise ValueError("Alignment and manifest coverage differ")
    rows = []
    thresholds = list(itertools.product((1.0, 1.5, 2.0), (2.0, 3.0, 4.0)))
    for row in aligned["rows"]:
        manifest_row, (times, f0) = records[(row["run"], row["id"])]
        if any(
            row[k] != manifest_row[k]
            for k in ("wav_sha256", "word", "context", "seed", "method", "nucleus")
        ):
            raise ValueError("Alignment metadata mismatch")
        intervals = [m for m in row["estimated_moras"] if m["role"] in {"target", "particle"}]
        drops = screen(times, f0, intervals)
        count = len(intervals) - 1
        matched = [n for n in range(count + 1) if classify(drops, n) is True]
        target = row["nucleus"]
        detail = {
            "run": row["run"],
            "id": row["id"],
            "word": row["word"],
            "method": row["method"],
            "nucleus": target,
            "exact_greedy_transcript": row["expected_latin"] == row["greedy_latin"],
            "default_matching_nuclei_raw": matched,
            "raw_assessable": all(d is not None for d in drops),
            "original_gated_screen": row["provisional_shape_agreement"],
            "threshold_grid": [
                {
                    "transition": t,
                    "tolerance": s,
                    "matching_nuclei_raw": [
                        n for n in range(count + 1) if classify(drops, n, t, s) is True
                    ],
                }
                for t, s in thresholds
            ],
            "boundary_sensitivity": [],
            "accent_verified": False,
        }
        if target is not None:
            for radius in (0.02, 0.04, 0.08):
                verdicts = Counter()
                for variant in boundary_variants(intervals, radius):
                    if variant[-1]["end"] > times[-1] + 0.005:
                        continue
                    decision = classify(screen(times, f0, variant), target)
                    verdicts[str(decision)] += 1
                detail["boundary_sensitivity"].append(
                    {"radius_seconds": radius, "valid_grid_counts": dict(verdicts)}
                )
        rows.append(detail)
    indexed = {(r["run"], r["id"]): r for r in rows}
    pairs = []
    for row in aligned["rows"]:
        if row["nucleus"] is None:
            continue
        source = records[(row["run"], row["id"])][0]
        candidates = [
            r
            for (run, _), (r, _) in records.items()
            if run == row["run"]
            and r["method"] == "reading"
            and all(r[k] == source[k] for k in ("word", "context", "seed", "repeat"))
        ]
        if len(candidates) != 1:
            raise ValueError("Need exactly one matched reading-only control")
        control = indexed[(row["run"], candidates[0]["id"])]
        actual = indexed[(row["run"], row["id"])]
        pairs.append(
            {
                "run": row["run"],
                "id": row["id"],
                "nucleus": row["nucleus"],
                "reading_control": control["id"],
                "intervention_raw_match": row["nucleus"] in actual["default_matching_nuclei_raw"],
                "control_raw_match": row["nucleus"] in control["default_matching_nuclei_raw"],
                "intervention_assessable": actual["raw_assessable"],
                "control_assessable": control["raw_assessable"],
            }
        )
    return {
        "schema": "dictionary-prosody-robustness-v1",
        "packages": {p: importlib.metadata.version(p) for p in ("numpy", "soundfile")},
        "source_sha256": sources,
        "scope": "Post-hoc sensitivity and matched negative controls; not accent accuracy or statistical inference",
        "accent_verified": False,
        "human_listening_performed": False,
        "exact_condition_groups": control_groups,
        "rows": rows,
        "matched_control_comparisons": pairs,
        "summary": {
            "clips": len(rows),
            "intended_nucleus_clips": len(pairs),
            "raw_intervention_matches": sum(p["intervention_raw_match"] for p in pairs),
            "raw_reading_control_matches": sum(p["control_raw_match"] for p in pairs),
            "raw_intervention_only": sum(
                p["intervention_raw_match"] and not p["control_raw_match"] for p in pairs
            ),
            "raw_control_only": sum(
                p["control_raw_match"] and not p["intervention_raw_match"] for p in pairs
            ),
            "clips_matching_multiple_nuclei_raw": sum(
                len(r["default_matching_nuclei_raw"]) > 1 for r in rows
            ),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must not exist")
    report = audit(args.results, args.results / "automatic-alignment.json")
    report["code_sha256"] = {
        p.name: sha256(p) for p in (Path(__file__), ROOT / "analyse.py", ROOT / "probes.py")
    }
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
