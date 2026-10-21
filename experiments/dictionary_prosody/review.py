"""Build an offline blind listening kit; import only explicitly checked boundaries."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
from pathlib import Path

from .align import sha256

ROOT = Path(__file__).parent


def build_kit(results: Path, output: Path, *, seed: int = 20261020) -> dict:
    if output.exists():
        raise ValueError("Review output must not already exist")
    automatic = json.loads((results / "automatic-alignment.json").read_text(encoding="utf-8"))
    estimates = {row["id"]: row for row in automatic["rows"]}
    all_rows = []
    for path in sorted(results.glob("*/manifest.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        all_rows.extend(
            (path.parent, row) for row in document["rows"] if row["status"] == "generated"
        )
    baselines = {
        (directory.name, row["word"], row["context"], row["seed"], row.get("repeat", 0)): row["id"]
        for directory, row in all_rows
        if row["method"] == "reading"
    }
    random.Random(seed).shuffle(all_rows)
    codes = {row["id"]: f"clip{index:03d}" for index, (_, row) in enumerate(all_rows, 1)}
    # Validate all assets first; generation is separate from analysis/annotation.
    for directory, row in all_rows:
        if row["id"] not in estimates or sha256(directory / row["wav"]) != row["wav_sha256"]:
            raise ValueError("Missing estimate or WAV integrity mismatch")
    output.mkdir(parents=True)
    (output / "clips").mkdir()
    clips, key = [], {}
    for directory, row in all_rows:
        code = codes[row["id"]]
        source = directory / row["wav"]
        shutil.copyfile(source, output / "clips" / (code + ".wav"))
        with source.with_name(row["id"] + "-f0.csv").open(encoding="utf-8") as stream:
            series = list(csv.DictReader(stream))
        estimate = estimates[row["id"]]
        baseline = codes[
            baselines[
                (directory.name, row["word"], row["context"], row["seed"], row.get("repeat", 0))
            ]
        ]
        clips.append(
            {
                "code": code,
                "audio": f"clips/{code}.wav",
                "baseline_audio": f"clips/{baseline}.wav",
                "duration": row["audio_seconds"],
                "reading": row["entry"]["reading"],
                "target_moras": len(
                    [m for m in estimate["estimated_moras"] if m["role"] == "target"]
                ),
                "f0": [
                    [float(sample["time_seconds"]), float(sample["f0_hz"])] for sample in series
                ],
                "estimated_moras": [
                    {k: m[k] for k in ("label", "role", "start", "end")}
                    for m in estimate["estimated_moras"]
                ],
            }
        )
        key[code] = {
            "id": row["id"],
            "run": directory.name,
            "method": row["method"],
            "nucleus": row["nucleus"],
            "wav_sha256": row["wav_sha256"],
            "duration": row["audio_seconds"],
            "target_moras": clips[-1]["target_moras"],
            "mora_labels": [m["label"] for m in clips[-1]["estimated_moras"]],
        }
    fingerprint = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()
    bundle = {"schema": "blind-prosody-review-v1", "fingerprint": fingerprint, "clips": clips}
    (output / "bundle.json").write_text(json.dumps(bundle, ensure_ascii=False), encoding="utf-8")
    (output / "review-key.json").write_text(
        json.dumps({"fingerprint": fingerprint, "clips": key}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    shutil.copyfile(ROOT / "review.html", output / "index.html")
    return {"clips": len(clips), "fingerprint": fingerprint}


def import_review(key: dict, draft: dict) -> dict:
    """Reject invalid claims instead of converting estimates into manual evidence."""
    if not isinstance(key, dict) or not isinstance(draft, dict):
        raise ValueError("Review and key must be objects")
    if draft.get("schema") != "blind-prosody-review-v1" or draft.get("fingerprint") != key.get(
        "fingerprint"
    ):
        raise ValueError("Review schema/fingerprint mismatch")
    reviewer = draft.get("reviewer")
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError("A reviewer identifier is required")
    records = draft.get("reviews")
    if not isinstance(records, list):
        raise ValueError("reviews must be a list")
    seen = set()
    reviews, annotations = {}, {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Review records must be objects")
        code = record.get("code")
        if not isinstance(code, str) or code not in key["clips"] or code in seen:
            raise ValueError("Unknown or duplicate blind clip code")
        seen.add(code)
        expected = key["clips"][code]
        for flag in ("transcript_checked", "alignment_checked", "uncertain"):
            if type(record.get(flag)) is not bool:
                raise ValueError(f"{flag} must be explicitly boolean")
        nucleus = record.get("perceived_nucleus")
        if nucleus is not None and (
            type(nucleus) is not int or not 0 <= nucleus <= expected["target_moras"]
        ):
            raise ValueError("Perceived nucleus outside target mora range")
        moras = record.get("moras")
        if (
            not isinstance(moras, list)
            or len(moras) != len(expected["mora_labels"])
            or any(not isinstance(m, dict) for m in moras)
        ):
            raise ValueError("moras must preserve the kit unit list")
        if [m.get("label") for m in moras] != expected["mora_labels"]:
            raise ValueError("Mora labels differ from the kit transcript")
        rating = record.get("naturalness")
        if rating is not None and (type(rating) is not int or not 1 <= rating <= 5):
            raise ValueError("Naturalness must be an integer in 1..5 or null")
        previous = 0.0
        for mora in moras:
            start, end = mora.get("start"), mora.get("end")
            if (
                type(start) not in (int, float)
                or type(end) not in (int, float)
                or not previous <= start < end <= expected["duration"]
                or not isinstance(mora.get("label"), str)
                or not mora["label"]
            ):
                raise ValueError("Invalid or overlapping mora boundaries")
            previous = end
        if record["alignment_checked"] and (not moras or not record["transcript_checked"]):
            raise ValueError(
                "Checked alignment needs a checked transcript and non-empty boundaries"
            )
        copied = dict(record) | {
            "reviewer": reviewer.strip(),
            "wav_sha256": expected["wav_sha256"],
            "accent_verified": False,
        }
        reviews[expected["id"]] = copied
        if record["transcript_checked"] and record["alignment_checked"]:
            annotations[expected["id"]] = {
                "reviewer": reviewer.strip(),
                "transcript_checked": True,
                "moras": moras,
                "wav_sha256": expected["wav_sha256"],
                "source": "explicit reviewer confirmation",
            }
    return {"reviews": reviews, "annotations": annotations, "accent_verified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--key", type=Path)
    parser.add_argument("--import-review", type=Path)
    parser.add_argument("--annotations-output", type=Path)
    args = parser.parse_args()
    if args.import_review is not None:
        if (
            args.key is None
            or args.output.exists()
            or (args.annotations_output is not None and args.annotations_output.exists())
        ):
            parser.error("Import requires --key and a new output file")
        result = import_review(
            json.loads(args.key.read_text()), json.loads(args.import_review.read_text())
        )
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        if args.annotations_output is not None:
            args.annotations_output.write_text(
                json.dumps(result["annotations"], ensure_ascii=False, indent=2, allow_nan=False)
                + "\n",
                encoding="utf-8",
            )
        print(
            f"Imported {len(result['reviews'])} reviews, {len(result['annotations'])} explicitly checked alignments"
        )
    else:
        print(json.dumps(build_kit(args.results, args.output)))


if __name__ == "__main__":
    main()
