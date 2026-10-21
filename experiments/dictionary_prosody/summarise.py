"""Summarise recorded runs and exact-input controls, without accent verdicts."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    runs = []
    controls = []
    for manifest_path in sorted(args.results.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        analysis = json.loads(manifest_path.with_name("analysis.json").read_text(encoding="utf-8"))
        analysed = {row["id"]: row for row in analysis["results"]}
        groups = defaultdict(list)
        generated = [row for row in manifest["rows"] if row["status"] == "generated"]
        for row in generated:
            key = (row["spoken_text"], row["caption"], row["seed"])
            groups[key].append(row)
        for group in groups.values():
            if len(group) < 2:
                continue
            controls.append(
                {
                    "run": manifest_path.parent.name,
                    "ids": [row["id"] for row in group],
                    "pcm_identical": len({analysed[row["id"]]["pcm_sha256"] for row in group}) == 1,
                }
            )
        runs.append(
            {
                "name": manifest_path.parent.name,
                "status": manifest["status"],
                "generated": len(generated),
                "planned_total": len(manifest["rows"]),
                "words": sorted({row["word"] for row in generated}),
                "contexts": sorted({row["context"] for row in generated}),
                "seeds": sorted({row["seed"] for row in generated}),
                "methods": sorted({row["method"] for row in generated}),
                "generation_seconds": sum(row["elapsed_seconds"] for row in generated),
            }
        )
    result = {
        "runs": runs,
        "natural_tts_wavs": sum(run["generated"] for run in runs),
        "exact_input_control_groups": controls,
        "all_exact_controls_identical": all(row["pcm_identical"] for row in controls),
        "lexical_accent_verified": False,
        "manual_mora_alignment_performed": False,
        "auditory_review_performed": False,
        "scope": "Exploratory acoustic changes only; no verified local accent accuracy or naturalness score",
    }
    (args.results / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
