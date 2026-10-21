"""Estimate mora anchors with optional CTC weights; never certify pronunciation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path

import numpy as np

from .analyse import summarise_f0
from .ctc import edit_distance, greedy_text, mora_intervals, viterbi_align
from .probes import pitch_labels

ROOT = Path(__file__).parent
MODEL_REPO = "MahmoudAshraf/mms-300m-1130-forced-aligner"
MODEL_REVISION = "49402e9577b1158620820667c218cd494cc44486"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def expected_units(row: dict, specification: dict) -> list[dict]:
    try:
        prefix = specification["prefixes"][str(row["context"])]
        target = specification["words"][row["word"]]
    except KeyError as exc:
        raise ValueError("No explicit expected transcript for this case") from exc
    return (
        [unit | {"role": "prefix"} for unit in prefix]
        + [unit | {"role": "target"} for unit in target]
        + [dict(unit) for unit in specification["suffix"]]
    )


def heuristic_shape(
    times, f0, intervals: list[dict], nucleus: int | None, *, boundary_shift: float = 0.0
) -> dict:
    selected = [dict(unit) for unit in intervals if unit["role"] in {"target", "particle"}]
    for unit in selected:
        unit["start"] += boundary_shift
        unit["end"] += boundary_shift
    if not selected or selected[0]["start"] < 0 or selected[-1]["end"] > times[-1] + 0.005:
        return {"agreement": None, "reason": "boundary outside audio"}
    summary = summarise_f0(times, f0, selected)
    medians = [unit["median_f0_hz"] for unit in summary["moras"]]
    if nucleus is None:
        return {"agreement": None, "reason": "no requested nucleus", "mora_f0": summary["moras"]}
    if any(value is None for value in medians):
        return {
            "agreement": None,
            "reason": "insufficient voiced frames",
            "mora_f0": summary["moras"],
        }
    labels = pitch_labels(len(selected) - 1, nucleus, particle=True)
    drops = summary["adjacent_drops_semitones"]
    agrees = []
    for left, right, drop in zip(labels, labels[1:], drops, strict=False):
        agrees.append(
            drop >= 1.5
            if left == "H" and right == "L"
            else drop <= -1.5
            if left == "L" and right == "H"
            else abs(drop) <= 3.0
        )
    return {
        "agreement": all(agrees),
        "expected_idealised_labels": labels,
        "adjacent_drops_semitones": drops,
        "mora_f0": summary["moras"],
        "note": "Uncalibrated F0 shape heuristic, not a lexical-accent verdict",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Local pinned model assets; weights are research-only CC-BY-NC-4.0",
    )
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must not exist")
    import soundfile as sf
    import torch
    from scipy.signal import resample_poly
    from transformers import AutoFeatureExtractor, Wav2Vec2ForCTC

    torch.set_num_threads(4)
    model = Wav2Vec2ForCTC.from_pretrained(args.model, local_files_only=True).eval()
    extractor = AutoFeatureExtractor.from_pretrained(args.model, local_files_only=True)
    vocabulary = json.loads((args.model / "vocab.json").read_text())
    specification = json.loads((ROOT / "alignment-transcripts.json").read_text(encoding="utf-8"))
    if (
        vocabulary.get("<blank>") != 0
        or model.config.pad_token_id != 0
        or int(extractor.sampling_rate) != 16000
    ):
        raise ValueError("Unexpected model blank/sample-rate configuration")
    stride, receptive = 1, 1
    for kernel, step in zip(model.config.conv_kernel, model.config.conv_stride, strict=True):
        receptive += (kernel - 1) * stride
        stride *= step
    report = {
        "schema": "provisional-ctc-mora-alignment-v1",
        "model_repo": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "model_licence": "CC-BY-NC-4.0 (research tool only; weights not redistributed)",
        "model_assets_sha256": {
            p.name: sha256(p) for p in sorted(args.model.iterdir()) if p.is_file()
        },
        "packages": {
            p: importlib.metadata.version(p)
            for p in ("torch", "transformers", "numpy", "scipy", "soundfile")
        },
        "generation_source": "withheld-pre-publication-commit",
        "alignment_code_sha256": {
            p.name: sha256(p)
            for p in [Path(__file__), ROOT / "ctc.py", ROOT / "alignment-transcripts.json"]
        },
        "policy": {
            "boundary_rule": "midpoints between neighbouring mora character anchors; not phoneme ground truth",
            "max_greedy_character_error_rate": 0.25,
            "min_mora_posterior": 0.2,
            "required_transition_semitones": 1.5,
            "same_level_tolerance_semitones": 3.0,
            "sensitivity_shifts_seconds": [-0.01, 0.0, 0.01],
        },
        "accent_verified": False,
        "human_listening_performed": False,
        "rows": [],
    }

    def save():
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(args.output)

    for manifest_path in sorted(args.results.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for row in manifest["rows"]:
            if row["status"] != "generated":
                continue
            wav = manifest_path.parent / row["wav"]
            if sha256(wav) != row["wav_sha256"]:
                raise ValueError("WAV integrity failure")
            units = expected_units(row, specification)
            transcript = "".join(unit["latin"] for unit in units)
            audio, sr = sf.read(wav, dtype="float64", always_2d=True)
            audio = audio.mean(axis=1)
            common = math.gcd(sr, 16000)
            samples = resample_poly(audio, 16000 // common, sr // common).astype(np.float32)
            inputs = extractor(samples, sampling_rate=16000, return_tensors="pt")
            with torch.inference_mode():
                log_probs = model(**inputs).logits[0].log_softmax(dim=-1).cpu().numpy()
            spans = viterbi_align(log_probs, [vocabulary[char] for char in transcript])
            intervals = mora_intervals(
                spans,
                units,
                stride_seconds=stride / 16000,
                centre_offset_seconds=(receptive - 1) / 32000,
                duration=len(audio) / sr,
            )
            greedy = greedy_text(log_probs, vocabulary)
            error_rate = edit_distance(transcript, greedy) / len(transcript)
            with wav.with_name(row["id"] + "-f0.csv").open(encoding="utf-8") as handle:
                series = list(csv.DictReader(handle))
            times = np.array([float(sample["time_seconds"]) for sample in series])
            f0 = np.array([float(sample["f0_hz"]) for sample in series])
            relevant = [unit for unit in intervals if unit["role"] in {"target", "particle"}]
            warnings = []
            if error_rate > 0.25:
                warnings.append("greedy transcript disagreement exceeds exploratory threshold")
            if min(unit["mean_posterior"] for unit in relevant) < 0.2:
                warnings.append("low target/particle alignment posterior")
            variants = [
                heuristic_shape(times, f0, intervals, row["nucleus"], boundary_shift=shift)
                for shift in (-0.01, 0.0, 0.01)
            ]
            agreement = None if warnings else variants[1]["agreement"]
            stable = len({variant["agreement"] for variant in variants}) == 1
            if not stable:
                warnings.append("heuristic changes under +/-10 ms boundary shift")
                agreement = None
            result = {
                "run": manifest_path.parent.name,
                "id": row["id"],
                "word": row["word"],
                "context": row["context"],
                "seed": row["seed"],
                "method": row["method"],
                "nucleus": row["nucleus"],
                "wav_sha256": row["wav_sha256"],
                "expected_latin": transcript,
                "greedy_latin": greedy,
                "greedy_character_error_rate": error_rate,
                "estimated_moras": intervals,
                "character_anchors": spans,
                "transcript_checked": False,
                "alignment_checked": False,
                "sensitivity": variants,
                "provisional_shape_agreement": agreement,
                "boundary_shift_stable": stable,
                "warnings": warnings,
                "accent_verified": False,
            }
            report["rows"].append(result)
            save()
            print(row["id"], f"CER={error_rate:.3f}", f"shape={agreement}", flush=True)
    report["status"] = "estimated_only"
    save()


if __name__ == "__main__":
    main()
