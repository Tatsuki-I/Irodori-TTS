"""Run labelled input probes with the unchanged public inference runtime."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

from irodori_tts.pronunciation_dictionary import PronunciationDictionary, prepare_text

from .probes import METHODS, prepare_probe, render_probe

ROOT = Path(__file__).parent


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def make_plan(
    config: dict,
    *,
    words: list[str],
    contexts: list[int],
    seeds: list[int],
    methods: list[str],
    repeats: int = 1,
) -> list[dict]:
    rows = []
    for word in config["words"]:
        if word["id"] not in words:
            continue
        for context in contexts:
            text = config["contexts"][context].format(surface=word["surface"])
            for seed in seeds:
                variants = [("original", None), ("reading", None)] + [
                    (method, nucleus) for method in methods for nucleus in word["nuclei"]
                ]
                for method, nucleus in variants:
                    entry = {"surface": word["surface"], "reading": word["reading"]}
                    if nucleus is not None:
                        entry["accent"] = {"dialect": "tokyo", "nucleus": nucleus}
                        probe = prepare_probe(text, entry, method, config["caption"])
                        replacement, _ = render_probe(entry, method)
                    else:
                        dictionary = (
                            None
                            if method == "original"
                            else PronunciationDictionary([(word["surface"], word["reading"])])
                        )
                        probe = asdict(prepare_text(text, dictionary)) | {
                            "caption": config["caption"],
                            "nucleus": None,
                            "method": method,
                        }
                        replacement = None if method == "original" else word["reading"]
                    for repeat in range(repeats):
                        identifier = f"{word['id']}-c{context}-s{seed}-{method}-n{nucleus if nucleus is not None else 'na'}-r{repeat}"
                        rows.append(
                            probe
                            | {
                                "id": identifier,
                                "word": word["id"],
                                "context": context,
                                "seed": seed,
                                "repeat": repeat,
                                "entry": entry,
                                "replacement": replacement,
                                "status": "planned",
                            }
                        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--codec", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--words", nargs="+", default=["ame", "hashi", "kokoro"])
    parser.add_argument("--contexts", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    config = json.loads((ROOT / "cases.json").read_text(encoding="utf-8"))
    if not set(args.words) <= {w["id"] for w in config["words"]} or any(
        c not in range(len(config["contexts"])) for c in args.contexts
    ):
        parser.error("Unknown word/context")
    if args.repeats < 1 or args.steps < 1 or args.seconds <= 0 or args.threads < 1:
        parser.error("repeats, steps, seconds and threads must be positive")
    rows = make_plan(
        config,
        words=args.words,
        contexts=args.contexts,
        seeds=args.seeds,
        methods=args.methods,
        repeats=args.repeats,
    )
    if args.limit is not None:
        if args.limit < 1:
            parser.error("limit must be positive")
        rows = rows[: args.limit]
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema": "dictionary-prosody-run-v1",
        "config": config,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        ),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "torchaudio", "transformers", "numpy", "safetensors", "soundfile")
        },
        "settings": {
            "steps": args.steps,
            "seconds": args.seconds,
            "threads": args.threads,
            "model_precision": "fp32",
            "cfg_text": 3.0,
            "cfg_caption": 3.0,
            "cfg_speaker": 5.0,
            "reference": None if args.reference is None else digest(args.reference),
        },
        "status": "planned",
        "source_sha256": {
            str(p.relative_to(ROOT.parent.parent)): digest(p) for p in sorted(ROOT.glob("*.py"))
        },
        "rows": rows,
    }

    def save():
        path = args.output / "manifest.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(path)

    save()
    if args.plan_only:
        print(f"Saved {len(rows)} planned cases; no audio generated")
        return
    if args.checkpoint is None or args.codec is None:
        parser.error("Audio generation requires local --checkpoint and --codec")
    try:
        import soundfile as sf
        import torch

        from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey, SamplingRequest

        torch.set_num_threads(args.threads)
        torch.use_deterministic_algorithms(True)
        manifest["checkpoint_sha256"] = digest(args.checkpoint)
        manifest["codec_sha256"] = digest(args.codec)
        manifest["tokenizer_sha256"] = {
            p.name: digest(p) for p in sorted((args.checkpoint.parent / "tokenizer").glob("*.json"))
        }
        save()
        runtime = InferenceRuntime.from_key(
            RuntimeKey(
                checkpoint=str(args.checkpoint),
                model_device="cpu",
                codec_device="cpu",
                codec_repo=str(args.codec),
            )
        )
        manifest["model_config"] = asdict(runtime.model_cfg)
        manifest["train_config"] = runtime.train_cfg
        manifest["watermark_enabled"] = runtime.watermarker.ready
        save()
        for row in rows:
            start = time.perf_counter()
            dictionary = (
                None
                if row["replacement"] is None
                else PronunciationDictionary([(row["entry"]["surface"], row["replacement"])])
            )
            request = SamplingRequest(
                text=row["original_text"],
                caption=row["caption"],
                seed=row["seed"],
                no_ref=args.reference is None,
                ref_wav=None if args.reference is None else str(args.reference),
                seconds=args.seconds,
                num_steps=args.steps,
                trim_tail=False,
                pronunciation_dictionary=dictionary,
            )
            row["request_settings"] = {
                key: value
                for key, value in vars(request).items()
                if key != "pronunciation_dictionary"
            }
            save()
            result = runtime.synthesize(request)
            if result.text_processing.spoken_text != row["spoken_text"]:
                raise RuntimeError("Probe and actual runtime text disagree")
            ids, mask = runtime.tokenizer.batch_encode(
                [row["spoken_text"]], max_length=runtime.default_text_max_len
            )
            row["token_ids"] = ids[mask].tolist()
            name = row["id"] + ".wav"
            audio = result.audio.detach().cpu().numpy().T
            sf.write(args.output / name, audio, result.sample_rate, subtype="FLOAT")
            row.update(
                status="generated",
                wav=name,
                wav_sha256=digest(args.output / name),
                pcm_sha256=hashlib.sha256(audio.astype("<f4").tobytes()).hexdigest(),
                sample_rate=result.sample_rate,
                audio_seconds=len(audio) / result.sample_rate,
                elapsed_seconds=time.perf_counter() - start,
                messages=result.messages,
            )
            save()
            print(row["id"], f"{row['elapsed_seconds']:.2f}s", flush=True)
        manifest["status"] = "generated"
        save()
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        save()
        raise


if __name__ == "__main__":
    main()
