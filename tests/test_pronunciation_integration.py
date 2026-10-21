"""CPU integration: real tiny random-weight model/samplers, offline test codec.

These tests do not establish pronunciation quality with released checkpoints.
Only model asset loading, the tokenizer backend and audio codec are substituted.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

import infer
from irodori_tts.config import ModelConfig
from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey, SamplingRequest
from irodori_tts.model import TextToLatentRFDiT
from irodori_tts.pronunciation_dictionary import PronunciationDictionary
from irodori_tts.tokenizer import PretrainedTextTokenizer


class CharacterBackend:
    """Offline character IDs, exercised through the real tokenizer wrapper."""

    pad_token_id = 0
    bos_token_id = 1

    def __init__(self):
        self.calls = []

    def __call__(self, texts, *, max_length, **kwargs):
        self.calls.append(list(texts))
        ids = torch.zeros((len(texts), max_length), dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, text in enumerate(texts):
            values = [2 + ord(char) % 61 for char in text[:max_length]]
            ids[i, : len(values)] = torch.tensor(values, dtype=torch.long)
            mask[i, : len(values)] = 1
        return {"input_ids": ids, "attention_mask": mask}


class OfflineCodec:
    sample_rate = 100
    model = SimpleNamespace(hop_length=10)

    def decode_latent(self, latent):
        return latent.mean(dim=-1).unsqueeze(1).repeat_interleave(10, dim=-1)


def tiny_runtime(*, flow="rf_velocity", duration=False):
    cfg = ModelConfig(
        flow_parameterization=flow,
        latent_dim=4,
        latent_patch_size=2,
        model_dim=32,
        num_layers=1,
        num_heads=2,
        mlp_ratio=2,
        text_vocab_size=64,
        text_dim=32,
        text_layers=1,
        text_heads=2,
        speaker_dim=32,
        speaker_layers=1,
        speaker_heads=2,
        timestep_embed_dim=32,
        adaln_rank=8,
        use_caption_condition=True,
        use_speaker_condition=True,
        use_duration_predictor=duration,
        duration_hidden_dim=32,
        duration_layers=1,
        duration_attention_heads=2,
        duration_dropout=0,
    )
    with torch.random.fork_rng():
        torch.manual_seed(123)
        model = TextToLatentRFDiT(cfg).eval()
        # A zero output projection would make equality tests vacuous.
        torch.nn.init.normal_(model.out_proj.weight, std=0.1)
    backend, caption_backend = CharacterBackend(), CharacterBackend()
    # Disable an optional watermark backend, even if installed locally.
    with patch(
        "irodori_tts.inference_runtime.SilentCipherWatermarker",
        return_value=SimpleNamespace(ready=False),
    ):
        runtime = InferenceRuntime(
            key=RuntimeKey(checkpoint="unused", model_device="cpu"),
            model_cfg=cfg,
            train_cfg=None,
            model=model,
            tokenizer=PretrainedTextTokenizer(backend),
            caption_tokenizer=PretrainedTextTokenizer(caption_backend),
            codec=OfflineCodec(),
            default_text_max_len=24,
            default_caption_max_len=24,
        )
    return runtime, backend, caption_backend


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_real_samplers_with_dictionary_and_reference(self):
        dictionary = PronunciationDictionary([("猫", "ねこ"), ("caption", "CHANGED")])
        with tempfile.TemporaryDirectory() as directory:
            reference = Path(directory) / "猫.pt"
            torch.save(torch.arange(32, dtype=torch.float32).reshape(8, 4) / 32, reference)
            for flow in ("rf_velocity", "meanflow"):
                for duration in (False, True):
                    for decode_mode in ("batch", "sequential"):
                        with self.subTest(flow=flow, duration=duration, decode_mode=decode_mode):
                            runtime, text_backend, caption_backend = tiny_runtime(
                                flow=flow, duration=duration
                            )
                            request = SamplingRequest(
                                text="「猫！」",
                                caption="caption 猫",
                                ref_latent=str(reference),
                                seed=24,
                                num_steps=2,
                                seconds=None if duration else 0.5,
                                max_seconds=1,
                                num_candidates=2,
                                decode_mode=decode_mode,
                                pronunciation_dictionary=dictionary,
                            )
                            original_reference = reference.read_bytes()
                            result = runtime.synthesize(request)
                            self.assertEqual(text_backend.calls[-1], ["ねこ!"] * 2)
                            self.assertEqual(caption_backend.calls[-1], ["caption 猫"] * 2)
                            self.assertEqual(result.used_seed, 24)
                            self.assertEqual(result.text_processing.original_text, "「猫！」")
                            self.assertEqual(result.text_processing.normalized_text, "猫!")
                            self.assertEqual(result.text_processing.spoken_text, "ねこ!")
                            self.assertEqual(reference.read_bytes(), original_reference)
                            self.assertEqual(request.text, "「猫！」")
                            self.assertEqual(request.ref_latent, str(reference))
                            self.assertEqual(len(result.audios), 2)
                            for audio in result.audios:
                                self.assertTrue(torch.isfinite(audio).all())
                                self.assertGreater(audio.numel(), 0)
                            # Equivalent explicit reading must be bit-identical, including seed,
                            # duration, caption and reference conditioning.
                            baseline = runtime.synthesize(
                                replace(request, text="ねこ!", pronunciation_dictionary=None)
                            )
                            for actual, expected in zip(
                                result.audios, baseline.audios, strict=True
                            ):
                                self.assertTrue(torch.equal(actual, expected))
                            self.assertEqual(result.messages, baseline.messages)
                            # Reusing the runtime must not retain the previous dictionary.
                            runtime.synthesize(replace(request, pronunciation_dictionary=None))
                            self.assertEqual(text_backend.calls[-1], ["猫!"] * 2)

    def test_reference_waveform_is_not_modified(self):
        runtime, _, _ = tiny_runtime()
        waveform = torch.linspace(-0.5, 0.5, 100).reshape(1, -1)
        before = waveform.clone()
        dictionary = PronunciationDictionary([("猫", "ねこ"), ("reference", "changed")])
        encoded = torch.arange(32, dtype=torch.float32).reshape(1, 8, 4) / 32
        with (
            patch(
                "irodori_tts.inference_runtime.torchaudio.load", return_value=(waveform, 100)
            ) as load,
            patch.object(
                runtime.codec, "encode_waveform", create=True, return_value=encoded
            ) as encode,
        ):
            request = SamplingRequest(
                text="猫",
                caption="猫",
                ref_wav="reference猫.wav",
                seed=12,
                seconds=0.5,
                num_steps=2,
                ref_normalize_db=-12.0,
                ref_ensure_max=False,
                pronunciation_dictionary=dictionary,
            )
            actual = runtime.synthesize(request)
            first_call = encode.call_args
            expected = runtime.synthesize(
                replace(request, text="ねこ", pronunciation_dictionary=None)
            )
        self.assertEqual(load.call_args.args, ("reference猫.wav",))
        self.assertEqual(
            first_call.kwargs, {"sample_rate": 100, "normalize_db": -12.0, "ensure_max": False}
        )
        self.assertTrue(torch.equal(first_call.args[0], waveform.unsqueeze(0)))
        self.assertTrue(torch.equal(waveform, before))
        self.assertTrue(torch.equal(actual.audio, expected.audio))

    def test_fixture_outputs_depend_on_conditioning_and_seed(self):
        runtime, _, _ = tiny_runtime()
        request = SamplingRequest(
            text="猫", caption="quiet", no_ref=True, seconds=0.5, num_steps=2, seed=23
        )
        baseline = runtime.synthesize(request).audio
        for changed in (
            replace(request, text="犬"),
            replace(request, caption="loud"),
            replace(request, seed=24),
        ):
            with self.subTest(changed=changed):
                self.assertFalse(torch.equal(baseline, runtime.synthesize(changed).audio))

    def test_dictionary_does_not_consume_random_state(self):
        with torch.random.fork_rng():
            torch.manual_seed(12)
            before = torch.get_rng_state().clone()
            PronunciationDictionary([("猫", "ねこ")]).apply("猫")
            self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_empty_text_fails_before_tokenization(self):
        runtime, backend, _ = tiny_runtime()
        with self.assertRaisesRegex(ValueError, "empty"):
            runtime.synthesize(
                SamplingRequest(
                    text="\t[n]", no_ref=True, pronunciation_dictionary=PronunciationDictionary([])
                )
            )
        self.assertEqual(backend.calls, [])

    def test_cli_applies_dictionary_and_reports_text(self):
        runtime, backend, caption_backend = tiny_runtime()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dictionary.json"
            path.write_text(
                json.dumps({"version": 1, "entries": [{"surface": "猫", "reading": "ねこ"}]}),
                encoding="utf-8",
            )
            argv = [
                "infer.py",
                "--checkpoint",
                "unused",
                "--text",
                "猫",
                "--caption",
                "猫 caption",
                "--no-ref",
                "--seed",
                "42",
                "--seconds",
                "0.5",
                "--num-steps",
                "2",
                "--pronunciation-dictionary",
                str(path),
                "--show-text",
            ]
            output = io.StringIO()
            with (
                patch.object(sys, "argv", argv),
                patch.object(infer, "_resolve_checkpoint_path", return_value="unused"),
                patch.object(infer.InferenceRuntime, "from_key", return_value=runtime),
                patch.object(
                    infer, "save_wav", return_value=Path(directory) / "output.wav"
                ) as save,
                contextlib.redirect_stdout(output),
            ):
                infer.main()
            self.assertEqual(backend.calls[-1], ["ねこ"])
            self.assertEqual(caption_backend.calls[-1], ["猫 caption"])
            self.assertIn("[text] original: '猫'", output.getvalue())
            self.assertIn("[text] spoken: 'ねこ'", output.getvalue())
            self.assertIn("used_seed: 42", output.getvalue())
            self.assertEqual(save.call_count, 1)
            self.assertTrue(torch.isfinite(save.call_args.args[1]).all())

    def test_cli_bad_dictionary_fails_before_model_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            for payload in (None, "{", "{}", "\ud800"):
                if payload is not None:
                    path.write_bytes(payload.encode("utf-8", errors="surrogatepass"))
                argv = [
                    "infer.py",
                    "--checkpoint",
                    "unused",
                    "--text",
                    "猫",
                    "--pronunciation-dictionary",
                    str(path),
                ]
                with (
                    self.subTest(payload=payload),
                    patch.object(sys, "argv", argv),
                    patch.object(infer, "_resolve_checkpoint_path") as load,
                    contextlib.redirect_stderr(io.StringIO()),
                    self.assertRaises(SystemExit) as caught,
                ):
                    infer.main()
                self.assertEqual(caught.exception.code, 2)
                load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
