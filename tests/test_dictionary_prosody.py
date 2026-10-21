"""Research contracts, not evidence of released-model accent control."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

import numpy as np

from experiments.dictionary_prosody.analyse import semitone_drop, summarise_f0
from experiments.dictionary_prosody.postprocess import modify_f0
from experiments.dictionary_prosody.probes import (
    METHODS,
    pitch_labels,
    prepare_probe,
    render_probe,
    split_mora,
    validate_entry,
)
from experiments.dictionary_prosody.run import make_plan
from irodori_tts.pronunciation_dictionary import PronunciationDictionary, prepare_text


def entry(reading="あめ", nucleus=1):
    return {"surface": "雨", "reading": reading, "accent": {"dialect": "tokyo", "nucleus": nucleus}}


class ProsodyTests(unittest.TestCase):
    def test_moras(self):
        for text, expected in {
            "あめ": ("あ", "め"),
            "きょう": ("きょ", "う"),
            "キャット": ("きゃ", "っ", "と"),
            "コーヒー": ("こ", "ー", "ひ", "ー"),
            "しんぶん": ("し", "ん", "ぶ", "ん"),
            "ﾃｨｰ": ("てぃ", "ー"),
            "ヴァイオリン": ("ゔぁ", "い", "お", "り", "ん"),
        }.items():
            with self.subTest(text=text):
                self.assertEqual(split_mora(text), expected)
        for text in (
            "",
            "ゃ",
            "ー",
            "あゃ",
            "きゃゅ",
            "漢字",
            "あ め",
            "あ\nめ",
            "あ1",
            "ゕ",
            "ゝ",
            "abc",
        ):
            with self.subTest(text=text), self.assertRaises(ValueError):
                split_mora(text)

    def test_all_nucleus_types_and_particle(self):
        for nucleus, isolated, particle in [
            (0, "LHH", "LHHH"),
            (1, "HLL", "HLLL"),
            (2, "LHL", "LHLL"),
            (3, "LHH", "LHHL"),
        ]:
            with self.subTest(nucleus=nucleus):
                validate_entry(entry("こころ", nucleus))
                self.assertEqual(pitch_labels(3, nucleus), isolated)
                self.assertEqual(pitch_labels(3, nucleus, particle=True), particle)
        self.assertEqual(pitch_labels(2, 0), pitch_labels(2, 2))
        self.assertNotEqual(pitch_labels(2, 0, particle=True), pitch_labels(2, 2, particle=True))

    def test_invalid_accent(self):
        for value in (-1, 3, True, 1.0, None, "1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_entry(entry(nucleus=value))
        for accent in (
            None,
            {},
            {"dialect": "other", "nucleus": 1},
            {"dialect": "tokyo", "nucleus": 1, "ignored": True},
        ):
            bad = entry()
            bad["accent"] = accent
            with self.subTest(accent=accent), self.assertRaises(ValueError):
                validate_entry(bad)
        for count, nucleus in [(0, 0), (2, 3), (True, 1), (2, True)]:
            with self.assertRaises(ValueError):
                pitch_labels(count, nucleus)

    def test_unvalidated_probes_are_explicit(self):
        for method in METHODS:
            probe = prepare_probe("雨がある。", entry(), method, "base")
            self.assertFalse(probe["validated_accent_control"])
            self.assertTrue(probe["experimental_only"])
            self.assertEqual(
                probe["caption"],
                "base" if method != "caption" else "base " + render_probe(entry(), method)[1],
            )
        with self.assertRaises(ValueError):
            render_probe(entry(), "auto")
        self.assertEqual(render_probe(entry(), "downstep")[0], "あꜜめ")
        self.assertEqual(render_probe(entry(nucleus=0), "arrows")[0], "あ↑め")

    def test_v1_is_unchanged_and_v2_is_not_silently_accepted(self):
        dictionary = PronunciationDictionary.from_dict(
            {
                "version": 1,
                "entries": [
                    {"surface": "禁書目録", "reading": "インデックス"},
                    {"surface": "インデックス", "reading": "再置換禁止"},
                ],
            }
        )
        self.assertEqual(prepare_text("禁書目録", dictionary).spoken_text, "インデックス")
        with self.assertRaises(ValueError):
            PronunciationDictionary.from_dict({"version": 2, "entries": [entry()]})
        self.assertEqual(prepare_text("ＡＩ").spoken_text, "AI")

    def test_plan_has_fixed_controls_and_no_claim(self):
        config = json.loads(
            (Path(__file__).parents[1] / "experiments/dictionary_prosody/cases.json").read_text(
                encoding="utf-8"
            )
        )
        rows = make_plan(
            config, words=["ame"], contexts=[0], seeds=[0], methods=["caption"], repeats=2
        )
        self.assertEqual(len(rows), 8)
        self.assertEqual(len({r["id"] for r in rows}), len(rows))
        self.assertTrue(all(r["status"] == "planned" and r["seed"] == 0 for r in rows))
        self.assertEqual(rows[0]["method"], "original")
        self.assertEqual(rows[2]["method"], "reading")
        for row in rows[4:]:
            self.assertEqual(row["spoken_text"], rows[2]["spoken_text"])
            self.assertNotEqual(row["caption"], rows[2]["caption"])

    def test_f0_without_alignment_is_not_accent_evidence(self):
        times = np.arange(100) * 0.005
        f0 = np.r_[np.full(50, 200.0), np.full(50, 100.0)]
        summary = summarise_f0(times, f0)
        self.assertFalse(summary["accent_verified"])
        self.assertEqual(summary["mora_alignment"], "unavailable")
        result = summarise_f0(
            times,
            f0,
            [
                {"label": "a", "start": 0.0, "end": 0.25},
                {"label": "me", "start": 0.25, "end": 0.49},
            ],
        )
        self.assertAlmostEqual(result["adjacent_drops_semitones"][0], 12.0)
        self.assertFalse(result["accent_verified"])
        result = summarise_f0(times, np.zeros(100))
        self.assertIsNone(result["median_f0_hz"])
        for before, after in [(0, 100), (100, float("nan"))]:
            with self.assertRaises(ValueError):
                semitone_drop(before, after)

    def test_invalid_alignment(self):
        times = np.arange(100) * 0.005
        for intervals in [
            [{"label": "a", "start": -0.1, "end": 0.2}],
            [{"label": "a", "start": 0.3, "end": 0.2}],
            [{"label": "a", "start": 0.0, "end": 2.0}],
        ]:
            with self.assertRaises(ValueError):
                summarise_f0(times, np.ones(100), intervals)

    def test_saved_generated_evidence_hashes(self):
        root = Path(__file__).parents[1] / "experiments/dictionary_prosody/results"
        manifests = list(root.glob("*/manifest.json"))
        self.assertTrue(manifests)
        import soundfile as sf

        for path in manifests:
            manifest = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "generated")
            analysis = json.loads(path.with_name("analysis.json").read_text(encoding="utf-8"))
            analysed = {row["id"]: row for row in analysis["results"]}
            for row in manifest["rows"]:
                self.assertEqual(row["status"], "generated")
                wav = path.parent / row["wav"]
                self.assertEqual(hashlib.sha256(wav.read_bytes()).hexdigest(), row["wav_sha256"])
                audio, _ = sf.read(wav, dtype="float32", always_2d=True)
                self.assertEqual(
                    hashlib.sha256(audio.astype("<f4").tobytes()).hexdigest(),
                    analysed[row["id"]]["pcm_sha256"],
                )
                self.assertFalse(analysed[row["id"]]["accent_verified"])

    def test_saved_negative_controls_are_sample_identical(self):
        root = Path(__file__).parents[1] / "experiments/dictionary_prosody/results"
        comparisons = 0
        for path in root.glob("*/analysis.json"):
            document = json.loads(path.read_text(encoding="utf-8"))
            rows = document.get("results", [])
            for row in rows:
                if row["method"] == "downstep" and row["nucleus"] == 0:
                    baseline_id = row["id"].replace("downstep-n0", "reading-nna")
                    baseline = next(item for item in rows if item["id"] == baseline_id)
                    self.assertEqual(row["pcm_sha256"], baseline["pcm_sha256"])
                    comparisons += 1
        self.assertGreater(comparisons, 0)

    def test_postprocessing_keeps_unvoiced_and_outside_window(self):
        times = np.arange(101) * 0.005
        f0 = np.full(101, 200.0)
        f0[30:40] = 0
        changed = modify_f0(times, f0, [{"start": 0.1, "end": 0.4, "semitones": -4.0}])
        self.assertTrue(np.array_equal(changed[f0 == 0], f0[f0 == 0]))
        self.assertTrue(np.array_equal(changed[times < 0.1], f0[times < 0.1]))
        self.assertLess(changed[50], f0[50])
        self.assertEqual(f0[50], 200.0)
        with self.assertRaises(ValueError):
            modify_f0(times, f0, [{"start": 0.1, "end": 0.4, "semitones": 7}])


if __name__ == "__main__":
    unittest.main()
