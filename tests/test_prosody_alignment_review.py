"""Independent path optimality and evidence provenance tests for follow-up tools."""

from __future__ import annotations

import copy
import itertools
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.dictionary_prosody.align import expected_units, heuristic_shape
from experiments.dictionary_prosody.analyse import validate_annotation
from experiments.dictionary_prosody.ctc import (
    edit_distance,
    greedy_text,
    mora_intervals,
    viterbi_align,
)
from experiments.dictionary_prosody.review import build_kit, import_review

ROOT = Path(__file__).parents[1] / "experiments/dictionary_prosody"


def collapse(path):
    result = []
    previous = None
    for token in path:
        if token != previous and token != 0:
            result.append(token)
        previous = token
    return result


class CTCTests(unittest.TestCase):
    def test_viterbi_matches_exhaustive_path_search(self):
        rng = np.random.default_rng(42)
        for target in ([1], [1, 2], [1, 1], [2, 1, 2]):
            for frames in (4, 5):
                scores = rng.normal(size=(frames, 3))
                scores -= np.log(np.exp(scores).sum(axis=1, keepdims=True))
                spans = viterbi_align(scores, list(target))
                actual = [0] * frames
                for span in spans:
                    actual[span["start_frame"] : span["end_frame"]] = [span["token"]] * (
                        span["end_frame"] - span["start_frame"]
                    )
                self.assertEqual(collapse(actual), list(target))
                actual_score = sum(scores[t, token] for t, token in enumerate(actual))
                expected = max(
                    sum(scores[t, token] for t, token in enumerate(path))
                    for path in itertools.product(range(3), repeat=frames)
                    if collapse(path) == list(target)
                )
                self.assertAlmostEqual(actual_score, expected)

    def test_repeat_requires_blank(self):
        with self.assertRaises(ValueError):
            viterbi_align(np.log(np.full((2, 2), 0.5)), [1, 1])
        spans = viterbi_align(np.log([[0.01, 0.99], [0.99, 0.01], [0.01, 0.99]]), [1, 1])
        self.assertEqual([s["start_frame"] for s in spans], [0, 2])

    def test_invalid_emissions_and_targets(self):
        for scores, targets in [
            ([], [1]),
            ([[float("nan"), 0]], [1]),
            ([[0, 0]], [1]),
            ([[0.2, 0.8]], [1]),
            (np.log([[0.5, 0.5]]), []),
            (np.log([[0.5, 0.5]]), [0]),
            (np.log([[0.5, 0.5]]), [True]),
            (np.log([[0.5, 0.5]]), [2]),
        ]:
            with self.subTest(scores=scores, targets=targets), self.assertRaises(ValueError):
                viterbi_align(scores, targets)

    def test_greedy_and_edit_distance(self):
        scores = np.log([[0.1, 0.9], [0.1, 0.9], [0.9, 0.1], [0.1, 0.9]])
        self.assertEqual(greedy_text(scores, {"_": 0, "a": 1}), "aa")
        self.assertEqual(edit_distance("amegaaru", "anegaaru"), 1)
        self.assertEqual(edit_distance("", "abc"), 3)

    def test_anchor_midpoints_and_validation(self):
        spans = [
            {"start_frame": 1, "end_frame": 2, "mean_posterior": 0.9},
            {"start_frame": 3, "end_frame": 4, "mean_posterior": 0.8},
        ]
        units = [{"label": "あ", "latin": "a"}, {"label": "め", "latin": "m"}]
        result = mora_intervals(
            spans, units, stride_seconds=0.02, centre_offset_seconds=0.0125, duration=0.1
        )
        self.assertAlmostEqual(result[0]["end"], result[1]["start"])
        self.assertAlmostEqual(result[0]["end"], 0.0525)
        with self.assertRaises(ValueError):
            mora_intervals(
                spans, units[:1], stride_seconds=0.02, centre_offset_seconds=0, duration=0.1
            )
        broken = copy.deepcopy(spans)
        broken[0]["start_frame"] = -1
        with self.assertRaises(ValueError):
            mora_intervals(
                broken, units, stride_seconds=0.02, centre_offset_seconds=0, duration=0.1
            )

    def test_idealised_shape_is_not_verified_accent(self):
        times = np.arange(60) * 0.005
        intervals = [
            {"label": "あ", "role": "target", "start": 0.0, "end": 0.1},
            {"label": "め", "role": "target", "start": 0.1, "end": 0.2},
            {"label": "が", "role": "particle", "start": 0.2, "end": 0.29},
        ]
        rising = np.r_[np.full(20, 100.0), np.full(40, 150.0)]
        self.assertTrue(heuristic_shape(times, rising, intervals, 0)["agreement"])
        self.assertFalse(heuristic_shape(times, rising, intervals, 1)["agreement"])
        self.assertIsNone(heuristic_shape(times, np.zeros(60), intervals, 0)["agreement"])
        self.assertIsNone(heuristic_shape(times, rising, intervals, None)["agreement"])
        self.assertIsNone(
            heuristic_shape(times, rising, intervals, 0, boundary_shift=-0.1)["agreement"]
        )

    def test_explicit_transcript_scope(self):
        specification = json.loads((ROOT / "alignment-transcripts.json").read_text())
        result = expected_units({"word": "ame", "context": 1}, specification)
        self.assertEqual("".join(unit["latin"] for unit in result), "kokoniamegaaru")
        self.assertEqual([unit["role"] for unit in result].count("target"), 2)
        with self.assertRaises(ValueError):
            expected_units({"word": "new", "context": 1}, specification)


class ReviewTests(unittest.TestCase):
    def fixture(self):
        key = {
            "fingerprint": "abc",
            "clips": {
                "clip001": {
                    "id": "actual",
                    "duration": 2.0,
                    "target_moras": 2,
                    "mora_labels": ["あ", "め"],
                    "wav_sha256": "hash",
                }
            },
        }
        record = {
            "code": "clip001",
            "transcript_checked": False,
            "alignment_checked": False,
            "uncertain": True,
            "perceived_nucleus": None,
            "naturalness": None,
            "moras": [
                {"label": "あ", "start": 0.1, "end": 0.3},
                {"label": "め", "start": 0.3, "end": 0.5},
            ],
        }
        return key, {
            "schema": "blind-prosody-review-v1",
            "fingerprint": "abc",
            "reviewer": "test-only reviewer",
            "reviews": [record],
        }

    def test_estimates_never_promote_themselves(self):
        key, draft = self.fixture()
        result = import_review(key, draft)
        self.assertEqual(result["annotations"], {})
        self.assertFalse(result["accent_verified"])
        draft["reviews"][0].update(
            transcript_checked=True, alignment_checked=True, perceived_nucleus=1
        )
        result = import_review(key, draft)
        self.assertEqual(result["annotations"]["actual"]["wav_sha256"], "hash")
        self.assertFalse(result["accent_verified"])

    def test_bad_provenance_or_acknowledgement_fails(self):
        mutations = [
            lambda d: d.update(reviewer=""),
            lambda d: d.update(fingerprint="other"),
            lambda d: d["reviews"][0].update(transcript_checked="true"),
            lambda d: d["reviews"][0].update(alignment_checked=True),
            lambda d: d["reviews"][0].update(perceived_nucleus=True),
            lambda d: d["reviews"][0].update(perceived_nucleus=3),
            lambda d: d["reviews"][0].update(naturalness=9),
            lambda d: d["reviews"][0].update(moras=[]),
            lambda d: d["reviews"][0]["moras"][0].update(end=float("nan")),
            lambda d: d["reviews"][0].update(code="other"),
            lambda d: d["reviews"].append(copy.deepcopy(d["reviews"][0])),
        ]
        for mutate in mutations:
            key, draft = self.fixture()
            mutate(draft)
            with self.assertRaises(ValueError):
                import_review(key, draft)

    def test_generated_kit_is_blind_and_asset_bound(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "kit"
            result = build_kit(ROOT / "results", output)
            bundle = json.loads((output / "bundle.json").read_text())
            key = json.loads((output / "review-key.json").read_text())
            self.assertEqual(result["clips"], 40)
            self.assertEqual(bundle["fingerprint"], key["fingerprint"])
            for clip in bundle["clips"]:
                self.assertTrue((output / clip["audio"]).is_file())
                self.assertTrue((output / clip["baseline_audio"]).is_file())
                self.assertNotIn("method", clip)
                self.assertNotIn("nucleus", clip)
                self.assertNotIn("id", clip)
            with self.assertRaises(ValueError):
                build_kit(ROOT / "results", output)

    def test_annotation_is_bound_to_reviewed_audio(self):
        valid = {
            "reviewer": "test",
            "transcript_checked": True,
            "moras": [{"label": "あ", "start": 0.1, "end": 0.2}],
            "wav_sha256": "one",
        }
        validate_annotation(valid, "one")
        for bad in (
            valid | {"wav_sha256": "other"},
            valid | {"reviewer": " "},
            valid | {"transcript_checked": False},
            valid | {"moras": []},
        ):
            with self.assertRaises(ValueError):
                validate_annotation(bad, "one")

    def test_saved_machine_results_remain_unreviewed(self):
        data = json.loads((ROOT / "results/automatic-alignment.json").read_text())
        self.assertEqual(len(data["rows"]), 40)
        self.assertFalse(data["accent_verified"])
        for row in data["rows"]:
            self.assertFalse(row["accent_verified"])
            self.assertFalse(row["alignment_checked"])
            self.assertFalse(row["transcript_checked"])
            starts = [m["start"] for m in row["estimated_moras"]]
            self.assertEqual(starts, sorted(starts))


if __name__ == "__main__":
    unittest.main()
