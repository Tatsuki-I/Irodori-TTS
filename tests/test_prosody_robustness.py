"""Adversarial F0 screens must keep null controls and uncertainty visible."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from experiments.dictionary_prosody.robustness import boundary_variants, classify, load_f0


class RobustnessTests(unittest.TestCase):
    def test_idealised_patterns(self):
        # Target has three moras and a following particle.
        for drops, nucleus in [([-4, 0, 0], 0), ([4, 0, 0], 1), ([-4, 4, 0], 2), ([-4, 0, 4], 3)]:
            self.assertEqual([n for n in range(4) if classify(drops, n)], [nucleus])

    def test_abstains_on_unvoiced(self):
        self.assertIsNone(classify([None, 4], 1))

    def test_heuristic_can_match_multiple_nuclei(self):
        # Loose equal-level tolerance overlaps the minimum required fall.
        self.assertTrue(classify([-4, 2], 0))
        self.assertTrue(classify([-4, 2], 2))
        self.assertFalse(classify([-4, 2], 0, tolerance=1))

    def test_bad_numeric_inputs(self):
        for drops, nucleus, transition, tolerance in [
            ([float("nan"), 1], 1, 1.5, 3),
            ([1, 2], 1, 0, 3),
            ([1, 2], 1, 1, -1),
            ([1, 2], 1, float("inf"), 3),
            ([1, 2], 3, 1, 3),
        ]:
            with self.assertRaises(ValueError):
                classify(drops, nucleus, transition, tolerance)

    def test_independent_boundaries_are_contiguous_and_ordered(self):
        original = [
            {"label": "あ", "start": 0.1, "end": 0.3},
            {"label": "め", "start": 0.3, "end": 0.5},
        ]
        variants = list(boundary_variants(original, 0.02))
        self.assertEqual(len(variants), 27)
        self.assertEqual(len({v[0]["end"] for v in variants}), 3)
        self.assertTrue(all(v[0]["end"] == v[1]["start"] for v in variants))
        self.assertEqual(original[0]["end"], 0.3)
        self.assertTrue(any(v[0]["start"] == 0.1 and v[0]["end"] != 0.3 for v in variants))

    def test_invalid_boundary_grids(self):
        for intervals, radius in [
            ([], 0.02),
            ([{"start": 0.1, "end": 0.2}], 0),
            ([{"start": 0.1, "end": 0.2}, {"start": 0.3, "end": 0.4}], 0.02),
        ]:
            with self.assertRaises(ValueError):
                list(boundary_variants(intervals, radius))
        small = [{"label": "a", "start": 0.01, "end": 0.02}]
        self.assertTrue(
            all(0 <= v[0]["start"] < v[0]["end"] for v in boundary_variants(small, 0.04))
        )

    def test_f0_input_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "f0.csv"

            def write(rows):
                with path.open("w", newline="") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(["time_seconds", "f0_hz", "voiced"])
                    writer.writerows(rows)

            write([(0, 100, 1), (0.005, 0, 0)])
            self.assertEqual(load_f0(path, 0.01)[1].tolist(), [100, 0])
            for rows in [[(0, 100, 0)], [(0, -1, 0)], [(0, 100, 1), (0, 100, 1)], [(0.2, 100, 1)]]:
                write(rows)
                with self.assertRaises(ValueError):
                    load_f0(path, 0.01)


class SavedEvidenceTests(unittest.TestCase):
    def setUp(self):
        import json
        import shutil

        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        source = Path(__file__).resolve().parents[1] / "experiments/dictionary_prosody/results"
        ids = {"ame-c0-s0-reading-nna-r0", "ame-c0-s0-downstep-n0-r0"}
        run = self.root / "pilot"
        run.mkdir()
        for filename, field in [("manifest.json", "rows"), ("analysis.json", "results")]:
            value = json.loads((source / "pilot" / filename).read_text())
            value[field] = [r for r in value[field] if r["id"] in ids]
            (run / filename).write_text(json.dumps(value))
        self.alignment = self.root / "automatic-alignment.json"
        value = json.loads((source / "automatic-alignment.json").read_text())
        value["rows"] = [r for r in value["rows"] if r["id"] in ids]
        self.alignment.write_text(json.dumps(value))
        for clip in ids:
            for suffix in (".wav", "-f0.csv"):
                shutil.copyfile(source / "pilot" / (clip + suffix), run / (clip + suffix))

    def test_real_evidence_and_matched_null_control(self):
        from experiments.dictionary_prosody.robustness import audit

        result = audit(self.root, self.alignment)
        self.assertEqual(result["summary"]["clips"], 2)
        self.assertEqual(len(result["matched_control_comparisons"]), 1)
        self.assertTrue(result["exact_condition_groups"][0]["pcm_identical"])
        self.assertFalse(result["accent_verified"])
        pair = result["matched_control_comparisons"][0]
        self.assertEqual(pair["intervention_raw_match"], pair["control_raw_match"])

    def test_refuses_audio_tampering(self):
        from experiments.dictionary_prosody.robustness import audit

        wav = next((self.root / "pilot").glob("*.wav"))
        data = bytearray(wav.read_bytes())
        data[-1] ^= 1
        wav.write_bytes(data)
        with self.assertRaisesRegex(ValueError, "audio integrity"):
            audit(self.root, self.alignment)

    def test_refuses_alignment_metadata_and_duplicate_coverage(self):
        import json

        from experiments.dictionary_prosody.robustness import audit

        value = json.loads(self.alignment.read_text())
        value["rows"].append(value["rows"][0])
        self.alignment.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "coverage"):
            audit(self.root, self.alignment)
        value["rows"].pop()
        value["rows"][0]["seed"] = 999
        self.alignment.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "metadata mismatch"):
            audit(self.root, self.alignment)


class ExtractorComparisonTests(unittest.TestCase):
    def test_octave_errors_voicing_and_absence(self):
        import numpy as np

        from experiments.dictionary_prosody.crosscheck import compare_series

        result = compare_series(np.array([100.0, 0.0, 100.0]), np.array([200.0, 100.0, 100.0]))
        self.assertEqual(result["octave_scale_disagreement_frames"], 1)
        self.assertEqual(result["joint_voiced_frames"], 2)
        self.assertAlmostEqual(result["voicing_disagreement_fraction"], 1 / 3)
        self.assertEqual(result["median_absolute_difference_semitones"], 6)
        self.assertIsNone(
            compare_series(np.array([0.0]), np.array([100.0]))[
                "median_absolute_difference_semitones"
            ]
        )
        for a, b in [([1], [1, 2]), ([float("nan")], [1]), ([-1], [0])]:
            with self.assertRaises(ValueError):
                compare_series(np.array(a), np.array(b))


if __name__ == "__main__":
    unittest.main()
