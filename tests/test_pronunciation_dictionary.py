"""Offline dictionary tests; run with python -m unittest discover -s tests -v."""

from __future__ import annotations

import itertools
import json
import random
import tempfile
import unittest
from pathlib import Path

from irodori_tts.pronunciation_dictionary import PronunciationDictionary, prepare_text
from irodori_tts.text_normalization import normalize_text


class DictionaryTests(unittest.TestCase):
    def test_unconventional_readings(self):
        dictionary = PronunciationDictionary(
            [("禁書目録", "インデックス"), ("超電磁砲", "レールガン"), ("一方通行", "アクセラレータ")]
        )
        text = "禁書目録と超電磁砲、一方通行。"
        self.assertEqual(dictionary.apply(text), "インデックスとレールガン、アクセラレータ。")
        self.assertEqual(prepare_text(text).spoken_text, text)

    def test_leftmost_longest_and_order_independence(self):
        entries = [("東京", "とうきょう"), ("東京都", "とうきょうと"), ("京都", "きょうと")]
        for permutation in itertools.permutations(entries):
            self.assertEqual(
                PronunciationDictionary(permutation).apply("東京都と京都"), "とうきょうとときょうと"
            )

    def test_no_recursive_or_cross_boundary_replacement(self):
        dictionary = PronunciationDictionary([("A", "B"), ("B", "C"), ("BC", "D")])
        self.assertEqual(dictionary.apply("ABAC"), "BCBC")

    def test_literal_matching(self):
        dictionary = PronunciationDictionary([("a.*[x]", "literal"), ("猫", "ねこ")])
        self.assertEqual(dictionary.apply("a.*[x] 山猫"), "literal 山ねこ")

    def test_unicode(self):
        dictionary = PronunciationDictionary([("ＡＩ", "ｴｰｱｲ"), ("が", "ガ")])
        self.assertEqual(dictionary.apply("AI ＡＩ か\u3099"), "エーアイ エーアイ ガ")
        self.assertEqual(prepare_text("ＡＩ", dictionary).spoken_text, "エーアイ")

    def test_punctuation_newlines_and_reading_not_renormalized(self):
        dictionary = PronunciationDictionary([("猫", "「ねこ」"), ("犬\n猫", "いぬ\nねこ")])
        self.assertEqual(dictionary.apply("猫、犬\n猫!\n猫"), "「ねこ」、いぬ\nねこ!\n「ねこ」")
        self.assertEqual(prepare_text("猫", dictionary).spoken_text, "「ねこ」")

    def test_empty_and_long_text(self):
        dictionary = PronunciationDictionary([("猫", "ねこ"), ("猫猫", "にゃん")])
        self.assertEqual(dictionary.apply(""), "")
        self.assertEqual(dictionary.apply("猫猫。" * 40000), "にゃん。" * 40000)
        self.assertEqual(prepare_text("", dictionary).spoken_text, "")

    def test_empty_dictionary(self):
        self.assertEqual(
            prepare_text("「猫！」", PronunciationDictionary([])).spoken_text,
            normalize_text("「猫！」").strip(),
        )

    def test_duplicate_and_nfkc_collision(self):
        for entries in [
            [("猫", "ねこ"), ("猫", "ねこ")],
            [("猫", "ねこ"), ("猫", "にゃん")],
            [("Ａ", "えー"), ("A", "あ")],
            [("が", "a"), ("か\u3099", "b")],
        ]:
            with (
                self.subTest(entries=entries),
                self.assertRaisesRegex(ValueError, "Duplicate surface"),
            ):
                PronunciationDictionary(entries)

    def test_invalid_schema(self):
        invalid = [
            None,
            [],
            {},
            {"version": 1},
            {"version": 2, "entries": []},
            {"version": True, "entries": []},
            {"version": 1.0, "entries": []},
            {"version": 1, "entries": {}},
            {"version": 1, "entries": [None]},
            {"version": 1, "entries": [{"surface": "a"}]},
            {"version": 1, "entries": [], "typo": 1},
        ]
        for document in invalid:
            with self.subTest(document=document), self.assertRaises(ValueError):
                PronunciationDictionary.from_dict(document)
        for surface, reading in [
            ("", "x"),
            ("x", ""),
            (" ", "x"),
            ("x", "\n"),
            (None, "x"),
            ("x", 1),
            ("\x00", "x"),
            ("x", "\ud800"),
        ]:
            with self.subTest(surface=surface, reading=reading), self.assertRaises(ValueError):
                PronunciationDictionary.from_dict(
                    {"version": 1, "entries": [{"surface": surface, "reading": reading}]}
                )

    def test_file_loading_and_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "辞書.json"
            with self.assertRaises(FileNotFoundError):
                PronunciationDictionary.from_file(path)
            path.write_text(
                json.dumps(
                    {"version": 1, "entries": [{"surface": "猫", "reading": "ねこ"}]},
                    ensure_ascii=False,
                ),
                encoding="utf-8-sig",
            )
            self.assertEqual(PronunciationDictionary.from_file(str(path)).apply("猫"), "ねこ")
            for payload in [
                "{",
                '{"version":1,"version":1,"entries":[]}',
                '{"version":1,"entries":[{"surface":"猫","surface":"犬","reading":"ねこ"}]}',
                "[]",
            ]:
                path.write_text(payload, encoding="utf-8")
                with self.subTest(payload=payload), self.assertRaises(ValueError):
                    PronunciationDictionary.from_file(path)
            path.write_bytes(b"\xff")
            with self.assertRaises(UnicodeError):
                PronunciationDictionary.from_file(path)
            with self.assertRaises(OSError):
                PronunciationDictionary.from_file(directory)

    def test_loaded_entries_are_copied_and_empty_dictionary_is_valid(self):
        document = {"version": 1, "entries": [{"surface": "猫", "reading": "ねこ"}]}
        dictionary = PronunciationDictionary.from_dict(document)
        document["entries"][0]["reading"] = "changed"
        self.assertEqual(dictionary.apply("猫"), "ねこ")
        self.assertEqual(
            PronunciationDictionary.from_dict({"version": 1, "entries": []}).apply("猫"), "猫"
        )

    def test_surfaces_match_normalized_input(self):
        # The legacy normaliser removes these symbols before substitution.
        dictionary = PronunciationDictionary([("○", "まる"), ("ー", "ながい"), ("!", "びっくり")])
        self.assertEqual(prepare_text("〇～！", dictionary).spoken_text, "まるながいびっくり")

    def test_no_dictionary_legacy_parity(self):
        samples = [
            "",
            "「『猫？！』」",
            "\t[n]猫\\[n\\]　…...",
            "ＡＢＣ ① ～ 〇♥",
            "\n 猫\n犬 \n",
            "（猫）と（犬）",
            "か\u3099",
        ]
        rng = random.Random(42)
        alphabet = "猫犬\n\t []n\\「」（）Ａa？！♥〇～…-か\u3099"
        samples.extend("".join(rng.choices(alphabet, k=rng.randrange(100))) for _ in range(1000))
        for raw in samples:
            with self.subTest(raw=raw):
                prepared = prepare_text(raw)
                self.assertEqual(prepared.original_text, raw)
                self.assertEqual(prepared.normalized_text, normalize_text(raw).strip())
                self.assertEqual(prepared.spoken_text, normalize_text(raw).strip())

    def test_randomised_against_simple_reference(self):
        rng = random.Random(19)
        for _ in range(100):
            keys = {"".join(rng.choices("abc", k=rng.randrange(1, 6))) for _ in range(12)}
            entries = [(key, f"<{key}>") for key in keys]
            text = "".join(rng.choices("abc!", k=100))
            expected = []
            position = 0
            ordered = sorted(entries, key=lambda entry: (-len(entry[0]), entry[0]))
            while position < len(text):
                for surface, reading in ordered:
                    if text.startswith(surface, position):
                        expected.append(reading)
                        position += len(surface)
                        break
                else:
                    expected.append(text[position])
                    position += 1
            self.assertEqual(
                PronunciationDictionary(reversed(entries)).apply(text), "".join(expected)
            )


if __name__ == "__main__":
    unittest.main()
