"""Strict research-only accent schema and explicitly unvalidated input probes."""

from __future__ import annotations

import unicodedata

from irodori_tts.pronunciation_dictionary import PronunciationDictionary, prepare_text

_SMALL = set("ぁぃぅぇぉゃゅょゎ")
_JOINS = {
    **{base: set("ゃゅょ") for base in "きぎしじちぢにひびぴみり"},
    "い": set("ぇ"),
    "う": set("ぃぇぉ"),
    "く": set("ぁぃぇぉゎ"),
    "ぐ": set("ぁぃぇぉゎ"),
    "し": set("ゃゅょぇ"),
    "じ": set("ゃゅょぇ"),
    "ち": set("ゃゅょぇ"),
    "つ": set("ぁぃぇぉ"),
    "て": set("ぃゅ"),
    "で": set("ぃゅ"),
    "と": set("ぅ"),
    "ど": set("ぅ"),
    "ふ": set("ぁぃぇぉゅ"),
    "ゔ": set("ぁぃぇぉゅ"),
}
METHODS = ("katakana", "downstep", "arrows", "space", "comma", "caption")


def split_mora(reading: str) -> tuple[str, ...]:
    """Conservative kana spelling segmentation, not a morphological analyser.

    NFKC and katakana→hiragana are applied. Small kana join a supported base;
    sokuon, moraic nasal and prolonged-sound mark each occupy a separate mora.
    Unrecognised spellings are rejected instead of guessing a count.
    """
    if not isinstance(reading, str) or not reading:
        raise ValueError("reading must be non-empty kana")
    text = unicodedata.normalize("NFKC", reading)
    text = "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in text)
    moras: list[str] = []
    for char in text:
        if char in _SMALL:
            if not moras or len(moras[-1]) != 1 or char not in _JOINS.get(moras[-1], set()):
                raise ValueError(f"Unsupported small-kana sequence: {reading!r}")
            moras[-1] += char
        elif char == "ー":
            if not moras:
                raise ValueError("A prolonged-sound mark needs a preceding mora")
            moras.append(char)
        elif "ぁ" <= char <= "ゔ" and char not in "ゎゕゖ":
            moras.append(char)
        else:
            raise ValueError(f"Unsupported kana spelling: {reading!r}")
    return tuple(moras)


def validate_entry(entry: object) -> tuple[str, str, int, tuple[str, ...]]:
    if not isinstance(entry, dict) or set(entry) != {"surface", "reading", "accent"}:
        raise ValueError("Research entry requires surface, reading and accent")
    # Reuse v1 validation, including blank/surrogate checks.
    PronunciationDictionary.from_dict(
        {"version": 1, "entries": [{"surface": entry["surface"], "reading": entry["reading"]}]}
    )
    accent = entry["accent"]
    if not isinstance(accent, dict) or set(accent) != {"dialect", "nucleus"}:
        raise ValueError("accent requires exactly dialect and nucleus")
    if accent["dialect"] != "tokyo":
        raise ValueError("Only the experimental tokyo convention is supported")
    moras = split_mora(entry["reading"])
    nucleus = accent["nucleus"]
    if type(nucleus) is not int or not 0 <= nucleus <= len(moras):
        raise ValueError(f"nucleus must be an integer in 0..{len(moras)}")
    return entry["surface"], "".join(moras), nucleus, moras


def pitch_labels(mora_count: int, nucleus: int, *, particle: bool = False) -> str:
    """Idealised H/L labels, not an acoustic target or a surface-accent predictor."""
    if (
        type(mora_count) is not int
        or mora_count < 1
        or type(nucleus) is not int
        or not 0 <= nucleus <= mora_count
    ):
        raise ValueError("Invalid mora count or nucleus")
    count = mora_count + int(particle)
    return "".join(
        "H"
        if (nucleus == 1 and i == 1) or (nucleus != 1 and i > 1 and (nucleus == 0 or i <= nucleus))
        else "L"
        for i in range(1, count + 1)
    )


def render_probe(entry: object, method: str) -> tuple[str, str | None]:
    """Return experimental replacement and optional caption suffix.

    There is no evidence contract for these notations. Callers must keep the
    method/target in the manifest and must not expose this as accent support.
    """
    _, reading, nucleus, moras = validate_entry(entry)
    if method not in METHODS:
        raise ValueError("An explicit experimental method is required")
    if method == "katakana":
        return "".join(chr(ord(c) + 0x60) if "ぁ" <= c <= "ゔ" else c for c in reading), None
    if method == "caption":
        return reading, (
            f"「{reading}」の東京式アクセント核は{nucleus}。"
            + (
                "平板型で、助詞まで高く続ける。"
                if nucleus == 0
                else f"第{nucleus}モーラの直後で音高を下げる。"
            )
        )
    if method == "arrows":
        if nucleus == 0:
            return moras[0] + "↑" + "".join(moras[1:]), None
        return "".join(moras[:nucleus]) + "↓" + "".join(moras[nucleus:]), None
    marker = {"downstep": "ꜜ", "space": " ", "comma": "、"}[method]
    if nucleus == 0:
        return reading, None
    return "".join(moras[:nucleus]) + marker + "".join(moras[nucleus:]), None


def prepare_probe(text: str, entry: object, method: str, caption: str) -> dict:
    surface, reading, nucleus, moras = validate_entry(entry)
    replacement, suffix = render_probe(entry, method)
    dictionary = PronunciationDictionary([(surface, replacement)])
    prepared = prepare_text(text, dictionary)
    return {
        "original_text": text,
        "normalized_text": prepared.normalized_text,
        "spoken_text": prepared.spoken_text,
        "caption": caption if suffix is None else caption + " " + suffix,
        "reading": reading,
        "moras": list(moras),
        "nucleus": nucleus,
        "method": method,
        "experimental_only": True,
        "validated_accent_control": False,
    }
