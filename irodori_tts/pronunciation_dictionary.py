"""Versioned, literal pronunciation substitutions without model dependencies."""

from __future__ import annotations

import json
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .text_normalization import normalize_text


def _validated_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-blank string.")
    # JSON permits lone surrogates; tokenizers/UTF-8 output do not.
    if any(unicodedata.category(char) == "Cs" or char == "\x00" for char in value):
        raise ValueError(f"{field} must not contain NUL or surrogate characters.")
    return unicodedata.normalize("NFKC", value)


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


class PronunciationDictionary:
    """An NFKC-normalised literal dictionary with leftmost-longest matching.

    Entries are copied and compiled once. Replacements are never scanned again.
    Duplicate surfaces (including NFKC aliases) are rejected, even when their
    readings agree. No regular expressions or word-boundary heuristics are used.
    """

    def __init__(self, entries: Iterable[tuple[str, str]]) -> None:
        self._trie: dict = {}
        seen: set[str] = set()
        for index, (surface, reading) in enumerate(entries):
            surface = _validated_text(surface, f"entries[{index}].surface")
            reading = _validated_text(reading, f"entries[{index}].reading")
            if surface in seen:
                raise ValueError(f"Duplicate surface after NFKC normalisation: {surface!r}")
            seen.add(surface)
            node = self._trie
            for char in surface:
                node = node.setdefault(char, {})
            # None cannot collide with any string character.
            node[None] = reading

    @classmethod
    def from_dict(cls, data: object) -> PronunciationDictionary:
        """Validate a version 1 JSON-compatible document (unknown fields fail)."""
        if not isinstance(data, dict) or set(data) != {"version", "entries"}:
            raise ValueError("Dictionary must contain exactly 'version' and 'entries'.")
        if type(data["version"]) is not int or data["version"] != 1:
            raise ValueError("Unsupported dictionary version; expected integer 1.")
        if not isinstance(data["entries"], list):
            raise ValueError("Dictionary entries must be a list.")
        entries = []
        for index, entry in enumerate(data["entries"]):
            if not isinstance(entry, dict) or set(entry) != {"surface", "reading"}:
                raise ValueError(f"entries[{index}] must contain exactly 'surface' and 'reading'.")
            entries.append((entry["surface"], entry["reading"]))
        return cls(entries)

    @classmethod
    def from_file(cls, path: str | Path) -> PronunciationDictionary:
        """Load UTF-8 JSON, accepting a BOM for Windows editors.

        I/O errors propagate as OSError; invalid JSON/schema raises ValueError.
        No partial dictionary is returned on failure.
        """
        with Path(path).expanduser().open(encoding="utf-8-sig") as handle:
            return cls.from_dict(json.load(handle, object_pairs_hook=_unique_object))

    def apply(self, text: str) -> str:
        """NFKC-normalise input and substitute once, scanning left to right.

        At each input position the longest matching surface wins. Unmatched
        characters, punctuation and line breaks are copied verbatim after NFKC.
        Runtime is O(len(text) * longest_surface_length) in the worst case.
        """
        text = unicodedata.normalize("NFKC", text)
        parts: list[str] = []
        start = 0
        while start < len(text):
            node = self._trie
            end = start
            match_end = start
            reading = None
            while end < len(text) and text[end] in node:
                node = node[text[end]]
                end += 1
                if None in node:
                    match_end, reading = end, node[None]
            if reading is None:
                parts.append(text[start])
                start += 1
            else:
                parts.append(reading)
                start = match_end
        return "".join(parts)


@dataclass(frozen=True)
class PreparedText:
    original_text: str
    normalized_text: str
    spoken_text: str


def prepare_text(
    text: str, pronunciation_dictionary: PronunciationDictionary | None = None
) -> PreparedText:
    """Prepare inference text without changing training-time normalisation.

    Normalise first, substitute second, then trim exactly as inference did before.
    Readings are not run through the legacy normaliser or dictionary again.
    """
    normalized = normalize_text(text).strip()
    spoken = (
        normalized
        if pronunciation_dictionary is None
        else pronunciation_dictionary.apply(normalized).strip()
    )
    return PreparedText(text, normalized, spoken)
