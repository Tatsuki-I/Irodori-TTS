# Custom pronunciation dictionaries

A user dictionary replaces selected spellings with readings before tokenisation.
It needs no new packages, model changes, checkpoint conversion or training.
This is text substitution, not a phoneme or pitch-accent interface: pronunciation
quality still depends on the checkpoint. It does not add English-language support.

## JSON format

Save a UTF-8 JSON file (a UTF-8 BOM is also accepted):

```json
{
  "version": 1,
  "entries": [
    {"surface": "日本橋", "reading": "にほんばし"},
    {"surface": "AI", "reading": "エーアイ"}
  ]
}
```

See [the example dictionary](../examples/pronunciation_dictionary.json).
Choose the desired regional/name reading explicitly; no built-in vocabulary is
added. Hiragana and katakana readings are recommended, but any non-blank UTF-8
string is accepted. Punctuation and internal line breaks are allowed.

Version 1 requires exactly `version` and `entries`, with exactly `surface` and
`reading` per entry. Unknown fields, unsupported versions, wrong types, blank
strings, NUL characters and lone Unicode surrogates raise `ValueError`. Empty
entry lists are valid. Duplicate JSON object keys and duplicate surfaces are
rejected, even if their readings agree. This also rejects Unicode aliases such
as `ＡＩ` and `AI` (NFKC), rather than choosing a reading by entry order.
Future format changes must use an explicitly supported version.

Missing/unreadable files raise `OSError` (including `FileNotFoundError`); malformed
JSON, encoding and schema errors raise `ValueError`. Loading is all-or-nothing.
CLI failures are reported by argparse before any checkpoint download or load.

## CLI

From the repository root, using an existing inference environment:

```sh
uv run infer.py --hf-checkpoint Aratako/Irodori-TTS-v4.1-Small --text "日本橋でAIの話をする。" --no-ref --pronunciation-dictionary examples/pronunciation_dictionary.json --show-text
```

The same single-line command works in PowerShell. Quote file paths containing
spaces. All existing reference audio, caption, seed and other sampling options
remain available. `--show-text` prints `repr`-escaped original, normalised and
spoken text after synthesis, so line breaks are visible. Omit this flag to avoid
printing input text. Omit `--pronunciation-dictionary` to retain existing behaviour.

## Python API

Load once and reuse the compiled dictionary across requests:

```python
from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey, SamplingRequest
from irodori_tts.pronunciation_dictionary import PronunciationDictionary, prepare_text

dictionary = PronunciationDictionary.from_file("examples/pronunciation_dictionary.json")
preview = prepare_text("日本橋でAIの話をする。", dictionary)
print(preview.original_text)
print(preview.normalized_text)
print(preview.spoken_text)  # にほんばしでエーアイの話をする。

runtime = InferenceRuntime.from_key(
    RuntimeKey(checkpoint="model.safetensors", model_device="cpu", codec_device="cpu")
)
result = runtime.synthesize(
    SamplingRequest(
        text="日本橋でAIの話をする。",
        pronunciation_dictionary=dictionary,
        no_ref=True,
        seed=42,
    )
)
print(result.text_processing)
```

`prepare_text` previews the exact pre-tokenisation text without loading a model.
The result is an immutable `PreparedText` with `original_text`, `normalized_text`
and `spoken_text` fields. Successful synthesis exposes the same object in
`SamplingResult.text_processing`. The new result field defaults to `None` for
compatibility with callers constructing their own results. New request/result
fields are appended, preserving existing positional argument order.

For in-memory definitions, use `PronunciationDictionary.from_dict(document)` or
`PronunciationDictionary([("日本橋", "にほんばし")])`. `dictionary.apply(text)`
performs NFKC normalisation and substitution only; use `prepare_text` to include
the existing inference normalisation and whitespace trimming.

## Matching and pipeline

1. Run the existing `normalize_text` and inference whitespace trimming unchanged.
2. If a dictionary was supplied, substitute readings in that normalised text.
3. Trim the result, reject empty inference text as before, then tokenise it.
   Automatic duration prediction also uses this final spoken text.

Surfaces and readings are NFKC-normalised when compiled. Matching is literal,
case-sensitive and leftmost-longest: at each position, consume the longest
matching surface, or copy the next character if none matches. A surface may
match inside another word; there is no morphological or word-boundary analysis.
Overlapping matches at later positions do not override an earlier match.
Dictionary order cannot change the result.

Replacement output is never scanned again. For `A → B` and `B → C`, input `AB`
becomes `BC`, not `CC`. Readings are not run through `normalize_text` again;
this avoids accidentally stripping their punctuation or changing their meaning.
Unmatched punctuation/newlines survive subject to the existing normaliser.

**Surfaces must match the text after the existing normaliser.** For example,
`〇` becomes `○`, a wave dash becomes `ー`, and outer enclosing brackets may be
removed. Use `○` or `ー` as surfaces for these inputs. Surface compilation uses
NFKC only; it does not strip brackets or apply those legacy symbol substitutions.
`prepare_text(text).normalized_text` shows the string to match.

The trie is compiled once, uses no regular expressions and has worst-case lookup
cost `O(text length × longest surface length)`. Use dictionaries appropriate to
available memory; this is intended for user vocabularies, not huge lexicons.

## Compatibility and scope

- No dictionary means exactly the previous normalisation/trim path. The shared
  `text_normalization.py` and `prepare_manifest.py` are unchanged, so training
  data preparation is unaffected.
- Only request `text` is processed. Caption, reference paths/audio/embeddings,
  seeds, LoRA settings and sampling options are not rewritten. Dictionaries are
  request-scoped, so a cached runtime does not retain one for the next request.
- Readings can change token count and predicted duration. Existing tokenizer
  truncation and `max_text_len` rules still apply; preview text is the complete
  pre-tokenisation string, not a claim that every character fits the token limit.
- JSON/path handling uses the standard library and supports Python 3.10+ on
  Windows/Linux. No Windows ROCm fork changes are included.
- Gradio UI controls, training dictionaries, SSML, morphology, accent control and
  fine-tuning are outside this change.

## Offline tests

With the project's dependencies installed:

```sh
python -m unittest discover -s tests -v
python -m ruff check infer.py irodori_tts/inference_runtime.py irodori_tts/pronunciation_dictionary.py tests
```

Tests require no network, GPU or released model downloads. Unit tests include
Unicode collisions, invalid files/schema, longest/overlapping matches,
non-recursion, long text and legacy normalisation parity. CPU integration tests
exercise real tiny random-weight models, both RF and MeanFlow samplers,
automatic/manual duration, multiple candidates and both decode modes. Only the
external tokenizer backend, audio codec and optional watermark backend are test
doubles; checkpoint loading is bypassed in the CLI test. Tests compare dictionary
input to the equivalent explicit reading and check caption/reference/seed
non-interference. They do **not** establish released-model audio quality.
Before submitting upstream, audition a short sample with a released checkpoint
on the intended CPU/GPU and confirm both its reading and voice quality.
