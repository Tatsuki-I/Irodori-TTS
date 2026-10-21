# Pronunciation dictionary implementation review

Reviewed against upstream `main` at
`89f9d8fbd4d51ea019867ee1197725ede1df13c5`.

## Design and compatibility review

- The feature is opt-in and request-scoped. Existing positional fields retain
  their order; the two new dataclass fields have defaults and are appended.
- Text preparation lives in a standalone module. The runtime changes only its
  text preparation call and attaches the preparation result; sampling, reference,
  caption, codec and seed code are unchanged.
- `text_normalization.py` and `prepare_manifest.py` are byte-identical to the
  upstream baseline. No-dictionary output is compared with the legacy normaliser
  over fixed cases plus 1,000 deterministic generated inputs.
- Dictionary matching is literal and leftmost-longest, with one input scan and
  no rescanning of output. Permutation tests and a separate simple matching
  reference check determinism. NFKC collisions and duplicate JSON keys fail
  explicitly rather than silently choosing an entry.
- JSON version/type/field validation is strict. Unknown fields fail, so typos do
  not silently disable an intended setting. Future changes can add a new version.
- No dependencies, checkpoint formats, licence/copyright notices, UI controls or
  training defaults change. Implementation is original; no Windows ROCm fork
  code or external dictionary code was copied. Example vocabulary is generic;
  unconventional readings from published fiction occur only as test inputs.
- The trie is a small standard-library implementation. Its worst-case lookup
  bound and the need to match legacy-normalised surfaces are documented.

## Review improvements made

The initial tiny-model integration fixture retained the model's zero-initialised
output projection. That made equal output comparisons too weak. The fixture now
initialises that projection with deterministic non-zero random weights, and a
separate test proves that text, caption and seed changes affect its output.
Dictionary input still produces bit-identical output to the equivalent direct
reading across both samplers, duration modes, decode modes and reference input.

Additional checks cover waveform/reference non-interference, runtime reuse without
retaining the preceding dictionary, invalid CLI files before model loading, and
Windows-style UTF-8 BOM dictionary files with Japanese paths.

## Validation results

Environment: Linux x86-64, PyTorch/Torchaudio `2.10.0+cpu`.

| Check | Result |
| --- | --- |
| Python 3.12.14, `python -m unittest discover -s tests -v` | 22 tests pass |
| Python 3.10.22, same suite in a separate environment | 22 tests pass |
| Dictionary module statement coverage | 81/81, 100% |
| `python -m ruff check .` | Pass across the repository |
| Ruff formatting of new Python files and CLI | Pass |
| `python -m compileall -q irodori_tts infer.py prepare_manifest.py tests` | Pass |
| Python 3.10 syntax parsing of changed/new Python files | Pass |
| `python infer.py --help` | Both new options present |
| `git diff --check` | Pass |
| Licence, dependency/lock files and shared normalisation vs baseline | Unchanged |

The baseline repository has no existing automated tests or CI workflows. The
new suite is offline and does not add or start GitHub Actions workflows.
Integration coverage includes 8 combinations of RF/MeanFlow, automatic/manual
duration and sequential/batch decoding, each with two candidates. This exercises
the real model, sampler, tokenizer wrapper and runtime, using a test tokenizer
backend and test codec. Waveform loading is mocked in the reference-audio test;
actual audio codec/checkpoint loading and audio quality are not established.

## Remaining acceptance checks before an upstream PR

1. Audition a short sample using a released checkpoint on the intended CPU/GPU.
   Check the selected reading, retained voice/style and tokenizer length limits.
2. Run the suite and a CLI example on Windows; BOM and portable path handling
   are tested on Linux, but a native Windows runtime has not been exercised.
3. Agree that v1 should match normalised surfaces and reject all duplicate
   surfaces, including identical readings and NFKC aliases.
4. Review the additive Python field names and CLI flags against maintainer
   preferences. Issue #21 currently puts pronunciation control outside the
   roadmap; this implementation does not claim English model support.

No blocking code defects remain from this self-review. The released-model and
native Windows checks above remain unverified and are not represented as passed.
No upstream PR (including draft), issue, comment or maintainer contact is part of
this work.
