# Provisional alignment and blind review follow-up

This continues the saved 40-clip study without changing synthesis, the production
dictionary, or the original audio/F0 evidence. **Accent control is still not
validated.** Human listening and phonetic boundary confirmation remain pending.

## What was done

A local multilingual CTC acoustic model was used to estimate Latin character
anchors for each saved WAV. Explicit expected mora/Latin units for these three
readings and two contexts are in `alignment-transcripts.json`; this is not a
general Japanese romaniser. The acoustic model is
`MahmoudAshraf/mms-300m-1130-forced-aligner`, pinned at
`49402e9577b1158620820667c218cd494cc44486`.

An original NumPy CTC Viterbi implementation handles blanks and repeated labels.
Its optimum is tested against exhaustive enumeration of all short paths. It does
not copy the external aligner package or use the removed TorchAudio forced-align
API. Wav2Vec2 convolution stride/receptive field are read from model configuration
for timestamp offsets. Neighbouring mora boundaries are estimated at midpoints
between character anchors. **These are not verified phoneme/mora boundaries.**

The script records hashes of local model assets and its generation code,
package versions, greedy decoding, a character-error diagnostic, character
anchors, estimated mora intervals, local voiced F0 and sensitivity to shifting
all estimated intervals by +/-10 ms. An observed repeated execution produced
identical per-clip result rows. Cross-version/platform repeatability is not claimed.

## Results and interpretation

Full records: `results/automatic-alignment.json`.
Counted observations: `results/alignment-summary.json`.

| Screen result | Count |
| --- | ---: |
| WAVs processed | 40 |
| Clips with an intended nucleus | 30 |
| No nucleus requested (original/reading controls) | 10 |
| Provisional F0-shape match | 2 |
| Provisional F0-shape non-match | 12 |
| No shape assessment after gating/insufficient evidence | 16 |
| Human-confirmed lexical accent successes | **Not measured** |

The +/-10 ms test shifts complete intervals together; it is not a bound on
phonetic-boundary error, which can be much larger.

This is **not an accuracy table**. Expected transcripts are forced onto audio;
missing/wrong sounds can still get anchors. Greedy CTC output is an imperfect
recogniser, not human transcript verification. Thresholds (CER <=0.25, mean target
mora posterior >=0.2) and F0 rules are exploratory and uncalibrated, not probabilities
of correct Japanese accent. A shape match can pass while the transcript is wrong.

The two provisional candidates show that problem directly:

| Candidate | Expected Latin | Greedy Latin | Target-to-particle adjacent F0 drops |
| --- | --- | --- | --- |
| `hashi-c0-s1-downstep-n2-r0` | `hashigaaru` | `hashiagaaru` | -4.74, +4.06 semitones |
| `ame-c0-s0-arrows-n0-r0` | `amegaaru` | `anegaaru` | -6.20, +2.18 semitones |

Positive values mean falling pitch. The flat target permits up to 3 semitones of
same-level drift, which is another heuristic assumption requiring review. Neither
candidate has an exact greedy transcript match; this may reflect real reading
errors **or recogniser errors**. Do not label either as a demonstrated accent
success. A stronger exact-greedy-transcript filter would leave neither candidate,
but that post-hoc observation is not a calibrated success/failure metric either.

All natural-speech rows retain `transcript_checked=false`,
`alignment_checked=false`, `accent_verified=false`. Automatic estimates do not
get written into the manual annotation channel.

## Reproduce alignment separately from inference

Use the existing Irodori environment, with optional analysis dependencies
`numpy`, `scipy`, `soundfile`, and `transformers==5.12.1`. No production dependencies
or lock files were changed. Download only the pinned alignment model assets:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    "MahmoudAshraf/mms-300m-1130-forced-aligner",
    revision="49402e9577b1158620820667c218cd494cc44486",
    allow_patterns=["*.json", "model.safetensors", "README.md"],
    local_dir="outputs/prosody-aligner",
)
```

```sh
python -m experiments.dictionary_prosody.align --model outputs/prosody-aligner --output outputs/provisional-alignment.json
```

The script refuses an existing output filename. It uses the committed WAV/F0 data
and does not load or modify a TTS checkpoint. Check the saved asset hashes and
code hashes before comparing reruns. For a repeatability check, run twice to new
files and compare the `rows` arrays, not just high-level screen counts.

### Model licence and scope

The alignment model card declares **CC-BY-NC-4.0**. Its weights were used only in
this optional research environment and are not redistributed, bundled into Irodori
or made a runtime dependency. The repository's MIT code/licence is unchanged.
A future commercial alignment product would need a suitably licensed acoustic
model or an explicit separate licence review; do not assume these weights are an
MIT production dependency. No external aligner implementation was copied.

## Blind listening kit

The generator makes a local kit with coded WAV names, F0 data, a reading-only
comparison from the same run/word/context/seed/repeat, and estimated mora boundaries.
The normal screen does not show method, intended nucleus, original IDs or paths.
The separate `review-key.json` maps codes back to conditions; this is practical
blinding for a reviewer, **not cryptographic protection** against inspecting files.

```sh
python -m experiments.dictionary_prosody.review --output outputs/prosody-review
python -m http.server 8765 --bind 127.0.0.1 --directory outputs/prosody-review
```

Open `http://127.0.0.1:8765/` on that computer. Listen before displaying CTC
estimates; then correct boundaries if needed. Record perceived nucleus,
uncertainty, naturalness and delivery differences against the matched baseline.
Drafts are saved locally when browser storage is available; if storage is blocked,
the UI keeps an in-memory draft and tells the reviewer to export it. No messages
or audio are sent to a third-party evaluation service.

Export a draft review, then import it with the matching key:

```sh
python -m experiments.dictionary_prosody.review --import-review /path/to/prosody-review.json --key outputs/prosody-review/review-key.json --output outputs/imported-review.json --annotations-output outputs/checked-annotations.json
```

An importer requires a reviewer identifier and matching kit fingerprint. It
rejects duplicate/unknown codes, non-boolean acknowledgements, invalid nucleus
ranges, changed unit labels, missing/overlapping/out-of-range boundaries and
invalid ratings. A draft with estimated boundaries is not manual evidence.
Only records with **both explicit transcript and alignment confirmation** enter
`checked-annotations.json`. Even those records do not automatically become
verified accent-control results. The imported annotations carry the original
WAV hash; `analyse.py` rejects a mismatched annotation hash. Legacy manually
written annotations without a hash retain their earlier interface.

The original analyser can then use the reviewed boundaries:

```sh
python -m experiments.dictionary_prosody.analyse /path/to/run --annotations outputs/checked-annotations.json
```

Use a copy of a run for human-reviewed analysis if preserving the original
unreviewed analysis alongside it. A second competent Japanese listener and
held-out word/context/seed tests are still needed before claiming a backend.

## Validation and limitations

- 45 Python tests pass, including all prior 33 tests.
- The new tests include exhaustive CTC-path optimality, repeated-label blanks,
  invalid inputs, explicit transcript scope, heuristic abstention, blind-kit
  asset checks, import provenance and refusal to promote unchecked estimates.
- Repository-wide Ruff, Python compileall and Python 3.10 syntax parsing pass.
  The new code was executed on Python 3.12/Linux/CPU; no new Windows/GPU test.
- `tests/prosody_review_dom.cjs` passes with JSDOM: masked conditions, baseline
  assets, selections, estimated-boundary visibility, navigation, local draft
  persistence, blocked-storage fallback and JSON export.
- This is **DOM logic testing**, not actual browser playback or auditory review.
  Native headless-browser downloads were unavailable in this environment, so
  rendering/audio-decoder behaviour is not marked as tested. No listening scores
  or completed human-review records were fabricated.

Optional DOM test, without adding a repository Node dependency:

```sh
npm install --prefix /tmp/prosody-dom-test jsdom
NODE_PATH=/tmp/prosody-dom-test/node_modules node tests/prosody_review_dom.cjs outputs/prosody-review
```

Self-review tightened two important boundaries: automatic estimates cannot be
promoted by opening the screen, and checked annotations cannot be silently
applied to a different WAV. The baseline matcher is scoped by run and repeat so
future experiments cannot accidentally compare against a different run's settings.

The production decision remains unchanged: keep accent support experimental,
retain version 1 reading dictionaries, and do not add ignored accent fields to
production APIs. No upstream PR, issue, comment or maintainer message was sent.

## Primary references

- [Pinned CTC model card and declared licence](https://huggingface.co/MahmoudAshraf/mms-300m-1130-forced-aligner/blob/49402e9577b1158620820667c218cd494cc44486/README.md)
- [TorchAudio multilingual alignment tutorial: normalisation and segmentation](https://docs.pytorch.org/audio/2.8/tutorials/forced_alignment_for_multilingual_data_tutorial.html)

Further machine-only checks, matched negative controls and extractor comparisons: [ROBUSTNESS.md](ROBUSTNESS.md).
