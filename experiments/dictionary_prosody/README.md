# Dictionary prosody experiment

**Research only. No production pitch-accent support is enabled.** The version 1
dictionary and inference runtime remain unchanged from `pre-publication research baseline`.

This experiment asks whether a released Irodori-TTS-v4.1-Small checkpoint can be
controlled through existing text/caption inputs. A target nucleus in a manifest
is an experimental intention, never evidence that the waveform has that accent.
See [REPORT.md](REPORT.md) for observations and limitations.

## Environment

Use an existing CPU Irodori environment installed according to the main README.
Install optional analysis tools separately:

```sh
uv pip install -r experiments/dictionary_prosody/requirements-analysis.txt
```

PyWORLD builds may need a C++ compiler (`CC=gcc CXX=g++` on this Linux host).
No root dependency/lock file is modified. Run commands from the repository root.
The experiment runner deliberately uses CPU FP32 for reproducibility. The
production runtime is unmodified and still supports its existing devices.

Download `model.safetensors` and the complete `tokenizer/` directory from:

- Model: `Aratako/Irodori-TTS-v4.1-Small`
- Revision: `2b28324dc263ed5e6638b3cf3dd94c82ead07b4b`
- Codec: `Aratako/Semantic-DACVAE-Japanese-32dim`, file `weights.pth`
- Codec revision: `47376ee24834d7a05a48ebabfe3cde29b3c5e214`
- DACVAE source used: `facebookresearch/dacvae` commit
  `414c20785fc3a28373073ea8ef7a1316eeeaca6e`

Record hashes (the runner does so automatically); do not use an unpinned download
and call it the same experiment. Model assets are not committed to this repo.

## Comparison protocol

`cases.json` supplies three kana readings, alternative nuclei, two contexts and
three seeds. These are **target hypotheses**; alternative nuclei are deliberate
counterfactuals, not claims about the correct dictionary accent of every kanji.
The following particle `が` helps distinguish unaccented and final-accent targets.

Generate a plan without loading model weights:

```sh
python -m experiments.dictionary_prosody.run --output /tmp/prosody-plan --plan-only
```

Generate the pilot (use a new output directory for every run):

```sh
python -m experiments.dictionary_prosody.run --output /tmp/prosody-pilot --checkpoint /path/to/model.safetensors --codec /path/to/weights.pth --words ame --contexts 0 --seeds 0 --methods downstep arrows caption --repeats 1 --steps 40 --seconds 2 --threads 4
```

Generalisation probe:

```sh
python -m experiments.dictionary_prosody.run --output /tmp/prosody-generalisation --checkpoint /path/to/model.safetensors --codec /path/to/weights.pth --words ame hashi kokoro --contexts 0 --seeds 1 --methods downstep caption --repeats 1 --steps 40 --seconds 2 --threads 4
```

Use `--reference /path/to/reference.wav` for a separate reference-controlled run.
No reference was used in the saved runs. Do not mix these two conditions in one
comparison. `--repeats 2` generates exact-input repeats for determinism checks.
Some methods/nuclei intentionally render identical inputs; these are negative
controls. Do not count different nucleus metadata as different model conditions
when text, caption and sampling parameters are identical.

The runner uses the real `SamplingRequest`/`InferenceRuntime`, not a mock.
Only replacement strings and caption suffixes vary. Manual duration is fixed to
2 seconds to avoid duration-prediction changes confounding the primary probe;
this can itself constrain naturalness, so follow-up acceptance must also test
automatic duration and longer sentences. RF steps=40, text/caption CFG=3,
speaker CFG=5 (inactive without reference), seed, FP32, no tail trim and codec
settings are recorded. Full request settings, token IDs, checkpoint/codec hashes,
packages, generation status and per-clip timing are saved in `manifest.json`.
A failure is recorded as `failed`, not a successful probe. Interrupted rows remain
`planned`; counts must use `status == generated`.

WAV files use float samples. Compare decoded PCM hashes, not just WAV file hashes:
libsndfile can write a timestamped PEAK chunk, so file hashes can differ even when
all audio samples agree exactly. Analysis records a separate `pcm_sha256`.

## Methods

| Probe | Transformation | Main confound |
| --- | --- | --- |
| original | No dictionary | Original kanji retains semantic/lexical cues |
| reading | Version 1 kana substitution | Homophones lose those kanji cues |
| katakana | Change kana script | Does not encode nucleus; negative control |
| downstep | Insert `ꜜ` after the target mora (none for 0) | No documented training semantics |
| arrows | Insert `↓` after the target mora, or `↑` after mora 1 for 0 | Symbol/phrase interpretation is unknown |
| space | Split after the target mora (none for 0) | Token/phrase boundary changes |
| comma | Insert `、` after the target mora (none for 0) | Pause and accent-phrase boundary changes |
| caption | Add an explicit Japanese instruction, retain kana text | Global style/identity/speed may change |

Symbols are inserted by the dictionary **after** the legacy normaliser; they are
not specially interpreted by any added runtime code. Record resulting token IDs;
a symbol being tokenisable is not evidence of a learned prosody protocol.

## F0 analysis and alignment

```sh
python -m experiments.dictionary_prosody.analyse /tmp/prosody-pilot
python -m experiments.dictionary_prosody.plot /tmp/prosody-pilot
```

WORLD Harvest extracts F0 every 5 ms at 60–600 Hz. Inspect octave errors,
voicing/devoicing and estimator range; extracted F0 is not ground truth. Zero
frames remain unvoiced, with no interpolation. Global median/range, RMS and
voiced fraction describe acoustic differences only. Fixed WAV duration does not
prove equal speaking rate. F0 and RMS do not establish speaker identity.

**No equal-duration mora assumption or automatic accent success verdict is used.**
For local evidence, manually verify the transcript and supply boundaries:

```json
{
  "ame-c0-s0-reading-nna-r0": {
    "reviewer": "reviewer identifier",
    "transcript_checked": true,
    "moras": [
      {"label": "a", "start": 0.20, "end": 0.35},
      {"label": "me", "start": 0.35, "end": 0.55},
      {"label": "ga", "start": 0.55, "end": 0.70}
    ]
  }
}
```

These times are illustrative, not annotations for the saved WAVs. Pass this as
`--annotations annotations.json` only after actual review. The tool summarises
voiced frames within each interval and adjacent drops in semitones; it still
leaves `accent_verified=false`. Listen blind, compare expected fall location,
initial rise, particle pitch, naturalness, surrounding words and speaker/style.
Have at least two competent Japanese listeners adjudicate ambiguous examples.
Do not infer flat vs final accent from an isolated word alone.

## Postprocessing investigation

`postprocess.py` is an opt-in WORLD resynthesis probe, not a dictionary backend.
It requires manually checked windows and an explicitly identified reviewer:

```json
{"reviewer":"reviewer identifier","mora_alignment_checked":true,
 "windows":[{"start":0.20,"end":0.55,"semitones":-4}]}
```

```sh
python -m experiments.dictionary_prosody.postprocess input.wav windows.json output.wav
```

It also writes a no-shift WORLD roundtrip control to separate vocoder artefacts
from F0-edit artefacts. Hann-tapered shifts are limited to +/-6 semitones and
preserve unvoiced **parameter** frames. This does not guarantee that reconstructed
unvoiced waveforms remain bit-identical. It does not align phonemes, identify
moras, set a linguistically correct contour or assess naturalness.

A reproducible synthetic-signal sanity check is available:

```sh
python -m experiments.dictionary_prosody.synthetic_postprocess /tmp/synthetic-prosody
```

Its known harmonic source and time window test F0 manipulation/resynthesis only;
they are never evidence of TTS lexical-accent control.

## Tests and schema proposal

```sh
python -m unittest discover -s tests -v
python -m ruff check .
```

Research tests cover conservative mora segmentation, 0/initial/medial/final
nuclei, particles, range/type errors, contracted/foreign small kana, sokuon,
nasal/long mora, NFKC, experimental labelling, v1 compatibility, non-recursion,
unvoiced preservation and refusal to infer accent without alignment. The
original dictionary/runtime tests also run unchanged.

`probes.validate_entry` accepts the proposed per-entry accent object only in
this research namespace. It is **not** a version 2 production loader. Production
continues to reject version 2. A future schema should require an explicit,
validated backend and reject unsupported targets/contexts instead of silently
ignoring accent. See REPORT for the proposed acceptance gate.

The saved boundary/script probe is reproduced with:

```sh
python -m experiments.dictionary_prosody.run --output /tmp/prosody-boundaries --checkpoint /path/to/model.safetensors --codec /path/to/weights.pth --words ame --contexts 1 --seeds 0 --methods katakana space comma --repeats 1 --steps 40 --seconds 2 --threads 4
python -m experiments.dictionary_prosody.summarise experiments/dictionary_prosody/results
```

The summariser counts only generated rows, checks exact-input control PCM hashes,
and keeps acoustic observations separate from lexical-accent acceptance. The
saved data cover a selected 40-clip subset, not the entire 720-row default plan.
For multi-word result sets, use `plot RUN --word ame` (or `hashi`, `kokoro`) to
avoid overlaying unrelated words. See [LISTENING.md](LISTENING.md) for the pending
blind listening/alignment procedure.

## Provisional alignment and listening tools

See [FOLLOWUP.md](FOLLOWUP.md) for the optional CTC alignment of all 40 saved
clips, its limitations/licence, and a local blind listening/annotation kit. The
estimates remain unreviewed and do not enable a production accent backend.

Further machine-only checks, matched negative controls and extractor comparisons: [ROBUSTNESS.md](ROBUSTNESS.md).

## Publication privacy note

The original pre-publication Git commit identifier has been withheld from saved
run manifests and reports because that history contained private test data.
A null `source_commit` with `source_commit_redacted=true` records this omission;
it does not identify a new generation run. Existing audio, seeds, request settings,
source-file hashes and measurements are unchanged. The historical dictionary test
vocabulary was replaced for publication. No new synthesis is claimed.
