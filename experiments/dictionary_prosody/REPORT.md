# Pitch-accent control investigation

## Decision

**Do not expose a production accent dictionary backend on the evidence collected
here.** Existing input perturbations change real generated F0, but that does not
establish reliable control of the intended Tokyo lexical-accent nucleus.
The generation experiments use the actual released checkpoint and actual codec;
no model weights or production source code were modified. The version 1 feature
branch remains recoverable at `pre-publication research baseline`.

This conclusion is **not** proof that every possible prompt fails. It means the
minimum acceptance evidence for a deterministic user-facing accent feature is
missing. Unsupported accent metadata is not accepted and silently discarded:
the production loader continues to reject version 2.

## Source analysis: text to waveform

The analysis is anchored to repository commit `pre-publication research baseline` and the exact public
checkpoint revision recorded in `cases.json` and the run manifests.

| Stage | Actual implementation | Implication |
| --- | --- | --- |
| Normalisation | `text_normalization.normalize_text`: replacements, outer-bracket stripping, NFKC and ellipsis handling | Some candidate symbols disappear/transform before tokenisation; dictionary output comes afterwards |
| User dictionary | `prepare_text` normalises/trims, then applies literal leftmost-longest v1 replacements | A reading changes orthographic conditioning, not an acoustic F0 parameter |
| Tokenisation | Bundled tokenizer JSON: Unigram model, Metaspace pre-tokeniser, BOS and right padding | A mora is not necessarily a token; a tokenisable marker has no guaranteed learned meaning |
| Text encoder | `PretrainedTextBackbone` loads ModernBERT configuration from checkpoint metadata and emits contextual token states | It accepts token IDs/masks, not mora/nucleus labels |
| Text/caption projectors | The pretrained backbone is shared; separate trainable residual projectors/norms produce the two condition sequences | Caption has a distinct conditioning route but no local accent binding interface |
| Speaker branch | Reference codec latents or speaker-inversion embedding tokens | Can carry identity/style/prosody implicitly; there is no nucleus selector |
| Duration | Token-sum duration predictor with speaker and caption fusion; runtime converts predicted log frame count into target latent length | Total output length is controlled, not an exposed phoneme/mora-to-time alignment or per-mora F0 trajectory |
| RF generation | `sample_euler_rf_cfg`, seeded latent noise, text/speaker/caption conditions, repeated DiT velocity prediction | Accent/prosody must emerge implicitly from learned conditioning |
| Codec | Unpatchify latent sequence and DACVAE decode; optional SilentCipher after decode | No explicit pitch manipulation in the dictionary/runtime path |

The checkpoint configuration has `text_encoder_type=pretrained`,
`text_tokenizer_repo=sbintuitions/modernbert-ja-310m`, text/caption dimensions 512,
12 DiT blocks of dimension 1280, latent dimension 32, and the token-sum dual
AdaRN-Zero duration architecture. These values were read from the actual
safetensors metadata, saved as `results/checkpoint-metadata.json`, rather than
assuming the training YAML exactly describes the published weights.

`forward_with_encoded_conditions` receives latent state, diffusion time,
text/caption/speaker states and masks, optional context cache and MeanFlow
interval time. None is a dedicated nucleus, mora, phoneme, F0 or accent-phrase
condition. This is a source/API finding; it does not assert the hidden states
contain no prosodic information.

The v4.1 model card says the update retrained/replaced only the duration
predictor; the RF-DiT and condition encoders remained the same. It also warns
about conflicting caption/reference conditions and prompt-adherence limitations.

## Tokyo accent representation and limits

The proposed data model is an optional `{dialect: "tokyo", nucleus: integer}`.
Zero means unaccented; `1..N` identifies the mora immediately before the lexical
pitch fall. For a three-mora citation form, idealised contours are:

| Nucleus | Isolated word | With a following particle |
| --- | --- | --- |
| 0 | LHH | LHHH |
| 1 | HLL | HLLL |
| 2 | LHL | LHLL |
| 3 | LHH | LHHL |

These are teaching abstractions, not absolute Hz targets. Initial lowering,
phrase-level declination, focus, devoicing and accent-phrase context affect the
surface contour. Flat/final contrasts need a following particle. Compound nouns,
conjugation and phrase joining can alter accent; literal substring replacement
cannot infer those boundaries. A word-level integer alone cannot settle compound
or inflected surface accent. Those contexts need either explicit supported
restrictions or a morphological/phrase frontend.

The experimental kana segmenter normalises NFKC and katakana, joins supported
small-kana combinations, and counts `っ`, `ん`, `ー` as separate moras. It rejects
unknown spellings. This validates an integer range only: it is not a complete
phonological legality checker, analyser or accent sandhi engine. In particular,
a valid in-range index is not a promise that every special-mora nucleus is a
natural citation-form accent. Devoiced nuclei also cannot be evaluated by simply
requiring a voiced F0 sample at that mora.

## Approaches compared

| Approach | Work/cost | Reproducibility and side effects | Decision |
| --- | --- | --- | --- |
| A: kana script, downstep, arrows, spacing, comma | Small external text adapter; no model changes | Deterministic strings, but acoustics are learned/unknown. Symbols change segmentation; punctuation may add pauses or phrase resets. Katakana alone does not encode a nucleus | Research probes only; no validated accent protocol |
| B: Japanese caption instruction | Small caption suffix, unchanged generation code | May alter global pitch, pacing, voice/style; cannot bind a nucleus to an exact local acoustic span through an existing typed API | Acoustic differences observed, local correctness unverified |
| C: reuse condition states/CFG/reference | Existing hooks can be inspected with no changes | CFG adjusts condition strength; arbitrary state edits or reference changes have no calibrated nucleus semantics. Duration supplies length, not alignment | No inference-only, semantically defined nucleus extension found |
| D: WORLD resynthesis | Optional external analysis package and local F0 edits | Does not require new TTS weights, but requires phonetic alignment and robust contour design; vocoder roundtrip can degrade quality | Synthetic manipulation works; natural lexical-accent correction unverified |

### Why an integer cannot simply be added to the existing model

A JSON field is invisible to the model unless translated to an input it has
learned to use. A new accent embedding or F0 condition adds parameters/checkpoint
semantics and normally needs supervised training. Arbitrarily modifying
ModernBERT states or DiT cross-attention would not supply that learned mapping.
An annotation-based adapter/LoRA could potentially learn a text-marker protocol,
but requires labelled contrastive examples and held-out acoustic/human evaluation;
there is no evidence here that adding tokens alone is sufficient.

## Reproduction and evidence interpretation

The run commands, assets, hashes, conditions and limitations are documented in
[README.md](README.md). The actual WAVs, per-frame F0 CSVs and manifests are under
`results/`. Checkpoint and codec assets themselves are not committed.

Primary comparisons fix CPU FP32, 40 RF steps, 2-second manual duration, no tail
trim, neutral Japanese caption, no reference, and all other request settings.
Only the documented text rendering or caption suffix varies. Caption conditions
therefore deliberately differ in arm B; that difference is the intervention.
Automatic-duration behaviour is inspected structurally, not claimed acoustically
validated by these fixed-duration trials. Reference-conditioned acoustics are
also not validated here.

WAV-container hashes can differ due to metadata while decoded PCM remains
identical. Reading-only vs unmarked downstep/zero is an exact-input negative
control. Use `analysis.json` PCM hashes for repeatability, not container hashes.
Identical controls within one loaded runtime do not establish cross-device,
cross-version or multi-process determinism.

No manually checked phonetic alignment or independent native-listener judgement
was available in this execution. The analysis therefore deliberately leaves
`accent_verified=false` for every natural-speech clip. It reports no unsupported
accent accuracy score. Neither global F0 nor a visible fall can establish the
intended lexical accent without a trustworthy word/mora location and transcript.
Naturalness, word identity, timbre, emotion and surrounding-word preservation
remain listening/alignment acceptance items, not presumed passes.

## Synthetic postprocessing result

A known 180 Hz harmonic signal, a silent interval and a manually known synthetic
time window were analysed and resynthesised. A Hann-tapered -4-semitone peak edit
produced approximately **-3.889 semitones** at the measured central window:
roundtrip 179.975 Hz versus shifted 143.767 Hz. Unvoiced parameter frames were
preserved. The no-shift roundtrip, shifted waveform and metrics are saved in
`results/synthetic-postprocess/`.

This establishes a narrow DSP sanity check only. It is not human speech, does
not test mora alignment, and does not establish natural-sounding accent repair.
The production dictionary has no postprocessing dependency or hook added.

## Future schema and smallest viable implementation options

`schema-proposal.json` shows a **reserved, unsupported** version 2 form. If a
backend is later demonstrated, retain version 1 unchanged, allow optional accent
per v2 entry, reject booleans/non-integer/out-of-range nuclei and unknown dialects,
and preserve all existing duplicate/Unicode/longest-match rules. The runtime
must select/declare a supported backend explicitly; unsupported accent requests
must fail before model execution. Do not guess a backend from dictionary content.

Minimum viable choices, in increasing modelling impact:

1. Establish a prompt/notation protocol through held-out, multiple-seed/multiple-
   context tests plus native listening. If this succeeds, add only a small,
   explicitly scoped input adapter, not a new acoustic model field.
2. Keep Irodori untouched and build an optional aligned F0-repair adapter. It needs
   phoneme/mora timing, unvoiced handling, accent-phrase contours and a codec
   roundtrip naturalness gate. This does not inherently require TTS retraining,
   but is materially more engineering than accepting a dictionary integer.
3. Train a lightweight adapter on explicit accent annotations, or add a typed
   accent embedding/condition and train it. Use paired homophones/nucleus variants,
   particle contexts, different voices and held-out vocabulary; determine data
   volume by learning curves rather than inventing a sample-count guarantee.

## Upstream scope recommendation

Submit the already separate **version 1 reading-dictionary feature** for review
when authorised, with its existing validation limitations. Keep this investigation
on `experiment/dictionary-prosody`. Do not present v2 accent support as complete.
A later accent PR should include an actually validated backend, declared context
restrictions, aligned local acoustic evidence and native listening results.
No PR, draft PR, issue or comment was created as part of this experiment.

## Primary references

- [Irodori v4.1 Small model card](https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small/blob/2b28324dc263ed5e6638b3cf3dd94c82ead07b4b/README.md)
- [Original implementation baseline](../../irodori_tts/pronunciation_dictionary.py)
- [Tokyo accent and OJAD teaching material, University of Tokyo](https://www.gavo.t.u-tokyo.ac.jp/~mine/japanese/CMP/OJAD_Kashiwa2018W.pdf)
- [WORLD implementation](https://github.com/mmorise/World)
- [PyWORLD API and analysis/resynthesis examples](https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder)

## Pilot measurements (observations, not accent scores)

Actual checkpoint, seed 0, input `雨がある。` (rendered as kana for dictionary
arms), neutral fixed caption except the caption arms:

| Arm | Intended nucleus | Whole-clip voiced median F0 (Hz) | 10–90% voiced pitch range (semitones) |
| --- | ---: | ---: | ---: |
| No dictionary | — | 161.96 | 17.32 |
| Reading only | — | 169.11 | 10.55 |
| Downstep | 0 | 169.11 | 10.55 |
| Downstep | 1 | 192.56 | 14.87 |
| Arrow | 0 | 158.08 | 10.06 |
| Arrow | 1 | 159.63 | 17.62 |
| Caption | 0 | 167.74 | 18.41 |
| Caption | 1 | 192.83 | 18.49 |

The downstep/zero arm has exactly the reading-only input and bit-identical
decoded samples, a successful negative control. The marked nucleus-1 string
changes tokenisation from `[1,50867,956,278]` to
`[1,1516,249,171,171,2582,956,278]`: it is ordinary token conditioning, not a
new dedicated accent channel. Arrow and caption arms also change generated PCM.

The caption-1 vs reading-only whole-clip median rises by about 2.27 semitones.
This is an observed global acoustic difference, **not** a proof of timbre change,
and **not** proof of the required first-mora fall. The curves in
`results/pilot/f0.png` intentionally have no guessed mora labels. A local accent
score, naturalness judgement or speaker-similarity claim is not justified from
these summary statistics.

## Completed run inventory and validation

| Saved run | Natural TTS WAVs | Words | Context | Seed | Interventions |
| --- | ---: | --- | --- | --- | --- |
| pilot | 8 | ame | target + がある | 0 | original, reading, downstep, arrows, caption |
| generalisation | 24 | ame / hashi / kokoro | target + がある | 1 | original, reading, downstep, caption, nucleus 0–3 where in range |
| boundaries | 8 | ame | ここに + target + がある | 0 | original, reading, katakana, space, comma |
| Total | **40** | 3 readings | 2 contexts | 2 seeds in the selected subset | All A/B probe families represented |

The full 720-row default plan was generated as a plan-only smoke test, **not**
executed. The saved sample is exploratory, not a balanced factorial study. Nucleus
and word coverage therefore do not constitute an accent-accuracy evaluation.

All natural TTS files are 2-second, 48 kHz floating-point WAVs from the public
checkpoint. Three additional WAVs are explicitly labelled synthetic DSP controls.
`results/summary.json` records generated counts, elapsed generation time and
exact-input negative controls. Those controls agree sample-for-sample; this
includes reading vs unmarked nucleus 0 and the nucleus-insensitive katakana
rendering. They prove repeatability for those inputs, not the accuracy of a
requested accent.

In the seed-1 `ame` trial, reading-only median F0 was 155.96 Hz, while caption
nucleus-0 and nucleus-1 medians were 255.76 and 265.23 Hz respectively. These
large global differences reinforce the need to test speaker/style non-interference
rather than assuming a local instruction has only local effects. They do not
establish voice-identity change or a correct nucleus. Complete unrounded values
are in the per-run JSON/CSV evidence.

Validation performed on Linux, Python 3.12.14, torch/torchaudio 2.10.0+cpu:

- 33 automated tests pass, including the 22 original dictionary/integration tests.
- Evidence-integrity tests verify every natural WAV file hash and decoded PCM
  hash against its manifest/analysis; exact-input control hashes agree.
- Repository-wide Ruff lint passes; all new Python files are formatted.
- Compileall and Python 3.10 syntax parsing pass for the new research sources.
  A Python 3.10 execution of the **new** research suite was not performed.
- Production code, training preparation, dependency/lock files and licence are
  byte-identical to `pre-publication research baseline`. Existing request/CLI behaviour remains covered by
  the unchanged baseline integration tests. No native Windows/GPU run was done.

### Self-review

The main review risk was overstating an acoustic perturbation as a successful
accent intervention. Every saved analysis retains `accent_verified=false`, the
report states the missing alignment/listening evidence, and the production loader
still rejects unsupported accent metadata. The plotting code labels time axes as
not mora-aligned. The postprocessor writes a no-shift roundtrip control and
preserves unvoiced parameters rather than claiming identical unvoiced waveforms.

Another review finding was that WAV container metadata can differ for identical
samples. Separate decoded PCM hashes were added and used for controls instead of
incorrectly treating a container hash change as a generation difference.

Generation was performed while research files were under development, which is
explicitly recorded as `source_dirty=true`. The later runs record generation
source hashes; their `run.py` and `probes.py` match the delivered versions. The
pilot predates the expanded source-hash/request-settings logging, but its model,
codec, seed, settings, text/caption, token IDs and audio hashes are preserved.
Subsequent analysis-only reporting/plotting additions are delivered in this final
commit and do not change production synthesis. Exact software package versions
and checkpoint metadata are retained under `results/`.

No blocking production regression was found. Missing lexical-accent acceptance
is a substantive limit, so no production accent feature is claimed or enabled.

## Follow-up

A subsequent research step added provisional CTC mora anchors and blind review
tools without modifying synthesis. See [FOLLOWUP.md](FOLLOWUP.md) for the new
observations, 45-test validation and remaining listening/alignment acceptance.
The original 40 WAVs/F0 measurements above are unchanged.
