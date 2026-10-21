# Automated evidence audit and negative controls

## Decision

The additional machine-only checks **do not establish a production accent-control
backend**. They explain why a shape match cannot be promoted to a successful
Tokyo accent instruction. They also preserve useful positive observations:
the two earlier candidates are locally stable under the tested boundary shifts
and both F0 estimators. No listening scores or verified labels were introduced.

This is a post-hoc investigation of the existing 40 WAVs, not a new synthesis
trial, a pre-registered evaluation or a population accuracy estimate. No TTS
checkpoint was loaded in this follow-up. The production inference implementation,
version 1 dictionaries, original WAVs and existing alignment records are unchanged.

## Integrity and matched comparisons

`robustness.py` checks WAV file hashes, decoded float32 PCM hashes against the
saved acoustic analyses, sample rate/duration metadata, F0 time ordering/voicing,
unique alignment coverage, and agreement of alignment/manifest identity and
conditions. SHA-256 hashes of analysis inputs and generation code are recorded.
These checks bind this analysis to the saved evidence; hashing alone does not
verify a linguistic transcript or prove that the original F0 extractor was right.

All six duplicate-condition groups have identical PCM. This is repeatability of
these particular saved conditions, not a cross-platform determinism guarantee.

For every requested-nucleus condition, a reading-only control is selected from
the same run, word, context, seed and repeat. The same requested nucleus is then
scored on that control. All 30 pairs have enough voiced frames for the raw screen.

| Raw screen, using the original 1.5/3-semitone thresholds | Matches |
| --- | ---: |
| Intervention audio at its requested nucleus | 4 / 30 conditions |
| Matched reading-only audio at that same nucleus | 5 / 30 conditions |
| Intervention matches, control does not | 2 conditions |
| Control matches, intervention does not | 3 conditions |

These raw counts deliberately omit the earlier CTC-confidence/CER/stability gates;
the earlier gated match count remains **2**, not 4. The same reading control is
reused for several requested nuclei, so the 30 controls are **not independent
utterances**. No significance test, causal effect or calibrated false-positive
rate is claimed. These results supply no evidence of an aggregate improvement
from the tested instructions, but they cannot establish equivalence or prove
that all possible prompting strategies fail.

## A match may not identify one nucleus

Each WAV is scored against **every** integer nucleus in `0..N`, including
hypothetical nuclei never requested for that particular word. This is an
adversarial counterfactual screen, not additional generated conditions.

Two of the 40 WAVs match multiple nuclei with the default thresholds. The reason
is structural: a falling transition of 2 semitones passes both the minimum fall
of 1.5 semitones and the allowed same-level drift of 3 semitones. The heuristic's
acceptance regions overlap. It is not a nucleus classifier.

For `ame-c0-s0-arrows-n0-r0`, the earlier flat-type candidate:

- Both nucleus **0 and 2** match the same saved F0, including the following particle.
- Lowering same-level tolerance to 2 semitones removes nucleus 0; nucleus 2 remains.
- The expected/greedy transcript discrepancy (`amegaaru` / `anegaaru`) remains unresolved.

Thus having the requested nucleus in a set of matches is insufficient evidence
that the intended flat/final contrast was generated.

## Boundary and threshold stress tests

Each target/particle boundary, including the outer endpoints, is independently
offset by `-radius`, `0`, or `+radius`. Shared boundaries stay shared. Non-monotonic
or out-of-audio grids are excluded. Radii are 20, 40 and 80 ms. These are exhaustive
finite perturbation grids around the CTC estimate, **not a probability model or
an upper bound on phonetic alignment error**. Changing each boundary independently
is stronger than shifting all intervals together, but still cannot verify them.

For the two previously gated candidates:

| Candidate | +/-20 ms | +/-40 ms | +/-80 ms |
| --- | --- | --- | --- |
| `hashi-c0-s1-downstep-n2-r0` | 81 matches / 81 valid grids | 81 / 81 | 72 matches, 9 insufficient-voicing grids |
| `ame-c0-s0-arrows-n0-r0` | 81 / 81 | 81 / 81 | 81 / 81 |

These local stability results are useful but do not resolve transcript accuracy,
phrase-accent effects or the overlapping acceptance regions.

A separate 3x3 threshold grid combines minimum transitions of 1, 1.5 and 2
semitones with same-level tolerances of 2, 3 and 4 semitones. Full per-clip nucleus
sets are saved, rather than selecting whichever setting produces most matches.
The `hashi` nucleus-2 candidate is unique at all nine settings. The `ame` nucleus-0
candidate loses its requested match in all three settings with tolerance 2.
Thresholds are exploratory; none has been fitted or validated as a Japanese
accent recognition criterion.

## F0 extractor cross-check

`crosscheck.py` uses WORLD **DIO + StoneMask** on the same WAVs, at 5 ms spacing
and 60–600 Hz, and applies the same unreviewed mora intervals. It also re-extracts
Harvest from the saved WAVs. The report retains the alternative F0 arrays, frame
grid, per-clip voicing/pitch disagreements, hashes and package versions.

| Raw intended-nucleus screen | Matches |
| --- | ---: |
| Saved Harvest | 4 / 30 |
| DIO + StoneMask | 6 / 30 |
| WAVs whose full matching-nucleus sets differ | 6 / 40 |

Harvest was re-extracted from all 40 WAVs and reproduced every saved F0 array
exactly in this environment.

Both previous gated Harvest candidates keep the same matching-nucleus sets under
DIO/StoneMask: `{2}` for `hashi`, `{0,2}` for `ame`. Jointly voiced frames have
median absolute extractor differences of approximately 0.060 and 0.081 semitones,
respectively; however, voicing decisions disagree on about 21% and 23% of all
frames. Across clips, the median voicing-disagreement fraction is about 17%.
A small F0 difference on jointly voiced frames does not imply agreement about
where reliable F0 exists.

Both estimators belong to WORLD and share the same unreviewed alignment and
shape heuristic. Agreement is an acoustic robustness observation, **not an
independent Japanese lexical-accent evaluation**. Neither extractor comparison
measures voice identity, naturalness or perceived prosodic intent.

## Reproduce

Use an Irodori environment plus the optional research analysis dependencies.
The executed analysis uses Python 3.12/Linux/CPU, NumPy 2.3.5, SoundFile 0.14.0
and PyWORLD 0.3.5. PyWORLD 0.3.5 imports the deprecated `pkg_resources`; this run
used setuptools 80.10.2 (`setuptools<81`). This compatibility pin belongs only to
the optional analysis environment. No production lock file was changed.

```sh
python -m experiments.dictionary_prosody.robustness --output outputs/robustness.json
python -m experiments.dictionary_prosody.crosscheck --output outputs/extractor-crosscheck.json
```

Output filenames must not already exist. Compare the result objects and source
hashes when rerunning. Platform/compiler/library changes may alter numerical F0
results; recorded results are observations on the specified environment.

Committed records:

- `results/robustness.json`: complete raw nucleus sets, threshold grids, boundary
  perturbation counts and matched reading-only comparisons.
- `results/extractor-crosscheck.json`: DIO/StoneMask trajectories and acoustic
  comparisons with the saved/replayed Harvest.

## Validation and remaining scope

The 56 Python tests pass (45 earlier + 11 added). New checks cover idealised
contours, the counterexample where two nuclei match, abstention, independently
perturbed boundaries, invalid F0 data, real saved WAV/PCM validation, matched
controls, tampered evidence and extractor comparison metrics. Repository Ruff
and Python 3.10 syntax checks also pass. Tests used Torch 2.14.1+cpu and
TorchAudio 2.11.0+cpu for the existing small-model tests; no released-checkpoint
TTS synthesis was rerun with those library versions. No Windows/GPU execution
or human auditory assessment is claimed.

Machine-only work has strengthened the provenance and falsification checks. The
remaining acceptance gap is correct target reading, correct phonetic alignment,
perceived nucleus and naturalness, then held-out words/contexts/seeds. Do not
convert these screens into production acceptance criteria or quietly accept
ignored accent metadata. A dedicated controllable condition or a validated
alignment/post-processing backend remains a separate implementation project.
No upstream PR, issue, comment or maintainer message was created.
