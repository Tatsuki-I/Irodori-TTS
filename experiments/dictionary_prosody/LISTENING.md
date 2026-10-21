# Pending listening/alignment acceptance

No auditory judgement was performed in the saved results. This sheet is a
follow-up protocol, not an evaluation result.

1. Randomise playback order and hide method/nucleus from the listener. Use each
   clip at the same playback level; do not pitch-normalise or time-stretch it.
2. Confirm the spoken transcript, no symbol/letter vocalisation, no extra/missing
   mora and no truncation. Exclude failed transcripts from accent scoring rather
   than treating a different word as a pitch failure/success.
3. Mark word onset/end and all target moras plus the following `が`. Do not divide
   the word into equal-duration bins. Inspect waveform/spectrogram and listen.
4. Label the perceived Tokyo accent independently of the target, or mark uncertain.
   Distinguish lexical fall from a pause/reset, question contour and phrase ending.
5. Rate naturalness, speaker identity consistency, emotion, rate, loudness and
   surrounding-word delivery. Compare to both no-dictionary and reading-only arms.
6. Use a second competent Japanese listener for independent labels. Report
   disagreement and missing/unvoiced F0; do not count uncertain cases as success.
7. Re-run additional seeds, new words/contexts, automatic duration and fixed
   reference voice before generalising. Saved short no-reference clips are a pilot.

Suggested per-clip record:

```json
{
  "clip_id": "...",
  "reviewer": "...",
  "transcript_checked": false,
  "perceived_nucleus": null,
  "uncertain": true,
  "naturalness_1_to_5": null,
  "identity_changed": null,
  "emotion_changed": null,
  "rate_changed": null,
  "surrounding_words_changed": null,
  "moras": [],
  "notes": "Pending review"
}
```

Use manually checked intervals with `analyse --annotations`. A reliable local
fall plus an intelligible/natural perceptual contrast across held-out conditions
is required to promote any probe to a supported backend. A whole-utterance F0
shift, or a changed waveform hash alone, is insufficient.
