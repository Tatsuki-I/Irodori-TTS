"""Original small CTC Viterbi implementation for provisional research alignment."""

from __future__ import annotations

import numpy as np


def viterbi_align(log_probs, targets: list[int], *, blank: int = 0) -> list[dict]:
    """Find the best complete CTC path, including mandatory repeat blanks.

    Returned spans are character posterior anchors, NOT phoneme boundaries.
    Input is normalised log probabilities, frames x vocabulary. No external
    aligner source code or deprecated TorchAudio alignment API is used.
    """
    scores = np.asarray(log_probs, dtype=float)
    if scores.ndim != 2 or min(scores.shape) < 1 or not np.isfinite(scores).all():
        raise ValueError("Expected finite frames x vocabulary log probabilities")
    if type(blank) is not int or not 0 <= blank < scores.shape[1]:
        raise ValueError("Invalid blank index")
    if (
        not targets
        or len(targets) > 500
        or any(type(t) is not int or not 0 <= t < scores.shape[1] or t == blank for t in targets)
    ):
        raise ValueError("Expected 1..500 non-blank target indices")
    if not np.allclose(np.exp(scores).sum(axis=1), 1, atol=1e-5):
        raise ValueError("Expected normalised log probabilities")
    minimum = len(targets) + sum(a == b for a, b in zip(targets, targets[1:], strict=False))
    if len(scores) < minimum:
        raise ValueError("Too few frames for a complete CTC path")
    labels = [blank]
    for target in targets:
        labels.extend([target, blank])
    states = len(labels)
    previous = np.full(states, -np.inf)
    previous[0], previous[1] = scores[0, blank], scores[0, targets[0]]
    parents = np.full((len(scores), states), -1, dtype=np.int32)
    for frame in range(1, len(scores)):
        current = np.full(states, -np.inf)
        for state, label in enumerate(labels):
            candidates = [state]
            if state >= 1:
                candidates.append(state - 1)
            if state >= 2 and label != blank and label != labels[state - 2]:
                candidates.append(state - 2)
            predecessor = candidates[int(np.argmax(previous[candidates]))]
            current[state] = previous[predecessor] + scores[frame, label]
            parents[frame, state] = predecessor
        previous = current
    state = states - 2 if previous[-2] >= previous[-1] else states - 1
    if not np.isfinite(previous[state]):
        raise ValueError("No complete CTC path")
    path = np.empty(len(scores), dtype=np.int32)
    for frame in range(len(scores) - 1, -1, -1):
        path[frame] = state
        if frame:
            state = int(parents[frame, state])
    result = []
    for index, token in enumerate(targets):
        frames = np.flatnonzero(path == 2 * index + 1)
        if not len(frames):
            raise ValueError("Missing target on CTC path")
        result.append(
            {
                "token": token,
                "start_frame": int(frames[0]),
                "end_frame": int(frames[-1] + 1),
                "mean_posterior": float(np.exp(scores[frames, token]).mean()),
            }
        )
    return result


def edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(previous[j] + 1, current[-1] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def greedy_text(log_probs, vocabulary: dict[str, int], *, blank: int = 0) -> str:
    reverse = {value: key for key, value in vocabulary.items()}
    chars = []
    previous = None
    for token in np.asarray(log_probs).argmax(axis=1).tolist():
        if token != previous and token != blank:
            chars.append(reverse[token])
        previous = token
    return "".join(chars)


def mora_intervals(
    spans: list[dict],
    units: list[dict],
    *,
    stride_seconds: float,
    centre_offset_seconds: float,
    duration: float,
) -> list[dict]:
    """Group Latin character anchors into declared expected mora units.

    Internal boundaries are midpoints between adjacent units' anchor centres.
    This explicitly provisional policy is not an acoustic phoneme segmentation.
    """
    if (
        sum(len(unit["latin"]) for unit in units) != len(spans)
        or not units
        or stride_seconds <= 0
        or centre_offset_seconds < 0
        or duration <= 0
    ):
        raise ValueError("Invalid unit/anchor mapping")
    previous_end = 0
    for span in spans:
        start, end = span.get("start_frame"), span.get("end_frame")
        if type(start) is not int or type(end) is not int or not previous_end <= start < end:
            raise ValueError("Character anchors must be ordered non-negative frame intervals")
        previous_end = end
    groups = []
    offset = 0
    for unit in units:
        part = spans[offset : offset + len(unit["latin"])]
        if not part:
            raise ValueError("Empty unit")
        first = part[0]["start_frame"] * stride_seconds + centre_offset_seconds
        last = (part[-1]["end_frame"] - 1) * stride_seconds + centre_offset_seconds
        groups.append(
            unit
            | {
                "first_anchor": first,
                "last_anchor": last,
                "mean_posterior": float(np.mean([p["mean_posterior"] for p in part])),
            }
        )
        offset += len(part)
    for index, group in enumerate(groups):
        group["start"] = (
            max(0.0, group["first_anchor"] - stride_seconds / 2)
            if index == 0
            else (groups[index - 1]["last_anchor"] + group["first_anchor"]) / 2
        )
        group["end"] = (
            min(duration, group["last_anchor"] + stride_seconds / 2)
            if index == len(groups) - 1
            else (group["last_anchor"] + groups[index + 1]["first_anchor"]) / 2
        )
        if not 0 <= group["start"] < group["end"] <= duration:
            raise ValueError("Non-monotone or out-of-audio estimated boundaries")
    return groups
