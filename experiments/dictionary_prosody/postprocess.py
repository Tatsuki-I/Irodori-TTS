"""Research-only WORLD resynthesis for manually aligned F0 windows."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def modify_f0(times, f0, windows: list[dict]):
    """Apply tapered local shifts; preserve all unvoiced frames exactly."""
    times, f0 = np.asarray(times), np.asarray(f0)
    if (
        times.shape != f0.shape
        or times.ndim != 1
        or not len(times)
        or not np.isfinite(times).all()
        or not np.isfinite(f0).all()
        or (f0 < 0).any()
        or (np.diff(times) <= 0).any()
    ):
        raise ValueError("Invalid F0/time series")
    result = f0.astype(float, copy=True)
    previous_end = 0.0
    for window in windows:
        start, end, shift = window["start"], window["end"], window["semitones"]
        if (
            not all(math.isfinite(v) for v in (start, end, shift))
            or start < previous_end
            or end <= start
            or end > times[-1]
            or abs(shift) > 6
        ):
            raise ValueError("Disjoint in-range windows and shifts within +/-6 semitones required")
        selected = (times >= start) & (times <= end) & (f0 > 0)
        # Hann taper avoids a hard discontinuity at the boundaries.
        phase = (times[selected] - start) / (end - start)
        result[selected] *= 2 ** ((shift * np.sin(np.pi * phase) ** 2) / 12)
        previous_end = end
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav", type=Path)
    parser.add_argument("windows", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    import pyworld
    import soundfile as sf

    if args.output.exists() or args.output.with_name(args.output.stem + "-roundtrip.wav").exists():
        parser.error("Output and roundtrip files must not already exist")
    document = json.loads(args.windows.read_text(encoding="utf-8"))
    if not document.get("reviewer") or document.get("mora_alignment_checked") is not True:
        parser.error(
            "Require reviewer and mora_alignment_checked=true; no automatic alignment is assumed"
        )
    x, sr = sf.read(args.wav, dtype="float64", always_2d=True)
    if sr < 16000 or not np.isfinite(x).all():
        parser.error("Require finite audio at >=16 kHz")
    x = np.ascontiguousarray(x.mean(axis=1))
    f0, times = pyworld.harvest(x, sr, f0_floor=60, f0_ceil=600, frame_period=5)
    sp = pyworld.cheaptrick(x, f0, times, sr)
    ap = pyworld.d4c(x, f0, times, sr)
    changed = modify_f0(times, f0, document["windows"])
    for suffix, contour in [("-roundtrip", f0), ("", changed)]:
        y = pyworld.synthesize(contour, sp, ap, sr, frame_period=5)
        if not np.isfinite(y).all():
            raise ValueError("Non-finite WORLD output")
        sf.write(
            args.output.with_name(args.output.stem + suffix + args.output.suffix),
            y,
            sr,
            subtype="FLOAT",
        )
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "experimental_only": True,
                "accent_verified": False,
                "alignment": document,
                "unvoiced_preserved": bool(np.array_equal(f0[f0 == 0], changed[f0 == 0])),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
