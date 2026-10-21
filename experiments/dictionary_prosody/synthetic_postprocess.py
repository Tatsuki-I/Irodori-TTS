"""WORLD manipulation sanity check on synthetic tones, never TTS accent evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .postprocess import modify_f0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    import pyworld
    import soundfile as sf

    args.output.mkdir(parents=True, exist_ok=False)
    sr = 24000
    times = np.arange(sr) / sr
    # Stationary harmonic source with a known silent interval, not a human voice.
    x = (
        sum(np.sin(2 * np.pi * 180 * harmonic * times) / harmonic for harmonic in range(1, 30))
        * 0.1
    )
    x *= np.sin(np.pi * times) ** 2
    x[(times >= 0.45) & (times <= 0.55)] = 0
    x = np.ascontiguousarray(x)
    f0, t = pyworld.harvest(x, sr, f0_floor=60, f0_ceil=600, frame_period=5)
    # DIO is included as a diagnostic because Harvest may reject synthetic tones.
    method = "harvest"
    if (f0 > 0).sum() < 20:
        f0, t = pyworld.dio(x, sr, f0_floor=60, f0_ceil=600, frame_period=5)
        f0 = pyworld.stonemask(x, f0, t, sr)
        method = "dio+stonemask"
    changed = modify_f0(t, f0, [{"start": 0.1, "end": 0.4, "semitones": -4}])
    sp, ap = pyworld.cheaptrick(x, f0, t, sr), pyworld.d4c(x, f0, t, sr)
    sf.write(args.output / "synthetic-original.wav", x, sr, subtype="FLOAT")
    metrics = {}
    for label, contour in [("roundtrip", f0), ("shifted", changed)]:
        y = pyworld.synthesize(contour, sp, ap, sr, frame_period=5)
        if not np.isfinite(y).all():
            raise ValueError("Non-finite synthesis")
        sf.write(args.output / f"synthetic-{label}.wav", y, sr, subtype="FLOAT")
        measured, measured_t = pyworld.dio(y, sr, f0_floor=60, f0_ceil=600, frame_period=5)
        mask = (measured_t >= 0.22) & (measured_t <= 0.28) & (measured > 0)
        metrics[label] = {
            "centre_f0_hz": float(np.median(measured[mask])) if mask.any() else None,
            "frames": int(mask.sum()),
        }
    before, after = metrics["roundtrip"]["centre_f0_hz"], metrics["shifted"]["centre_f0_hz"]
    report = {
        "scope": "synthetic harmonic signal only; no phonemes or lexical accent",
        "input_f0_hz": 180,
        "extractor": method,
        "requested_peak_shift_semitones": -4,
        "measured_centre_shift_semitones": None
        if before is None or after is None
        else float(12 * np.log2(after / before)),
        "unvoiced_parameter_frames_unchanged": bool(np.array_equal(changed[f0 == 0], f0[f0 == 0])),
        "metrics": metrics,
        "accent_verified": False,
    }
    (args.output / "analysis.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
