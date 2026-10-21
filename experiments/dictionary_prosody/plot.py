"""Plot observed F0 without inventing mora boundaries or accent labels."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--word", help="Filter an ASCII word ID from cases.json")
    args = parser.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    analysis = json.loads((args.run / "analysis.json").read_text(encoding="utf-8"))
    rows = [
        row
        for row in analysis["results"]
        if args.word is None or row["id"].split("-")[0] == args.word
    ]
    if not rows:
        parser.error("No matching generated results")
    methods = list(dict.fromkeys(row["method"] for row in rows))
    fig, axes = plt.subplots(len(methods), 1, figsize=(10, 2.5 * len(methods)), squeeze=False)
    for ax, method in zip(axes[:, 0], methods, strict=True):
        for row in rows:
            if row["method"] != method:
                continue
            with (args.run / row["f0_csv"]).open(encoding="utf-8") as handle:
                values = list(csv.DictReader(handle))
            times = [float(x["time_seconds"]) for x in values]
            f0 = np.array([float(x["f0_hz"]) for x in values])
            f0[f0 == 0] = np.nan
            ax.plot(times, f0, label=row["id"], linewidth=1)
        ax.set(title=method, xlabel="Time (s), NOT mora-aligned", ylabel="F0 (Hz)", ylim=(60, 600))
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(alpha=0.2)
    fig.suptitle("Observed WORLD F0 — lexical accent NOT verified", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(args.run / ("f0.png" if args.word is None else f"f0-{args.word}.png"), dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
