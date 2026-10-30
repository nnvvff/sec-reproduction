"""Regenerate the MNIST comparison from saved scores; no model fitting."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
METHODS = ["KMeans", "SC+ridge", "SEC/SC", "SEC/LR"]


def main():
    folder = ROOT / "results/mnist"
    with (folder / "runs.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    paper = json.loads((folder / "paper_reference.json").read_text())["methods"]
    stats = []
    for method in METHODS:
        group = sorted((r for r in rows if r["method"] == method), key=lambda r: int(r["split"]))
        if len(group) != 20 or [int(r["split"]) for r in group] != list(range(20)):
            raise ValueError(f"Expected every split exactly once for {method}")
        if any(r["dataset"] != "mnist" or int(r["seed"]) != 20261029 + int(r["split"]) for r in group):
            raise ValueError("Unexpected dataset or seed")
        stat = dict(method=method, n=20)
        for phase in ("train", "test"):
            scores = np.array([float(r[f"{phase}_acc"]) for r in group]) * 100
            if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 100)):
                raise ValueError("Invalid accuracy")
            stat[f"{phase}_mean_percent"] = scores.mean()
            stat[f"{phase}_std_percent"] = scores.std(ddof=1)
        stats.append(stat)
    with (folder / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(stats[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(stats)
    table = ["| Method | Our train ACC | Paper train ACC | Our test ACC | Paper test ACC |",
             "|---|---:|---:|---:|---:|"]
    for s in stats:
        ref = paper[s["method"]]
        cells = [s["method"]]
        for phase in ("train", "test"):
            cells.extend([f"{s[f'{phase}_mean_percent']:.2f} ± {s[f'{phase}_std_percent']:.2f}",
                          f"{ref[f'{phase}_mean_percent']:.1f} ± {ref[f'{phase}_std_percent']:.1f}"])
        table.append("| " + " | ".join(cells) + " |")
    (folder / "TABLE.md").write_text("\n".join(table) + "\n")
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "pdf.fonttype": 42})
    (ROOT / "figures").mkdir(exist_ok=True)
    for phase, heading in [("train", "Training samples"), ("test", "Held-out samples")]:
        fig, ax = plt.subplots(figsize=(7.2, 4.5))
        fig.subplots_adjust(left=.17, right=.96, bottom=.20, top=.78)
        for i, s in enumerate(stats):
            ref = paper[s["method"]]
            ax.errorbar(s[f"{phase}_mean_percent"], i-.13, xerr=s[f"{phase}_std_percent"],
                        fmt="o", ms=7, capsize=4, color="#285778", elinewidth=2,
                        label="Our implementation" if i == 0 else None)
            ax.errorbar(ref[f"{phase}_mean_percent"], i+.13, xerr=ref[f"{phase}_std_percent"],
                        fmt="s", ms=6, capsize=4, color="#A4A4A4", elinewidth=2,
                        label="Paper (2011)" if i == 0 else None)
        ax.set_yticks(range(4), METHODS)
        ax.set_ylim(3.55, -.55)
        ax.set_xlim(45, 80)
        ax.set_xticks(range(45, 81, 5))
        ax.set_xlabel("Clustering accuracy (%)")
        ax.grid(axis="x", color="#E4E7EB", linewidth=.7)
        ax.set_axisbelow(True)
        ax.legend(loc="upper center", bbox_to_anchor=(.5, 1.19), ncol=2, frameon=False, fontsize=10)
        fig.suptitle(f"MNIST · {heading}", fontsize=16, fontweight="bold", y=.97)
        fig.text(.5, .055, "Mean ± SD over 20 splits; independent partitions and replacement CV.",
                 ha="center", fontsize=9, color="#4B5563")
        for extension in ("png", "svg", "pdf"):
            fig.savefig(ROOT / "figures" / f"mnist_{phase}.{extension}", dpi=220)
        plt.close(fig)
    print("Regenerated MNIST comparison table and figures from 80 saved scores; no training.")


if __name__ == "__main__":
    main()
