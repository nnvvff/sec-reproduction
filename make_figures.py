"""Regenerate the presentation table and figures from saved scores; no training."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
METHODS = ["KMeans", "SC+ridge", "SEC/SC", "SEC/LR"]
COLORS = ["#88929B", "#66798B", "#4B83A5", "#285778"]


def main():
    with (ROOT / "results/optdigits/runs.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    stats = []
    for method in METHODS:
        group = sorted((r for r in rows if r["method"] == method), key=lambda r: int(r["split"]))
        if len(group) != 20 or [int(r["split"]) for r in group] != list(range(20)):
            raise ValueError(f"Expected every split exactly once for {method}")
        if any(int(r["seed"]) != 20261029 + int(r["split"]) for r in group):
            raise ValueError("Unexpected seed")
        stat = dict(method=method, n=20)
        for phase in ("train", "test"):
            scores = np.array([float(r[f"{phase}_acc"]) for r in group]) * 100
            if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 100)):
                raise ValueError("Invalid accuracy")
            stat[f"{phase}_mean_percent"] = scores.mean()
            stat[f"{phase}_std_percent"] = scores.std(ddof=1)
        stats.append(stat)
    with (ROOT / "results/optdigits/summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(stats[0]))
        writer.writeheader()
        writer.writerows(stats)
    table = ["| Method | Train ACC (%) | Test ACC (%) |", "|---|---:|---:|"]
    for s in stats:
        table.append(f"| {s['method']} | {s['train_mean_percent']:.2f} ± {s['train_std_percent']:.2f} | "
                     f"{s['test_mean_percent']:.2f} ± {s['test_std_percent']:.2f} |")
    (ROOT / "results/optdigits/TABLE.md").write_text("\n".join(table) + "\n")
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "pdf.fonttype": 42})
    (ROOT / "figures").mkdir(exist_ok=True)
    for phase, heading in [("train", "Training samples"), ("test", "Held-out samples")]:
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        fig.subplots_adjust(left=.17, right=.96, bottom=.22, top=.80)
        for i, (s, color) in enumerate(zip(stats, COLORS)):
            mean, std = s[f"{phase}_mean_percent"], s[f"{phase}_std_percent"]
            ax.errorbar(mean, i, xerr=std, fmt="o", ms=8, capsize=5,
                        color=color, elinewidth=2)
            ax.annotate(f"{mean:.2f} ± {std:.2f}", (mean, i), xytext=(0, 12),
                        textcoords="offset points", ha="center", fontsize=10)
        ax.set_yticks(range(4), METHODS)
        ax.set_ylim(3.6, -.65)
        ax.set_xlim(65, 100)
        ax.set_xticks(range(65, 101, 5))
        ax.set_xlabel("Clustering accuracy (%)")
        ax.grid(axis="x", color="#E4E7EB", linewidth=.7)
        ax.set_axisbelow(True)
        fig.suptitle(f"Optdigits · {heading}", fontsize=16, fontweight="bold", y=.96)
        fig.text(.5, .86, "Same 20 outer splits · SEC parameters selected by inner CV",
                 ha="center", fontsize=10, color="#4B5563")
        fig.text(.5, .045, "Points: mean ACC · Error bars: sample SD (n = 20), not confidence intervals",
                 ha="center", fontsize=9, color="#4B5563")
        for extension in ("png", "svg", "pdf"):
            fig.savefig(ROOT / "figures" / f"optdigits_{phase}.{extension}", dpi=220)
        plt.close(fig)
    print("Regenerated table and two figures from 80 saved scores; no model fitting.")


if __name__ == "__main__":
    main()
