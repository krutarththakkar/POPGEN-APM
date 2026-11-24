"""Plot tipping point vs q from a sweep CSV: one panel per N, one line per h.

    python plot.py results/tipping.csv
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


def plot_tipping(csv_path):
    csv_path = Path(csv_path)
    data = defaultdict(lambda: defaultdict(list))  # N -> h -> [(q, tip, q_crit)]
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            data[int(r["N"])][float(r["h"])].append((float(r["q"]), float(r["tipping_white_frac"]), float(r["q_critical"])))

    Ns = sorted(data)
    keys = sorted({h for N in Ns for h in data[N]})
    if len(keys) > len(SERIES):
        raise SystemExit(f"{len(keys)} h values is more than the {len(SERIES)} colors available; plot fewer per figure.")
    color = dict(zip(keys, SERIES))

    plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10})
    fig, axes = plt.subplots(1, len(Ns), figsize=(4.4 * len(Ns) + 0.6, 5.2), sharey=True, squeeze=False, facecolor=SURFACE)
    gaps = False
    for ax, N in zip(axes[0], Ns):
        ax.set_facecolor(SURFACE)
        for h in sorted(data[N]):
            pts = sorted(data[N][h])
            qs, tips = [p[0] for p in pts], [p[1] for p in pts]
            ax.plot(qs, tips, color=color[h], lw=2, solid_capstyle="round", label=f"h = {h:g}")
            qc = pts[0][2]
            if qs[0] <= qc <= qs[-1]:
                ax.plot(qc, 1, marker="v", ms=8, color=color[h], mec=SURFACE, mew=1, clip_on=False, zorder=5)
            gaps = gaps or any(t != t for t in tips)
        ax.set_title(f"N = {N:,}", color=INK, fontsize=11, loc="left", pad=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.03)
        ax.set_xlabel("q  (handsome guy probability per asexual generation)", color=INK_2)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(AXIS)
    axes[0][0].set_ylabel("tipping point  (starting white fraction x)", color=INK_2)

    fig.suptitle("Tipping point: starting white fraction x at which white and blue are equally likely to take over",
                 color=INK, fontsize=13, x=0.02, ha="left", y=0.98)
    note = ("Start above a line and white usually wins; below it, blue.   "
            "▼ = predicted q above which rare blue declines:  ln 2 / ln(1 + h − 1/N)")
    if gaps:
        note += "\nGaps in a line: blue and white coexist for the whole run, so there is no tipping point to report."
    fig.text(0.02, 0.915, note, color=INK_2, fontsize=9.5, va="top", linespacing=1.5)
    handles, labels = axes[0][0].get_legend_handles_labels()
    top = 0.80 if gaps else 0.84
    fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.012, top), ncol=len(labels), frameon=False,
               labelcolor=INK_2, fontsize=9.5, handlelength=1.4, columnspacing=1.6)
    fig.tight_layout(rect=(0, 0, 0.99, top + 0.04), w_pad=3)
    png_path = csv_path.with_suffix(".png")
    fig.savefig(png_path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return png_path


if __name__ == "__main__":
    print(plot_tipping(sys.argv[1] if len(sys.argv) > 1 else "results/tipping.csv"))
