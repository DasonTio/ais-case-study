"""Draw the two-layer architecture diagram (results/figures/fig00_architecture.png)."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from ais_patrec import viz  # noqa: E402
from ais_patrec.viz import plt  # noqa: E402


def box(ax, xy, w, h, title, body, color):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.03",
                                facecolor=color + "22", edgecolor=color, linewidth=1.6))
    ax.text(xy[0] + w / 2, xy[1] + h - 0.055, title, ha="center", va="top", fontsize=10.5,
            fontweight="bold", color=viz.TEXT)
    ax.text(xy[0] + w / 2, xy[1] + h - 0.135, body, ha="center", va="top", fontsize=8.6,
            color=viz.TEXT_2, linespacing=1.35)


def arrow(ax, a, b, text=None, offset=(0.0, 0.0)):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14, linewidth=1.4,
                                 color=viz.TEXT_2))
    if text:
        ax.text((a[0] + b[0]) / 2 + offset[0], (a[1] + b[1]) / 2 + offset[1], text, fontsize=8.5,
                color=viz.TEXT, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.2", facecolor=viz.SURFACE, edgecolor="none"))


def main():
    viz.apply_style()
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.set_xlim(0, 1.6); ax.set_ylim(0, 1); ax.axis("off")
    box(ax, (0.02, 0.62), 0.32, 0.3, "Antigen", "network connection record\n41 features\n-> shape-space [0,1]^12\n(log, one-hot, PCA)", viz.SERIES[6])
    box(ax, (0.46, 0.62), 0.42, 0.3, "Layer 1 - B-cell memory",
        "clonal selection (CLONALG)\nnearest memory cell + affinity\nwithin class threshold?", viz.SERIES[0])
    box(ax, (1.04, 0.62), 0.5, 0.3, "Known pattern",
        "normal | DoS | Probe | R2L | U2R\n(fast 'secondary response')", viz.SERIES[2])
    box(ax, (0.46, 0.12), 0.42, 0.3, "Layer 2 - T-cell tolerance",
        "negative selection (V-Detector)\ntrained on normal traffic only\ninside any detector?", viz.SERIES[1])
    box(ax, (1.04, 0.27), 0.5, 0.2, "NOVEL", "unfamiliar and non-self\n-> analyst triage", viz.NEUTRAL)
    box(ax, (1.04, 0.02), 0.5, 0.2, "Tolerated", "'tolerance' policy: normal\n'alarm_union' policy: memory's best guess", viz.SERIES[0])
    arrow(ax, (0.34, 0.77), (0.46, 0.77))
    arrow(ax, (0.88, 0.77), (1.04, 0.77), "recognised")
    arrow(ax, (0.67, 0.62), (0.67, 0.42), "not recognised", offset=(0.0, 0.0))
    arrow(ax, (0.88, 0.32), (1.04, 0.37), "yes")
    arrow(ax, (0.88, 0.20), (1.04, 0.12), "no")
    ax.add_patch(FancyArrowPatch((1.14, 0.47), (0.86, 0.63), connectionstyle="arc3,rad=0.3",
                                 arrowstyle="-|>", mutation_scale=14, linewidth=1.2,
                                 linestyle="--", color=viz.SERIES[2]))
    ax.text(1.20, 0.545, "learn_new_pattern(): an analyst labels a few\nNOVEL records; they become new memory cells",
            fontsize=8.3, color=viz.TEXT_2, ha="left", va="center")
    ax.set_title("Two-layer immune pattern recognizer", fontsize=13)
    out = ROOT / "results" / "figures" / "fig00_architecture.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out); print(out)


if __name__ == "__main__":
    main()
