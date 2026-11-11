"""Shared matplotlib styling (palette validated for colour-vision deficiency)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
# Categorical slots in fixed order (never cycled).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
          "#008300", "#6250d6", "#e34948"]
NEUTRAL = "#9b9a94"
SELF_COLOR = "#2a78d6"
NONSELF_COLOR = "#e34948"

FAMILY_COLOR = {"normal": SERIES[0], "DoS": SERIES[1], "Probe": SERIES[2],
                "R2L": SERIES[3], "U2R": SERIES[4], "novel": NEUTRAL}


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT_2, "xtick.color": TEXT_2, "ytick.color": TEXT_2,
        "text.color": TEXT, "axes.titlecolor": TEXT, "axes.titleweight": "bold",
        "axes.titlesize": 12, "axes.labelsize": 10, "font.size": 10,
        "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "lines.linewidth": 2.0,
        "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
    })
