"""Figures for the Google Docs report, in a minimal grey/blue style.

    python experiments/make_report_figures.py      # needs results/metrics.json

Writes PNGs to report/figures/.  The method being tested is always blue;
baselines are greys, so the eye goes to the AIS first.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch  # noqa: E402

from ais_patrec import ClonalSelectionClassifier, VDetector  # noqa: E402

OUT = ROOT / "report" / "figures"
BLUE = "#3c78d8"
BLUE_L = "#a4c2f4"
BLUE_BG = "#e8f0fe"
G1, G2, G3, G4 = "#9e9e97", "#c4c4bd", "#dcdcd6", "#ececE8"
INK, INK2, INK3 = "#202124", "#5f6368", "#9aa0a6"
RED = "#d9534f"
GREEN = "#34a853"
BOX_BG, BOX_EDGE = "#ffffff", "#cfcfc9"


def style():
    plt.rcParams.update({
        "font.family": "Inter", "font.size": 9.5, "text.color": INK,
        "axes.edgecolor": "#dadce0", "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#ececec", "grid.linewidth": 0.8,
        "axes.axisbelow": True, "xtick.major.size": 0, "ytick.major.size": 0,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "savefig.dpi": 200, "savefig.bbox": "tight", "legend.frameon": False,
    })


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name)
    plt.close(fig)
    print("[fig]", name)


def title(ax, text, x=0.0, y=1.06):
    ax.text(x, y, text, transform=ax.transAxes, fontsize=10, fontweight="bold", ha="left", va="bottom")


# ------------------------------------------------------------- diagram kit
def box(ax, x, y, w, h, head, body="", hl=False, head_size=9.5, body_size=8.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12",
                                facecolor=BLUE_BG if hl else BOX_BG,
                                edgecolor=BLUE if hl else BOX_EDGE, linewidth=1.6 if hl else 1.1))
    if body:
        n = body.count("\n") + 1
        block = 0.30 + 0.27 * n          # head line + body lines (data units)
        top = y + h / 2 + block / 2
        ax.text(x + w / 2, top - 0.15, head, ha="center", va="center", fontsize=head_size, fontweight="bold")
        ax.text(x + w / 2, top - 0.33, body, ha="center", va="top", fontsize=body_size, color=INK2,
                linespacing=1.25)
    else:
        ax.text(x + w / 2, y + h / 2, head, ha="center", va="center", fontsize=head_size, fontweight="bold")


def arrow(ax, a, b, color=INK3, rad=0.0, ls="-", lw=1.1):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=10, color=color, linewidth=lw,
                                 linestyle=ls, connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


def canvas(w, h, figsize):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off"); ax.set_aspect("equal")
    return fig, ax


# ------------------------------------------------------------------ figures
def fig_analogy():
    fig, ax = canvas(12, 6.4, (10, 5.3))
    ax.text(1.9, 6.05, "Immune system", ha="center", fontsize=10.5, fontweight="bold")
    ax.text(9.9, 6.05, "This AIS", ha="center", fontsize=10.5, fontweight="bold")
    rows = [("Self", "the body's own cells", "Normal traffic", "67,343 normal connections"),
            ("T-cell + thymus", "recruits that react to self\nare deleted", "V-Detector",
             "detectors may not touch\nnormal traffic"),
            ("B-cell + clonal selection", "best binders multiply\nand mutate", "Memory cells",
             "863 prototypes matured\nwith clones"),
            ("Secondary response", "second infection is\nbeaten faster", "learn_new_pattern()",
             "a few labelled examples\nbecome new memory")]
    for i, (a, ab, b, bb) in enumerate(rows):
        y = 4.6 - i * 1.45
        box(ax, 0.1, y, 3.6, 1.15, a, ab)
        box(ax, 8.1, y, 3.6, 1.15, b, bb, hl=(i in (1, 2)))
        arrow(ax, (3.8, y + 0.575), (8.0, y + 0.575))
        ax.text(5.9, y + 0.72, ["is", "becomes", "becomes", "becomes"][i], ha="center", fontsize=8.2, color=INK2)
    save(fig, "r01_immune_analogy.png")


def fig_dataset(m):
    fam = ["normal", "DoS", "Probe", "R2L", "U2R"]
    tr, te = m["dataset"]["family_counts_train"], m["dataset"]["family_counts_test"]
    ptr = np.array([tr[f] for f in fam]) / sum(tr.values()) * 100
    pte = np.array([te[f] for f in fam]) / sum(te.values()) * 100
    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    x = np.arange(len(fam)); w = 0.36
    b1 = ax.bar(x - w / 2 - 0.01, ptr, w, color=G2, label="KDDTrain+ (training)")
    b2 = ax.bar(x + w / 2 + 0.01, pte, w, color=BLUE, label="KDDTest+ (test)")
    for bars in (b1, b2):
        for r in bars:
            v = r.get_height()
            ax.text(r.get_x() + r.get_width() / 2, v + 1, f"{v:.1f}%" if v >= 0.1 else f"{v:.2f}%",
                    ha="center", fontsize=8.5, fontweight="bold" if bars is b2 else "normal")
    ax.set_xticks(x, fam, fontweight="bold"); ax.set_ylim(0, 62)
    ax.set_yticks([0, 20, 40, 60], ["0%", "20%", "40%", "60%"])
    ax.annotate("R2L: 0.8% of training,\n12.8% of test", xy=(3.2, 14.5), xytext=(3.55, 33),
                fontsize=8.5, color=INK, ha="left", arrowprops=dict(arrowstyle="-", color=INK3, lw=0.9))
    ax.legend(loc="upper right", ncol=2, fontsize=8.5, handlelength=0.9, handleheight=0.9, bbox_to_anchor=(1, 1.12))
    title(ax, "Share of each family (%)", y=1.07)
    save(fig, "r02_dataset_shift.png")


def fig_decision_flow():
    fig, ax = canvas(14.2, 5.0, (11, 3.9))
    ax.text(0.05, 4.75, "(a) From one connection to one decision", fontsize=9.5, ha="left")
    box(ax, 0.05, 2.35, 2.35, 1.5, "Connection", "41 features\n(bytes, flags, rates…)", hl=True)
    box(ax, 2.9, 2.35, 2.5, 1.5, "Shape-space", "log + one-hot\nPCA-12, scaled [0,1]")
    box(ax, 5.9, 2.35, 2.6, 1.5, "Memory (B-cells)", "nearest memory cell\ndistance d, label c")
    box(ax, 9.3, 3.35, 2.4, 1.25, "Recognised", "label c returned")
    box(ax, 9.3, 0.9, 2.4, 1.35, "V-Detector (T-cells)", "inside any\ndetector?")
    box(ax, 12.3, 1.55, 1.85, 0.95, "NOVEL", "", hl=True)
    box(ax, 12.3, 0.25, 1.85, 0.95, "normal", "")
    for a, b in [((2.4, 3.1), (2.9, 3.1)), ((5.4, 3.1), (5.9, 3.1)), ((8.5, 3.35), (9.3, 3.95)),
                 ((8.5, 2.85), (9.3, 1.65)), ((11.7, 1.85), (12.3, 2.0)), ((11.7, 1.3), (12.3, 0.75))]:
        arrow(ax, a, b)
    ax.text(9.05, 2.45, "d > τ_c", fontsize=8, color=INK2, ha="left")
    ax.text(8.85, 4.12, "d ≤ τ_c", fontsize=8, color=INK2, ha="center")
    ax.text(11.95, 2.15, "yes", fontsize=8, color=INK2, ha="center")
    ax.text(11.95, 0.75, "no", fontsize=8, color=INK2, ha="center")
    ax.text(7.2, 0.35, "τ_c = recognition radius of class c (0.99 quantile)", fontsize=7.8, color=INK3, ha="center")
    save(fig, "r03_decision_flow.png")


def equation(tex, name, size=17):
    fig = plt.figure(figsize=(8, 0.7))
    fig.text(0.5, 0.5, tex, ha="center", va="center", fontsize=size,
             math_fontfamily="cm", color=INK)
    save(fig, name)


def ring_data(seed=3):
    rng = np.random.default_rng(seed)
    t = rng.uniform(0, 2 * np.pi, 600)
    r = 0.28 + rng.normal(0, 0.025, 600)
    S = np.c_[0.5 + r * np.cos(t), 0.5 + r * np.sin(t)]
    ta = rng.uniform(0, 2 * np.pi, 60)
    ra = np.where(rng.random(60) < 0.5, 0.28 + rng.uniform(0.07, 0.12, 60), 0.28 - rng.uniform(0.07, 0.12, 60))
    A = np.vstack([np.c_[0.5 + ra * np.cos(ta), 0.5 + ra * np.sin(ta)], rng.uniform(0, 1, (20, 2))])
    A = A[np.min(np.linalg.norm(A[:, None] - S[None], axis=2), 1) > 0.05]
    return S, A


def fig_negative_selection_2d():
    S, A = ring_data()
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6))
    for ax, c0 in zip(axes, [0.90, 0.99]):
        vd = VDetector(self_radius=0.02, target_coverage=c0, random_state=1).fit(S)
        for c, r in zip(vd.centers_, vd.radii_):
            ax.add_patch(Circle(c, r, facecolor=BLUE_L, alpha=0.22, edgecolor=BLUE, linewidth=0.4))
        ax.scatter(*S.T, s=5, color=G1, zorder=3, label="normal traffic (self)")
        hit = vd.predict(A) == 1
        ax.scatter(*A[hit].T, s=28, marker="x", color=BLUE, linewidth=1.6, zorder=4, label="attack, caught")
        ax.scatter(*A[~hit].T, s=55, facecolor="white", edgecolor=RED, linewidth=1.6, zorder=4,
                   label="attack, missed (hole)")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.grid(False)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True); s.set_color("#dadce0")
        title(ax, f"coverage target {c0}", y=1.035)
        ax.text(1.0, 1.035, f"{vd.n_detectors} detectors · {hit.mean():.0%} caught", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8.5, color=INK2)
    axes[0].legend(loc="lower left", fontsize=7.8, bbox_to_anchor=(0, -0.2), ncol=3)
    save(fig, "r04_negative_selection_2d.png")


def fig_clonal_step():
    rng = np.random.default_rng(7)
    cells = np.array([[0.30, 0.62], [0.44, 0.34], [0.72, 0.70], [0.18, 0.25], [0.80, 0.28]])
    x = np.array([0.52, 0.52])
    d = np.linalg.norm(cells - x, axis=1)
    order = np.argsort(d)[:3]
    alpha, k = 0.3, 2
    best = order[0]
    sig = alpha * d[best] / np.sqrt(k)
    # draw clones for each selected cell
    fig = plt.figure(figsize=(10.4, 4.6))
    ax = fig.add_axes([0.02, 0.06, 0.56, 0.88])
    ax.set_xlim(0.05, 0.95); ax.set_ylim(0.1, 0.9); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    ax.set_facecolor("#f4f4f2")
    for s in ax.spines.values():
        s.set_visible(True); s.set_color("#dadce0")
    ax.text(0.07, 0.87, "shape-space (2 of 12 dimensions)", fontsize=8, color=INK2, va="top")
    pool, owner = [], []
    for rank, c in enumerate(order):
        n = int(round(10 / (rank + 1)))
        cl = cells[c] + rng.normal(0, alpha * d[c] / np.sqrt(k), (n, 2))
        pool.append(cl); owner += [c] * n
    pool = np.vstack(pool)
    dc = np.linalg.norm(pool - x, axis=1); b = int(np.argmin(dc))
    ax.scatter(*pool.T, s=14, color=G2, zorder=2)
    others = [i for i in range(len(cells)) if i not in order]
    ax.scatter(*cells[others].T, s=40, color=G3, edgecolor=G1, zorder=3)
    for rank, c in enumerate(order):
        ax.plot([cells[c][0], x[0]], [cells[c][1], x[1]], ls="--", color=INK3, lw=0.8, zorder=1)
        off = [-0.085, 0.0] if rank == 0 else [0.0, 0.045]
        ax.text(*(cells[c] + off), f"cell {rank + 1}\nd = {d[c]:.2f}", fontsize=7.8,
                ha="right" if rank == 0 else "center", va="center" if rank == 0 else "bottom", color=INK2)
    ax.scatter(*cells[order[1:]].T, s=55, color=G1, zorder=3)
    ax.scatter(*cells[best], s=95, facecolor="white", edgecolor=RED, linewidth=1.8, zorder=4)
    ax.scatter(*pool[b], s=85, color=GREEN, zorder=5)
    ax.annotate("", xy=pool[b], xytext=cells[best],
                arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.8), zorder=4)
    ax.scatter(*x, marker="*", s=230, color=INK, zorder=6)
    ax.text(x[0] + 0.03, x[1] - 0.045, "antigen x\n(one DoS record)", fontsize=8.2)
    ax.text(0.93, 0.13, "grey dots: clones", fontsize=7.8, color=INK2, ha="right")

    rt = fig.add_axes([0.6, 0.06, 0.39, 0.88]); rt.axis("off")
    rt.text(0, 0.97, "One clonal-selection step", fontsize=10, fontweight="bold", va="top")
    steps = [("select", f"3 nearest cells of class DoS\n(d = {d[order[0]]:.2f}, {d[order[1]]:.2f}, {d[order[2]]:.2f})"),
             ("expand", "clones by rank: 10, 5, 3"),
             ("mutate", f"σ = α·d/√k, so a far cell mutates more\n(cell 1: σ = 0.3·{d[best]:.2f}/√2 = {sig:.3f})"),
             ("compare", f"best clone d = {dc[b]:.3f}  vs  cell 1 d = {d[best]:.3f}"),
             ("keep", "closer, so it replaces cell 1\n(affinity maturation)")]
    for i, (h, t) in enumerate(steps):
        y = 0.86 - i * 0.175
        rt.add_patch(Circle((0.035, y - 0.012), 0.03, transform=rt.transAxes, color=BLUE_BG, ec="none"))
        rt.text(0.035, y - 0.012, str(i + 1), ha="center", va="center", fontsize=8.5, color=BLUE, fontweight="bold")
        rt.text(0.1, y, h, fontsize=8.8, fontweight="bold", va="center")
        rt.text(0.1, y - 0.035, t, fontsize=8.2, color=INK2, va="top", linespacing=1.25)
    rt.text(0.1, 0.0, "red ring: old cell · green: matured cell", fontsize=7.8, color=INK2)
    save(fig, "r05_clonal_step.png")


def fig_clonal_2d():
    rng = np.random.default_rng(3)
    centers = np.array([[0.25, 0.3], [0.7, 0.3], [0.5, 0.75]])
    X = np.vstack([rng.normal(c, [0.09, 0.06], (300, 2)) for c in centers])
    y = np.repeat(np.array(["A", "B", "C"], dtype=object), 300)
    cols = {"A": BLUE, "B": G1, "C": "#e69138"}
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6))
    for ax, gens, label in [(axes[0], 0, "generation 0: random recruits"), (axes[1], 10, "generation 10: matured")]:
        clf = ClonalSelectionClassifier(n_memory_per_class=6, n_generations=gens, random_state=2).fit(X, y)
        gx, gy = np.meshgrid(np.linspace(0, 1, 220), np.linspace(0, 1, 220))
        _, _, rec = clf.predict_with_affinity(np.c_[gx.ravel(), gy.ravel()])
        ax.contourf(gx, gy, (~rec).reshape(gx.shape).astype(float), levels=[0.5, 1.5], colors=["#f1f3f4"])
        for c in "ABC":
            ax.scatter(*X[y == c].T, s=4, color=cols[c], alpha=0.35)
            mcell = clf.memory_[clf.memory_labels_ == c]
            ax.scatter(*mcell.T, s=110, marker="*", color=cols[c], edgecolor=INK, linewidth=0.7, zorder=4)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.grid(False)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True); s.set_color("#dadce0")
        title(ax, label, y=1.035)
    axes[1].text(0.98, 0.02, "grey area = not recognised", transform=axes[1].transAxes, ha="right",
                 fontsize=8, color=INK2)
    axes[0].text(0.02, 0.02, "★ = memory cell, dots = training records", transform=axes[0].transAxes,
                 fontsize=8, color=INK2)
    save(fig, "r06_clonal_2d.png")


def fig_architecture():
    fig, ax = canvas(14.4, 5.4, (11, 4.3))
    ax.set_ylim(-0.45, 5.4)
    xs = [0.1, 2.95, 5.8, 8.65, 11.5]
    items = [("Record", "network connection\nin shape-space"),
             ("Memory", "863 cells, nearest\ncell + distance"),
             ("Recognised?", "d ≤ τ_c of\nthat class"),
             ("V-Detector", "275 detectors,\nnormal-only training"),
             ("NOVEL", "unfamiliar and\nnon-self")]
    W = 2.35
    xs = [0.1 + i * 2.95 for i in range(5)]
    for i, (x, (h, b)) in enumerate(zip(xs, items)):
        box(ax, x, 3.3, W, 1.45, h, b, hl=(i == 4))
        if i:
            arrow(ax, (xs[i - 1] + W, 4.02), (x, 4.02))
    ax.text(xs[2] + W + 0.3, 4.2, "no", fontsize=8, color=INK2, ha="center")
    ax.text(xs[3] + W + 0.3, 4.2, "in", fontsize=8, color=INK2, ha="center")
    # yes branch from Recognised
    box(ax, xs[2], 0.4, W, 1.3, "Known label", "normal · DoS · Probe\nR2L · U2R")
    arrow(ax, (xs[2] + W / 2, 3.3), (xs[2] + W / 2, 1.7)); ax.text(xs[2] + W / 2 + 0.15, 2.45, "yes", fontsize=8, color=INK2)
    box(ax, xs[3], 0.4, W, 1.3, "normal", "outside every\ndetector")
    arrow(ax, (xs[3] + W / 2, 3.3), (xs[3] + W / 2, 1.7)); ax.text(xs[3] + W / 2 + 0.15, 2.45, "outside", fontsize=8, color=INK2)
    box(ax, xs[4], 0.4, W, 1.3, "Analyst labels", "5–80 examples →\nlearn_new_pattern()", hl=True)
    arrow(ax, (xs[4] + W / 2, 3.3), (xs[4] + W / 2, 1.7))
    cx_a, cx_m = xs[4] + W / 2, xs[1] + W / 2
    ax.plot([cx_a, cx_a, cx_m, cx_m], [0.4, -0.25, -0.25, 2.2], color=BLUE, lw=1.2, ls="--")
    arrow(ax, (cx_m, 2.2), (cx_m, 3.3), color=BLUE)
    ax.text(cx_m - 0.15, 1.2, "new memory cells,\nold memory untouched", fontsize=8, color=BLUE, ha="right")
    save(fig, "r07_architecture.png")


def fig_experiment_flow():
    fig, ax = canvas(14.4, 5.0, (11, 3.8))
    steps = [("1. Split", "KDDTrain+ 80/20\nfit / validation"),
             ("2. Tune", "grid on validation\nonly"),
             ("3. Final runs", "KDDTest+\n5 seeds"),
             ("4. Hold-out", "remove one family\nfrom training"),
             ("5. Secondary", "5, 20, 80 labelled\nexamples")]
    xs = [0.1, 2.95, 5.8, 8.65, 11.5]
    for i, (x, (h, b)) in enumerate(zip(xs, steps)):
        box(ax, x, 2.6, 2.6, 1.4, h, b, hl=(i == 1))
        if i:
            arrow(ax, (xs[i - 1] + 2.6, 3.3), (x, 3.3))
    ax.plot([5.8, 14.1], [4.45, 4.45], color=BLUE, lw=1.2)
    for xx in (5.8, 14.1):
        ax.plot([xx, xx], [4.3, 4.6], color=BLUE, lw=1.2)
    ax.text(9.95, 4.62, "settings frozen from here", fontsize=8.5, ha="center")
    subs = [("E1 Anomaly", "normal-only models"), ("E2 Recognition", "5 families"), ("E3 Two-layer", "full system")]
    for j, (h, b) in enumerate(subs):
        x = 3.4 + j * 3.0
        box(ax, x, 0.25, 2.7, 1.2, h, b)
        arrow(ax, (7.1, 2.6), (x + 1.35, 1.45))
    save(fig, "r08_experiment_flow.png")


def fig_dimensionality(m):
    rows = m["E1b_dimensionality"]["rows"]
    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    for smp, col, lab in [("uniform", G1, "classic V-Detector (uniform)"),
                          ("self_proximal", BLUE, "self-proximal V-Detector")]:
        r = [q for q in rows if q["sampler"] == smp]
        dims = [q["dims"] for q in r]; dr = [100 * q["DR"] for q in r]
        ax.plot(dims, dr, marker="o", ms=5, color=col, lw=2, label=lab)
    sp = {q["dims"]: q for q in rows if q["sampler"] == "self_proximal"}
    un = {q["dims"]: q for q in rows if q["sampler"] == "uniform"}
    ax.axvline(12, color=INK3, ls="--", lw=0.9)
    ax.text(12.3, 50, f"12 dims (used):\n{100*sp[12]['DR']:.0f}% vs {100*un[12]['DR']:.0f}%", fontsize=8.5,
            fontweight="bold")
    ax.set_ylim(0, 105); ax.set_yticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100"])
    ax.set_xticks([2, 4, 8, 12, 16, 24]); ax.set_xlabel("shape-space dimensions (PCA components)")
    ax.legend(loc="lower left", fontsize=8.5)
    title(ax, "Detection rate on validation (%)", y=1.03)
    save(fig, "r09_dimensionality.png")


def fig_anomaly(m):
    e1 = m["E1_anomaly"]["models"]
    order = ["V-Detector (self-proximal)", "V-Detector (uniform)", "Nearest-self (positive selection)",
             "Isolation Forest", "One-class SVM", "LOF (novelty)"]
    short = ["V-Detector\n(self-prox.)", "V-Detector\n(uniform)", "Nearest\nself", "Isolation\nForest",
             "One-class\nSVM", "LOF"]
    val = lambda v: v["mean"] if isinstance(v, dict) else v  # noqa: E731
    dr = [100 * val(e1[k]["native"]["DR"]) for k in order]
    far = [100 * val(e1[k]["native"]["FAR"]) for k in order]
    cols = [BLUE, BLUE_L, G2, G2, G2, G2]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.8), gridspec_kw={"wspace": 0.22})
    for ax, vals, t, top in [(axes[0], dr, "Detection rate (%) · higher is better", 100),
                             (axes[1], far, "False-alarm rate (%) · lower is better", 14)]:
        bars = ax.bar(range(6), vals, 0.66, color=cols)
        for i, b in enumerate(bars):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + top * 0.015, f"{vals[i]:.1f}",
                    ha="center", fontsize=8.5, fontweight="bold" if i == 0 else "normal")
        ax.set_xticks(range(6), short, fontsize=7.4); ax.set_ylim(0, top)
        title(ax, t, y=1.03)
    save(fig, "r10_anomaly_bars.png")


def fig_recall(m):
    e2 = m["E2_recognition"]["models"]
    names = ["CLONALG memory (AIS)", "Random prototypes (ablation)", "Random Forest (balanced)",
             "Random Forest", "1-NN (all training data)"]
    labels = ["CLONALG memory (AIS)", "random prototypes", "Random Forest (balanced)", "Random Forest", "1-NN"]
    cols = [BLUE, G1, G2, G3, G4]
    fam = ["normal", "DoS", "Probe", "R2L", "U2R"]
    val = lambda v: v["mean"] if isinstance(v, dict) else v  # noqa: E731
    fig, ax = plt.subplots(figsize=(10.4, 4.1))
    w = 0.16
    for i, n in enumerate(names):
        v = [100 * val(e2[n]["recall"][f]) for f in fam]
        bars = ax.bar(np.arange(5) + (i - 2) * w, v, w * 0.94, color=cols[i], label=labels[i],
                      edgecolor="#d5d5cf" if i == 4 else "none", linewidth=0.6)
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.2, f"{b.get_height():.0f}", ha="center",
                    fontsize=7, fontweight="bold" if i == 0 else "normal", color=INK if i == 0 else INK2)
    ax.set_xticks(range(5), fam, fontweight="bold"); ax.set_ylim(0, 108)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.legend(ncol=5, fontsize=8, loc="upper left", bbox_to_anchor=(0, 1.16), handlelength=0.9, handleheight=0.9)
    title(ax, "Recall per family on KDDTest+ (%)", y=1.17)
    save(fig, "r11_recall_family.png")


def fig_ablation(m):
    e2 = m["E2_recognition"]["models"]
    names = ["CLONALG memory (AIS)", "Random prototypes (ablation)", "Random Forest (balanced)",
             "1-NN (all training data)", "Random Forest"]
    short = ["CLONALG\nmemory", "random\nprototypes", "RF\n(balanced)", "1-NN", "Random\nForest"]
    val = lambda v: v["mean"] if isinstance(v, dict) else v  # noqa: E731
    f1 = [val(e2[n]["macro_F1"]) for n in names]
    size = [val(e2[n]["model_size"]) for n in names]
    fig, ax = plt.subplots(figsize=(8.4, 3.7))
    bars = ax.bar(range(5), f1, 0.62, color=[BLUE, G1, G2, G2, G2])
    for i, b in enumerate(bars):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.012, f"{f1[i]:.3f}", ha="center",
                fontsize=9, fontweight="bold" if i == 0 else "normal")
        ax.text(b.get_x() + b.get_width() / 2, 0.03, f"{int(size[i]):,}\nstored", ha="center", fontsize=7.3,
                color="white" if i < 2 else INK2)
    yb = f1[0] + 0.06
    ax.plot([0, 0, 1, 1], [yb - 0.015, yb, yb, yb - 0.015], color=INK3, lw=0.9)
    ax.text(0.5, yb + 0.012, f"affinity maturation adds +{f1[0]-f1[1]:.3f}", ha="center", fontsize=8.3)
    ax.set_xticks(range(5), short, fontsize=8.5); ax.set_ylim(0, 0.78)
    title(ax, "Macro-F1 on KDDTest+ (numbers in bars: cells, samples or tree nodes kept)", y=1.03)
    save(fig, "r12_ablation_f1.png")


def fig_outcomes(m):
    tab = m["E3_two_layer"]["outcome_table_selected"]
    rows = list(tab)
    labels = [f"{r} (known types)" if r != "novel types" else "17 test-only types" for r in rows]
    cats = ["correct family", "other attack family", "flagged NOVEL", "missed (called normal)"]
    cols = [BLUE, BLUE_L, G2, "#f4c7c3"]
    data = np.array([[tab[r][c] for c in cats] for r in rows]) * 100
    fig, ax = plt.subplots(figsize=(10.2, 3.8))
    left = np.zeros(len(rows))
    for j, c in enumerate(cats):
        ax.barh(labels, data[:, j], left=left, color=cols[j], label=c, height=0.62, edgecolor="white", linewidth=1.5)
        for i, v in enumerate(data[:, j]):
            if v >= 7:
                ax.text(left[i] + v / 2, i, f"{v:.0f}%", ha="center", va="center", fontsize=8.5,
                        color="white" if j == 0 else INK, fontweight="bold" if j == 0 else "normal")
        left += data[:, j]
    ax.invert_yaxis(); ax.set_xlim(0, 100); ax.grid(False); ax.set_xticks([])
    ax.tick_params(axis="y", labelsize=8.8)
    ax.legend(ncol=4, fontsize=8, loc="upper left", bbox_to_anchor=(0, 1.13), handlelength=0.9, handleheight=0.9)
    save(fig, "r13_two_layer_outcomes.png")


def fig_zero_day(m):
    e4 = m["E4_zero_day"]
    fams = list(e4)
    series = [("immune_detected", "AIS: detected", BLUE), ("immune_flagged_novel", "AIS: labelled NOVEL", BLUE_L),
              ("rf_detected", "Random Forest: detected", G2)]
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    w = 0.26
    for i, (k, lab, col) in enumerate(series):
        v = [100 * e4[f][k]["mean"] for f in fams]
        bars = ax.bar(np.arange(len(fams)) + (i - 1) * w, v, w * 0.94, color=col, label=lab)
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.3, f"{b.get_height():.1f}", ha="center",
                    fontsize=8.3, fontweight="bold" if i == 0 else "normal")
    ax.set_xticks(range(len(fams)), [f"{f} removed" for f in fams], fontweight="bold")
    ax.set_ylim(0, 90)
    ax.legend(ncol=3, fontsize=8.3, loc="upper left", bbox_to_anchor=(0, 1.15), handlelength=0.9, handleheight=0.9)
    title(ax, "Test traffic of the removed family that was flagged (%)", y=1.16)
    save(fig, "r14_zero_day.png")


def fig_secondary(m):
    e4 = m["E4_zero_day"]
    fams = list(e4); shots = [5, 20, 80]
    fig, axes = plt.subplots(1, len(fams), figsize=(10.6, 3.5), sharey=True, gridspec_kw={"wspace": 0.12})
    for ax, f in zip(axes, fams):
        for key, col, lab in [("immune_recall", BLUE, "AIS (add memory, 2–30 ms)"),
                              ("rf_recall", G1, "Random Forest (retrain, 10–15 s)")]:
            v = [100 * e4[f]["shots"][str(k)][key]["mean"] for k in shots]
            ax.plot(range(3), v, marker="o", ms=5, color=col, lw=2, label=lab)
            for i, vv in enumerate(v):
                up = key[0] == "i"
                ax.text(i, vv + 4 if up else vv - 8, f"{vv:.0f}", ha="center", fontsize=8,
                        color=INK if up else INK2, fontweight="bold" if up else "normal")
        ax.set_xticks(range(3), [str(s) for s in shots]); ax.set_ylim(-10, 85); ax.set_xlim(-0.3, 2.3); ax.set_yticks([0, 20, 40, 60, 80])
        title(ax, f"{f}", y=1.03)
        ax.set_xlabel("labelled examples")
    axes[0].legend(fontsize=8, loc="upper left", bbox_to_anchor=(0, 1.3), ncol=2)
    axes[0].set_ylabel("recall of the new family (%)")
    save(fig, "r15_secondary_response.png")


def main():
    style()
    m = json.load(open(ROOT / "results" / "metrics.json"))
    fig_analogy(); fig_dataset(m); fig_decision_flow()
    equation(r"$r_j \;=\; \min_{s\,\in\,\mathrm{Self}} \Vert c_j - s \Vert \;-\; r_s$", "eq1_detector_radius.png")
    equation(r"$x\ \mathrm{is\ nonself} \Leftrightarrow \exists\, j:\ \Vert x - c_j \Vert < r_j$", "eq2_nonself.png")
    equation(r"$n_i = \mathrm{round}\!\left(\frac{\beta}{i}\right),\qquad \sigma_i = \alpha \cdot \frac{d_i}{\sqrt{k}}$",
             "eq3_clone_mutation.png")
    equation(r"$\tau_c = \max\!(Q_{0.99}(d_{\mathrm{memory}}),\ Q_{0.99}(d_{\mathrm{neighbour}}))$",
             "eq4_threshold.png")
    equation(r"$m_c = 150 \cdot \sqrt{n_c \,/\, \mathrm{median}(n)}$", "eq5_memory_budget.png")
    fig_negative_selection_2d(); fig_clonal_step(); fig_clonal_2d(); fig_architecture(); fig_experiment_flow()
    fig_dimensionality(m); fig_anomaly(m); fig_recall(m); fig_ablation(m); fig_outcomes(m); fig_zero_day(m)
    fig_secondary(m)


if __name__ == "__main__":
    main()
