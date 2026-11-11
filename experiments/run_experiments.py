"""Reproduce every number and figure in the report.

    python experiments/run_experiments.py            # full run (~10-15 min, 2 CPUs)
    python experiments/run_experiments.py --quick    # smoke test (~1-2 min)

Protocol
--------
* KDDTrain+ is split 80/20 (stratified) into FIT and VALIDATION.  All
  hyper-parameters and thresholds are chosen on VALIDATION; KDDTest+ is only
  used for the final numbers.
* KDDTest+ contains attack types never seen in KDDTrain+ ("novel types"),
  reported separately as a natural zero-day test.
* Stochastic models are run over several seeds; we report mean +/- std.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.svm import OneClassSVM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ais_patrec import (FAMILIES, NOVEL, ClonalSelectionClassifier,  # noqa: E402
                        ImmuneRecognizer, NearestSelfDetector, ShapeSpace,
                        VDetector, load_nsl_kdd)
from ais_patrec import viz  # noqa: E402
from ais_patrec.viz import plt  # noqa: E402

TARGET_FAR = 0.02  # alert budget used to set score thresholds on validation


# ---------------------------------------------------------------- helpers
def binary_metrics(y_attack: np.ndarray, pred: np.ndarray, novel: np.ndarray | None = None) -> dict:
    tp = np.sum((pred == 1) & (y_attack == 1)); fp = np.sum((pred == 1) & (y_attack == 0))
    fn = np.sum((pred == 0) & (y_attack == 1))
    out = {"DR": float(pred[y_attack == 1].mean()), "FAR": float(pred[y_attack == 0].mean()),
           "F1": float(2 * tp / max(2 * tp + fp + fn, 1))}
    if novel is not None:
        out["novel_DR"] = float(pred[novel].mean())
    return out


def thr_at_far(scores_normal: np.ndarray, far: float = TARGET_FAR) -> float:
    return float(np.quantile(scores_normal, 1 - far))


def per_class_recall(y: np.ndarray, p: np.ndarray) -> dict:
    return {c: float((p[y == c] == c).mean()) for c in FAMILIES if np.any(y == c)}


def agg(runs: list[dict]) -> dict:
    """mean/std over seeds for every numeric (possibly nested) field."""
    out = {}
    for k, v in runs[0].items():
        if isinstance(v, dict):
            out[k] = agg([r[k] for r in runs])
        elif isinstance(v, (int, float, np.floating)):
            vals = np.array([r[k] for r in runs], float)
            out[k] = {"mean": float(vals.mean()), "std": float(vals.std())}
        else:
            out[k] = v
    return out


def savefig(fig, path: Path) -> None:
    fig.savefig(path)
    plt.close(fig)
    print(f"[fig] {path.name}")


# ------------------------------------------------------- E0: 2-D concepts
def e0_concepts(figs: Path) -> None:
    rng = np.random.default_rng(3)
    # Self = a noisy ring ("normal behaviour"); anomalies near and far from it.
    t = rng.uniform(0, 2 * np.pi, 600)
    r = 0.28 + rng.normal(0, 0.025, 600)
    self_pts = np.c_[0.5 + r * np.cos(t), 0.5 + r * np.sin(t)]
    ta = rng.uniform(0, 2 * np.pi, 60)
    ra = np.where(rng.random(60) < 0.5, 0.28 + rng.uniform(0.07, 0.12, 60), 0.28 - rng.uniform(0.07, 0.12, 60))
    anom = np.vstack([np.c_[0.5 + ra * np.cos(ta), 0.5 + ra * np.sin(ta)], rng.uniform(0, 1, (20, 2))])
    anom = anom[np.min(np.linalg.norm(anom[:, None] - self_pts[None], axis=2), 1) > 0.05]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.2))
    for ax, c0 in [(axes[0], 0.90), (axes[1], 0.99)]:
        vd = VDetector(self_radius=0.02, target_coverage=c0, candidate_sampler="uniform",
                       random_state=1).fit(self_pts)
        for c, rad in zip(vd.centers_, vd.radii_):
            ax.add_patch(plt.Circle(c, rad, facecolor=viz.NONSELF_COLOR, alpha=0.10,
                                    edgecolor=viz.NONSELF_COLOR, linewidth=0.6))
        ax.scatter(*self_pts.T, s=6, color=viz.SELF_COLOR, label="self (normal traffic)", zorder=3)
        caught = vd.predict(anom) == 1
        ax.scatter(*anom[caught].T, s=34, marker="x", color=viz.TEXT, label="anomaly - detected", zorder=4)
        ax.scatter(*anom[~caught].T, s=60, marker="o", facecolor="none", edgecolor=viz.TEXT,
                   linewidth=1.6, label="anomaly - missed (in a hole)", zorder=4)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.grid(False)
        ax.set_title(f"Target coverage c0 = {c0}: {vd.n_detectors} detectors, "
                     f"{caught.mean():.0%} of anomalies caught", fontsize=10.5)
    axes[0].legend(loc="lower left", fontsize=8.5)
    fig.suptitle("Negative selection in 2-D: detectors (red discs) grow until they touch self",
                 fontsize=12, fontweight="bold")
    savefig(fig, figs / "fig01_negative_selection_2d.png")

    # Clonal selection on three 2-D pattern classes.
    centers = np.array([[0.25, 0.3], [0.7, 0.3], [0.5, 0.75]])
    X = np.vstack([rng.normal(c, [0.09, 0.06], (300, 2)) for c in centers])
    y = np.repeat(np.array(["A", "B", "C"], dtype=object), 300)
    init = ClonalSelectionClassifier(n_memory_per_class=6, n_generations=0, random_state=2).fit(X, y)
    clf = ClonalSelectionClassifier(n_memory_per_class=6, n_generations=10, random_state=2).fit(X, y)
    gx, gy = np.meshgrid(np.linspace(0, 1, 250), np.linspace(0, 1, 250))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.2))
    cmap = {"A": viz.SERIES[0], "B": viz.SERIES[1], "C": viz.SERIES[2]}
    for ax, model, title in [(axes[0], init, "Generation 0: random recruits"),
                             (axes[1], clf, "Generation 10: matured memory cells")]:
        lab, dist, rec = model.predict_with_affinity(np.c_[gx.ravel(), gy.ravel()])
        img = np.array([[int(l == "A"), int(l == "B"), int(l == "C")] for l in lab]).argmax(1).reshape(gx.shape)
        from matplotlib.colors import ListedColormap
        ax.contourf(gx, gy, img, levels=[-.5, .5, 1.5, 2.5], alpha=0.10,
                    cmap=ListedColormap([cmap["A"], cmap["B"], cmap["C"]]))
        unk = (~rec).reshape(gx.shape)
        ax.contourf(gx, gy, unk.astype(float), levels=[0.5, 1.5], colors=[viz.NEUTRAL], alpha=0.18)
        for c in "ABC":
            ax.scatter(*X[y == c].T, s=5, color=cmap[c], alpha=0.35)
            m = model.memory_[model.memory_labels_ == c]
            ax.scatter(*m.T, s=110, marker="*", color=cmap[c], edgecolor=viz.TEXT, linewidth=0.8,
                       label=f"memory cells class {c}", zorder=4)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.grid(False)
        ax.set_title(title, fontsize=10.5)
    axes[1].text(0.02, 0.97, "grey = 'not recognised' (novel)", transform=axes[1].transAxes,
                 va="top", fontsize=8.5, color=viz.TEXT_2)
    axes[0].legend(loc="upper left", fontsize=8.5)
    fig.suptitle("Clonal selection: memory cells migrate toward the patterns they recognise",
                 fontsize=12, fontweight="bold")
    savefig(fig, figs / "fig02_clonal_selection_2d.png")


# ------------------------- E1b: why random receptors fail in high dimension
def e1b_dimensionality(fit_df, val_df, figs, quick) -> dict:
    """V-Detector detection rate vs. shape-space dimension (validation set).

    Reproduces the scaling critique of Stibor et al. (2005) and shows how
    self-proximal receptor generation mitigates it.
    """
    dims = [2, 4, 8, 12] if quick else [2, 4, 8, 12, 16, 24]
    rows = []
    yv_att = (val_df["family"].to_numpy() != "normal").astype(int)
    for k in dims:
        ss = ShapeSpace(n_components=k).fit(fit_df)
        Xf, Xv = ss.transform(fit_df), ss.transform(val_df)
        normal = np.flatnonzero(fit_df["family"].to_numpy() == "normal")
        X_self = Xf[np.random.default_rng(0).choice(normal, min(8000, len(normal)), replace=False)]
        for sampler in ["uniform", "self_proximal"]:
            vd = VDetector(self_radius=0.01, candidate_sampler=sampler, proximity_sigma=0.15,
                           random_state=0).fit(X_self)
            m = binary_metrics(yv_att, vd.predict(Xv))
            rows.append({"dims": k, "sampler": sampler, "n_detectors": vd.n_detectors, **m})
            print(f"[E1b] k={k} {sampler}: DR={m['DR']:.3f} FAR={m['FAR']:.3f} n={vd.n_detectors}")
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    for i, (sampler, label) in enumerate([("uniform", "uniform receptors (classic)"),
                                          ("self_proximal", "self-proximal receptors")]):
        d = df[df.sampler == sampler]
        ax.plot(d.dims, d.DR, marker="o", markersize=7, color=viz.SERIES[i], label=f"{label} - detection rate")
        ax.plot(d.dims, d.FAR, marker="s", markersize=6, linestyle="--", color=viz.SERIES[i],
                label=f"{label} - false-alarm rate")
    ax.set_xlabel("Shape-space dimension k (PCA components)"); ax.set_ylabel("Rate on validation set")
    ax.set_ylim(0, 1.02); ax.set_xticks(dims)
    ax.set_title("E1b - Negative selection vs. dimensionality")
    ax.legend(fontsize=8, loc="center right")
    savefig(fig, figs / "fig03b_dimensionality.png")
    return {"rows": rows}


# -------------------------------------------------- E1: anomaly detection
def e1_anomaly(Xf, yf, Xv, yv, Xt, yt, novel, seeds, figs, quick) -> dict:
    rng = np.random.default_rng(0)
    n_self = 3000 if quick else 8000
    normal_idx = np.flatnonzero(yf == "normal")
    X_self = Xf[rng.choice(normal_idx, n_self, replace=False)]
    yv_att = (yv != "normal").astype(int); yt_att = (yt != "normal").astype(int)

    # --- tune V-Detector on validation (native rule, FAR <= 3%)
    grid = []
    for sampler in ["uniform", "self_proximal"]:
        sigmas = [None] if sampler == "uniform" else [0.1, 0.15, 0.25]
        for rs in [0.01, 0.02, 0.04]:
            for sig in sigmas:
                vd = VDetector(self_radius=rs, candidate_sampler=sampler,
                               proximity_sigma=sig or 0.1, random_state=0).fit(X_self)
                m = binary_metrics(yv_att, vd.predict(Xv))
                grid.append({"sampler": sampler, "self_radius": rs, "sigma": sig,
                             "n_detectors": vd.n_detectors, **m})
                print(f"[E1 grid] {sampler:13s} rs={rs} sig={sig}: DR={m['DR']:.3f} FAR={m['FAR']:.3f} n={vd.n_detectors}")
    gdf = pd.DataFrame(grid)

    def best(sampler):
        g = gdf[(gdf.sampler == sampler) & (gdf.FAR <= 0.03)]
        g = g if len(g) else gdf[gdf.sampler == sampler]
        return g.sort_values(["DR", "FAR"], ascending=[False, True]).iloc[0].to_dict()

    best_u, best_p = best("uniform"), best("self_proximal")
    ns_grid = []
    for rs in [0.01, 0.02, 0.04, 0.08]:
        ns = NearestSelfDetector(rs).fit(X_self)
        ns_grid.append({"self_radius": rs, **binary_metrics(yv_att, ns.predict(Xv))})
    ns_df = pd.DataFrame(ns_grid)
    ns_ok = ns_df[ns_df.FAR <= 0.03]
    best_ns = (ns_ok if len(ns_ok) else ns_df).sort_values("DR", ascending=False).iloc[0].to_dict()

    # --- final evaluation over seeds
    def score_eval(name, score_v_normal, score_t, native_pred=None, extra=None):
        thr = thr_at_far(score_v_normal)
        res = {"AUC": float(roc_auc_score(yt_att, score_t)),
               "at_2pct_val_FAR": binary_metrics(yt_att, (score_t > thr).astype(int), novel)}
        if native_pred is not None:
            res["native"] = binary_metrics(yt_att, native_pred, novel)
        res.update(extra or {})
        return res

    results, roc_store = {}, {}
    for name, params in [("V-Detector (uniform)", best_u), ("V-Detector (self-proximal)", best_p)]:
        runs = []
        for s in seeds:
            t0 = time.perf_counter()
            vd = VDetector(self_radius=params["self_radius"], candidate_sampler=params["sampler"],
                           proximity_sigma=params["sigma"] or 0.1, random_state=s).fit(X_self)
            fit_t = time.perf_counter() - t0
            sv = vd.decision_function(Xv[yv == "normal"]); st = vd.decision_function(Xt)
            runs.append(score_eval(name, sv, st, vd.predict(Xt),
                                   {"n_detectors": vd.n_detectors, "fit_seconds": fit_t}))
            if s == seeds[0]:
                roc_store[name] = roc_curve(yt_att, st)[:2]
        results[name] = {"params": {k: params[k] for k in ("sampler", "self_radius", "sigma")}, **agg(runs)}

    deterministic = {
        "Nearest-self (positive selection)": lambda: NearestSelfDetector(best_ns["self_radius"]),
        "Isolation Forest": lambda: IsolationForest(n_estimators=200, random_state=0),
        "One-class SVM": lambda: OneClassSVM(nu=0.02, gamma="scale"),
        "LOF (novelty)": lambda: LocalOutlierFactor(n_neighbors=20, novelty=True),
    }
    for name, make in deterministic.items():
        t0 = time.perf_counter()
        model = make().fit(X_self)
        fit_t = time.perf_counter() - t0
        sign = 1.0 if isinstance(model, NearestSelfDetector) else -1.0
        sv = sign * model.decision_function(Xv[yv == "normal"]); st = sign * model.decision_function(Xt)
        native = model.predict(Xt)
        native = native if isinstance(model, NearestSelfDetector) else (native == -1).astype(int)
        results[name] = score_eval(name, sv, st, native, {"fit_seconds": fit_t})
        roc_store[name] = roc_curve(yt_att, st)[:2]
        print(f"[E1] {name}: AUC={results[name]['AUC']:.3f}")

    # --- ROC figure
    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    order = ["V-Detector (self-proximal)", "V-Detector (uniform)", "Nearest-self (positive selection)",
             "Isolation Forest", "One-class SVM", "LOF (novelty)"]
    for i, name in enumerate(order):
        fpr, tpr = roc_store[name]
        auc = results[name]["AUC"]; auc = auc["mean"] if isinstance(auc, dict) else auc
        ax.plot(fpr, tpr, color=viz.SERIES[i], label=f"{name}  (AUC {auc:.3f})",
                linewidth=2.4 if "V-Detector" in name else 1.6)
    ax.plot([0, 1], [0, 1], color=viz.NEUTRAL, linewidth=1, linestyle="--")
    ax.axvline(TARGET_FAR, color=viz.TEXT_2, linewidth=0.8, linestyle=":")
    ax.text(TARGET_FAR + 0.01, 0.05, "2% alert budget", fontsize=8.5, color=viz.TEXT_2)
    ax.set_xlabel("False-alarm rate (normal traffic flagged)"); ax.set_ylabel("Detection rate (attacks flagged)")
    ax.set_title("E1 - Anomaly detection on KDDTest+ (trained on normal traffic only)")
    ax.legend(loc="lower right", fontsize=8)
    savefig(fig, figs / "fig03_anomaly_roc.png")
    return {"grid_vdetector": grid, "grid_nearest_self": ns_grid, "models": results,
            "n_self": n_self}


# ------------------------------------------- E2: attack-pattern recognition
def e2_recognition(Xf, yf, Xv, yv, Xt, yt, seeds, figs, quick) -> tuple[dict, dict]:
    grid = []
    ms_opts = [40, 80] if quick else [40, 80, 150]
    for alloc in ["balanced", "sqrt"]:
        for m in ms_opts:
            for mut in [0.3, 0.8]:
                for nso in [True, False]:
                    c = ClonalSelectionClassifier(n_memory_per_class=m, memory_allocation=alloc,
                                                  mutation_scale=mut, negative_selection_operator=nso,
                                                  n_generations=3 if quick else 6, random_state=0).fit(Xf, yf)
                    f1 = f1_score(yv, c.predict(Xv), average="macro")
                    grid.append({"memory_allocation": alloc, "n_memory_per_class": m, "mutation_scale": mut,
                                 "nso": nso, "n_cells": c.n_memory_cells, "val_macroF1": float(f1)})
                    print(f"[E2 grid] {alloc} m={m} mut={mut} nso={nso}: val macro-F1={f1:.3f}")
    best = max(grid, key=lambda g: g["val_macroF1"])
    best_params = dict(n_memory_per_class=best["n_memory_per_class"], memory_allocation=best["memory_allocation"],
                       mutation_scale=best["mutation_scale"], negative_selection_operator=best["nso"],
                       n_generations=3 if quick else 6)

    def evaluate(pred, fit_t, pred_t, size):
        return {"accuracy": float(accuracy_score(yt, pred)),
                "macro_F1": float(f1_score(yt, pred, average="macro")),
                "recall": per_class_recall(yt, pred), "fit_seconds": fit_t,
                "predict_ms_per_1k": 1000 * pred_t / len(Xt) * 1000, "model_size": size}

    res, preds = {}, {}
    runs_c, runs_r = [], []
    for s in seeds:
        t0 = time.perf_counter(); c = ClonalSelectionClassifier(random_state=s, **best_params).fit(Xf, yf)
        ft = time.perf_counter() - t0; t0 = time.perf_counter(); p = c.predict(Xt); pt = time.perf_counter() - t0
        runs_c.append(evaluate(p, ft, pt, c.n_memory_cells))
        if s == seeds[0]:
            preds["CLONALG memory"] = p
        # Ablation: same number of prototypes per class, picked at random, no maturation.
        rng = np.random.default_rng(s)
        idx = np.concatenate([rng.choice(np.flatnonzero(yf == k), int(np.sum(c.memory_labels_ == k)),
                                         replace=False) for k in FAMILIES])
        t0 = time.perf_counter(); kn = KNeighborsClassifier(1).fit(Xf[idx], yf[idx]); ft = time.perf_counter() - t0
        t0 = time.perf_counter(); p = kn.predict(Xt); pt = time.perf_counter() - t0
        runs_r.append(evaluate(p, ft, pt, len(idx)))
    res["CLONALG memory (AIS)"] = agg(runs_c)
    res["Random prototypes (ablation)"] = agg(runs_r)

    n_trees = 50 if quick else 100
    others = {
        "1-NN (all training data)": lambda: KNeighborsClassifier(1),
        "Random Forest": lambda: RandomForestClassifier(n_trees, n_jobs=-1, random_state=0),
        "Random Forest (balanced)": lambda: RandomForestClassifier(n_trees, n_jobs=-1, random_state=0,
                                                                   class_weight="balanced"),
    }
    for name, make in others.items():
        t0 = time.perf_counter(); m = make().fit(Xf, yf); ft = time.perf_counter() - t0
        t0 = time.perf_counter(); p = m.predict(Xt); pt = time.perf_counter() - t0
        size = len(Xf) if name.startswith("1-NN") else int(sum(e.tree_.node_count for e in m.estimators_))
        res[name] = evaluate(p, ft, pt, size)
        preds[name] = p
        print(f"[E2] {name}: macro-F1={res[name]['macro_F1']:.3f}")

    # per-family recall figure
    names = ["CLONALG memory (AIS)", "Random prototypes (ablation)", "1-NN (all training data)",
             "Random Forest", "Random Forest (balanced)"]
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    w = 0.16
    x = np.arange(len(FAMILIES))
    for i, n in enumerate(names):
        rec = res[n]["recall"]
        vals = [rec[f]["mean"] if isinstance(rec[f], dict) else rec[f] for f in FAMILIES]
        errs = [rec[f]["std"] if isinstance(rec[f], dict) else 0 for f in FAMILIES]
        ax.bar(x + (i - 2) * w, vals, w * 0.92, yerr=errs, color=viz.SERIES[i], label=n,
               error_kw={"elinewidth": 0.8, "ecolor": viz.TEXT_2})
    ax.set_xticks(x, FAMILIES); ax.set_ylim(0, 1.05); ax.set_ylabel("Recall on KDDTest+")
    ax.set_title("E2 - Which attack patterns does each model recognise?")
    ax.legend(ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    savefig(fig, figs / "fig04_recall_per_family.png")

    # confusion matrix of CLONALG
    cm = confusion_matrix(yt, preds["CLONALG memory"], labels=FAMILIES, normalize="true")
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=1); ax.grid(False)
    for i in range(len(FAMILIES)):
        for j in range(len(FAMILIES)):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if cm[i, j] > 0.55 else viz.TEXT)
    ax.set_xticks(range(5), FAMILIES); ax.set_yticks(range(5), FAMILIES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("CLONALG memory - row-normalised confusion")
    savefig(fig, figs / "fig05_confusion_clonalg.png")
    return {"grid": grid, "best_params": best_params, "models": res}, best_params


# -------------------------------------------- E3: two-layer immune system
def e3_two_layer(Xf, yf, Xv, yv, Xt, yt, novel, vd_params, cs_params, seeds, figs) -> dict:
    yt_att = (yt != "normal").astype(int); yv_att = (yv != "normal").astype(int)
    out = {"configs": {}}
    preds = {}
    for q in [0.90, 0.95, 0.99]:
        runs = {"tolerance": [], "alarm_union": []}
        for s in seeds:
            rec = ImmuneRecognizer(
                vdetector=VDetector(self_radius=vd_params["self_radius"], candidate_sampler=vd_params["sampler"],
                                    proximity_sigma=vd_params["sigma"] or 0.1, random_state=s),
                clonal=ClonalSelectionClassifier(random_state=s, recognition_quantile=q, **cs_params))
            rng = np.random.default_rng(s)
            # V-Detector trained on a self sample (speed); memory on all FIT data.
            pool_n = np.flatnonzero(yf == "normal")
            self_idx = rng.choice(pool_n, min(8000, len(pool_n)), replace=False)
            rec.vdetector.fit(Xf[self_idx]); rec.clonal.fit(Xf, yf)
            for policy in runs:
                rec.policy = policy
                pv = rec.predict(Xv)
                ex = rec.explain(Xt); p = ex["final"]
                att = (p != "normal").astype(int)
                fam_ok = np.where(p == NOVEL, "__wrong__", p)
                runs[policy].append({
                    "val_F1": binary_metrics(yv_att, (pv != "normal").astype(int))["F1"],
                    **binary_metrics(yt_att, att, novel),
                    "macro_F1_families": float(f1_score(yt, fam_ok, labels=FAMILIES, average="macro")),
                    "novel_label_rate_on_novel_types": float((p[novel] == NOVEL).mean()),
                    "novel_label_rate_on_known_types": float((p[~novel] == NOVEL).mean()),
                    "memory_only_DR": float((ex["memory_label"][yt_att == 1] != "normal").mean()),
                    "memory_only_FAR": float((ex["memory_label"][yt_att == 0] != "normal").mean()),
                    "memory_only_novel_DR": float((ex["memory_label"][novel] != "normal").mean()),
                })
                if s == seeds[0]:
                    preds[(q, policy)] = p
        for policy, r in runs.items():
            key = f"q={q}|{policy}"
            out["configs"][key] = {"recognition_quantile": q, "policy": policy, **agg(r)}
            a = out["configs"][key]
            print(f"[E3] {key}: val F1 {a['val_F1']['mean']:.3f} | test DR {a['DR']['mean']:.3f} "
                  f"FAR {a['FAR']['mean']:.3f} novel DR {a['novel_DR']['mean']:.3f}")
    best_key = max(out["configs"], key=lambda k: out["configs"][k]["val_F1"]["mean"])
    out["selected_on_validation"] = best_key
    best_cfg = out["configs"][best_key]
    keep_pred = preds[(best_cfg["recognition_quantile"], best_cfg["policy"])]

    rf = RandomForestClassifier(100, n_jobs=-1, random_state=0).fit(Xf, yf)
    prf = rf.predict(Xt)
    out["Random Forest"] = binary_metrics(yt_att, (prf != "normal").astype(int), novel)

    # stacked outcome bars: what happens to each true family (selected configuration)
    cats = ["correct family", "other attack family", "flagged NOVEL", "missed (called normal)"]
    rows = FAMILIES[1:] + ["novel types"]
    data = []
    for r in rows:
        mask = novel if r == "novel types" else ((yt == r) & ~novel)
        p = keep_pred[mask]; truth = yt[mask]
        data.append([np.mean(p == truth), np.mean((p != truth) & (p != NOVEL) & (p != "normal")),
                     np.mean(p == NOVEL), np.mean(p == "normal")])
    data = np.array(data)
    out["outcome_table_selected"] = {r: dict(zip(cats, map(float, d))) for r, d in zip(rows, data)}
    plot_e3(out["outcome_table_selected"], best_key, figs)
    return out


def plot_e3(table: dict, best_key: str, figs: Path) -> None:
    cats = ["correct family", "other attack family", "flagged NOVEL", "missed (called normal)"]
    rows = list(table)
    labels = [f"{r} (known types)" if r != "novel types" else "17 novel test-only types" for r in rows]
    data = np.array([[table[r][c] for c in cats] for r in rows])
    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    colors = [viz.SERIES[2], viz.SERIES[3], viz.NEUTRAL, viz.SERIES[7]]
    left = np.zeros(len(rows))
    for j, c in enumerate(cats):
        ax.barh(labels, data[:, j], left=left, color=colors[j], label=c, edgecolor=viz.SURFACE, linewidth=2)
        for i, v in enumerate(data[:, j]):
            if v > 0.07:
                ax.text(left[i] + v / 2, i, f"{v:.0%}", ha="center", va="center", fontsize=8.5,
                        color="white" if j == 3 else viz.TEXT)
        left += data[:, j]
    ax.set_xlim(0, 1); ax.invert_yaxis(); ax.set_xlabel("Share of test connections")
    ax.grid(axis="y", visible=False)
    ax.set_title(f"E3 - Two-layer immune recognizer: fate of each attack family (KDDTest+)\n"
                 f"configuration selected on validation: {best_key.replace('|', ', ')}", fontsize=11)
    ax.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    savefig(fig, figs / "fig06_two_layer_outcomes.png")


# ------------------------------- E4: zero-day hold-out + secondary response
def e4_zero_day(Xf, yf, Xt, yt, vd_params, cs_params, q, policy, seeds, figs, quick) -> dict:
    shots_list = [5, 20, 80]
    held_list = ["Probe"] if quick else ["Probe", "DoS", "R2L"]
    out = {}
    for held in held_list:
        keep = yf != held
        runs = []
        for s in seeds:
            rng = np.random.default_rng(s)
            pool_n = np.flatnonzero(yf == "normal")
            self_idx = rng.choice(pool_n, min(8000, len(pool_n)), replace=False)
            rec = ImmuneRecognizer(
                vdetector=VDetector(self_radius=vd_params["self_radius"], candidate_sampler=vd_params["sampler"],
                                    proximity_sigma=vd_params["sigma"] or 0.1, random_state=s),
                clonal=ClonalSelectionClassifier(random_state=s, recognition_quantile=q, **cs_params),
                policy=policy)
            rec.vdetector.fit(Xf[self_idx]); rec.clonal.fit(Xf[keep], yf[keep])
            rf = RandomForestClassifier(100, n_jobs=-1, random_state=s).fit(Xf[keep], yf[keep])
            mask = yt == held
            p_imm = rec.predict(Xt[mask]); p_rf = rf.predict(Xt[mask])
            r = {"immune_detected": float(np.mean(p_imm != "normal")),
                 "immune_flagged_novel": float(np.mean(p_imm == NOVEL)),
                 "rf_detected": float(np.mean(p_rf != "normal")),
                 "immune_FAR": float(np.mean(rec.predict(Xt[yt == "normal"]) != "normal")),
                 "rf_FAR": float(np.mean(rf.predict(Xt[yt == "normal"]) != "normal")),
                 "shots": {}}
            pool = np.flatnonzero(yf == held)
            for k in shots_list:
                pick = rng.choice(pool, k, replace=False)
                imm = copy.deepcopy(rec)
                t0 = time.perf_counter()
                imm.learn_new_pattern(Xf[pick], held, n_memory=min(k, cs_params["n_memory_per_class"]))
                t_imm = time.perf_counter() - t0
                t0 = time.perf_counter()
                rf2 = RandomForestClassifier(100, n_jobs=-1, random_state=s).fit(
                    np.vstack([Xf[keep], Xf[pick]]), np.concatenate([yf[keep], yf[pick]]))
                t_rf = time.perf_counter() - t0
                r["shots"][str(k)] = {
                    "immune_recall": float(np.mean(imm.predict(Xt[mask]) == held)),
                    "rf_recall": float(np.mean(rf2.predict(Xt[mask]) == held)),
                    "immune_update_seconds": t_imm, "rf_retrain_seconds": t_rf}
            runs.append(r)
            print(f"[E4] held-out {held} seed {s}: immune det {r['immune_detected']:.3f} "
                  f"(novel {r['immune_flagged_novel']:.3f}), RF det {r['rf_detected']:.3f}")
        out[held] = agg(runs)

    plot_e4(out, shots_list, figs)
    return out


def plot_e4(out: dict, shots_list: list, figs: Path) -> None:
    held_list = list(out)
    fig, axes = plt.subplots(1, len(held_list), figsize=(4.2 * len(held_list), 3.9), squeeze=False)
    for ax, held in zip(axes[0], held_list):
        for i, (key, label) in enumerate([("immune_recall", "Immune memory (incremental, no retrain)"),
                                          ("rf_recall", "Random Forest (full retrain)")]):
            m = [out[held]["shots"][str(k)][key]["mean"] for k in shots_list]
            sd = [out[held]["shots"][str(k)][key]["std"] for k in shots_list]
            ax.errorbar(shots_list, m, yerr=sd, marker="o", markersize=7, capsize=3,
                        color=viz.SERIES[i], label=label)
        ax.set_xscale("log"); ax.minorticks_off(); ax.set_xticks(shots_list, [str(k) for k in shots_list])
        ax.set_ylim(0, 1.02); ax.set_title(f"Held-out family: {held}", fontsize=10.5)
        ax.set_xlabel("Labelled examples of the new family")
    axes[0][0].set_ylabel("Recall of new family on KDDTest+")
    axes[0][0].legend(fontsize=8, loc="lower right")
    fig.suptitle("E4 - Secondary response: learning a never-seen attack family from a few examples",
                 fontsize=12, fontweight="bold", y=1.05)
    savefig(fig, figs / "fig07_secondary_response.png")



# ------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="small smoke-test run")
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args()

    viz.apply_style()
    out = Path(args.out); figs = out / "figures"; figs.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    train, test = load_nsl_kdd(args.data_dir)
    if args.quick:
        idx = np.concatenate([g.sample(min(len(g), 6000), random_state=0).index
                              for _, g in train.groupby("family")])
        train = train.loc[np.sort(idx)]
    fit_df, val_df = train_test_split(train, test_size=0.2, stratify=train["family"], random_state=0)
    ss = ShapeSpace(n_components=12).fit(fit_df)
    Xf, Xv, Xt = ss.transform(fit_df), ss.transform(val_df), ss.transform(test)
    yf, yv, yt = (d["family"].to_numpy(dtype=object) for d in (fit_df, val_df, test))
    novel = test["novel_type"].to_numpy()
    seeds = [0, 1] if args.quick else [0, 1, 2, 3, 4]

    summary = {
        "dataset": {"fit_rows": len(fit_df), "val_rows": len(val_df), "test_rows": len(test),
                    "test_novel_type_rows": int(novel.sum()),
                    "test_novel_types": sorted(test.loc[novel, "attack_type"].unique().tolist()),
                    "family_counts_train": train["family"].value_counts().to_dict(),
                    "family_counts_test": test["family"].value_counts().to_dict(),
                    "shape_space_dims": ss.n_components,
                    "pca_explained_variance": ss.explained_variance},
        "seeds": seeds, "quick": args.quick,
    }
    e0_concepts(figs)
    summary["E1b_dimensionality"] = e1b_dimensionality(fit_df, val_df, figs, args.quick)
    summary["E1_anomaly"] = e1_anomaly(Xf, yf, Xv, yv, Xt, yt, novel, seeds, figs, args.quick)
    vd_params = summary["E1_anomaly"]["models"]["V-Detector (self-proximal)"]["params"]
    summary["E2_recognition"], cs_params = e2_recognition(Xf, yf, Xv, yv, Xt, yt, seeds, figs, args.quick)
    summary["E3_two_layer"] = e3_two_layer(Xf, yf, Xv, yv, Xt, yt, novel, vd_params, cs_params, seeds[:3], figs)
    sel = summary["E3_two_layer"]["configs"][summary["E3_two_layer"]["selected_on_validation"]]
    summary["E4_zero_day"] = e4_zero_day(Xf, yf, Xt, yt, vd_params, cs_params, sel["recognition_quantile"],
                                         sel["policy"], seeds[:2], figs, args.quick)
    summary["runtime_seconds"] = time.perf_counter() - t_start

    with open(out / "metrics.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"[done] {summary['runtime_seconds']:.0f}s -> {out / 'metrics.json'}")


if __name__ == "__main__":
    main()
