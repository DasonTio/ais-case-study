"""Turn results/metrics.json into results/summary.md (tables used in the report)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def ms(d, pct=False, digits=3):
    """Format a value or a {mean, std} dict."""
    if isinstance(d, dict) and "mean" in d:
        m, s = d["mean"], d["std"]
        return f"{100*m:.1f} ± {100*s:.1f}%" if pct else f"{m:.{digits}f} ± {s:.{digits}f}"
    if isinstance(d, (int, float)):
        return f"{100*d:.1f}%" if pct else f"{d:.{digits}f}"
    return str(d)


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    return "\n".join(out)


def build(m: dict) -> dict[str, list]:
    t = {}
    e1 = m["E1_anomaly"]["models"]
    t["E1"] = (["Model", "AUC", "DR (native rule)", "FAR (native rule)", "Novel-type DR", "Detectors"],
               [[k, ms(v["AUC"]), ms(v["native"]["DR"], True), ms(v["native"]["FAR"], True),
                 ms(v["native"]["novel_DR"], True), ms(v.get("n_detectors", "-"), digits=0)]
                for k, v in e1.items()])
    e2 = m["E2_recognition"]["models"]
    fams = ["normal", "DoS", "Probe", "R2L", "U2R"]
    t["E2"] = (["Model", "Accuracy", "Macro-F1", *[f"Recall {f}" for f in fams], "Model size", "Fit (s)"],
               [[k, ms(v["accuracy"], True), ms(v["macro_F1"]), *[ms(v["recall"][f], True) for f in fams],
                 ms(v["model_size"], digits=0), ms(v["fit_seconds"], digits=2)] for k, v in e2.items()])
    e3 = m["E3_two_layer"]
    rows = []
    for k, v in e3["configs"].items():
        mark = " (selected)" if k == e3["selected_on_validation"] else ""
        rows.append([k + mark, ms(v["val_F1"]), ms(v["DR"], True), ms(v["FAR"], True),
                     ms(v["novel_DR"], True), ms(v["novel_label_rate_on_novel_types"], True)])
    rf = e3["Random Forest"]
    rows.append(["Random Forest (reference)", "-", ms(rf["DR"], True), ms(rf["FAR"], True),
                 ms(rf["novel_DR"], True), "n/a"])
    t["E3"] = (["Configuration", "Val. F1", "Test DR", "Test FAR", "Novel-type DR", "Novel types labelled NOVEL"], rows)
    e4 = m["E4_zero_day"]
    t["E4a"] = (["Held-out family", "Immune: detected", "Immune: labelled NOVEL", "RF: detected",
                 "Immune FAR", "RF FAR"],
                [[h, ms(v["immune_detected"], True), ms(v["immune_flagged_novel"], True),
                  ms(v["rf_detected"], True), ms(v["immune_FAR"], True), ms(v["rf_FAR"], True)]
                 for h, v in e4.items()])
    rows = []
    for h, v in e4.items():
        for k, s in v["shots"].items():
            rows.append([h, k, ms(s["immune_recall"], True), ms(s["rf_recall"], True),
                         ms(s["immune_update_seconds"], digits=3), ms(s["rf_retrain_seconds"], digits=1)])
    t["E4b"] = (["Held-out family", "Shots", "Immune recall", "RF recall", "Immune update (s)", "RF retrain (s)"], rows)
    t["E1b"] = (["Dimensions k", "Sampler", "Val. DR", "Val. FAR", "Detectors"],
                [[r["dims"], r["sampler"], ms(r["DR"], True), ms(r["FAR"], True), r["n_detectors"]]
                 for r in m["E1b_dimensionality"]["rows"]])
    return t


def main(path: str = str(ROOT / "results" / "metrics.json")) -> None:
    m = json.load(open(path))
    t = build(m)
    titles = {"E1": "E1 - Anomaly detection (trained on normal traffic only), KDDTest+",
              "E1b": "E1b - V-Detector vs. shape-space dimension (validation)",
              "E2": "E2 - Attack-pattern recognition (5 families), KDDTest+",
              "E3": "E3 - Two-layer immune recognizer, KDDTest+",
              "E4a": "E4a - Zero-day hold-out: detection of a family never seen in training",
              "E4b": "E4b - Secondary response: learning the held-out family from a few labelled examples"}
    md = ["# Results summary", "", f"Generated from `{Path(path).name}`. Values are mean ± std over seeds where "
          "the model is stochastic. DR = detection rate, FAR = false-alarm rate.", ""]
    for k in ["E1", "E1b", "E2", "E3", "E4a", "E4b"]:
        md += [f"## {titles[k]}", "", table(*t[k]), ""]
    out = Path(path).with_name("summary.md")
    out.write_text("\n".join(md))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
