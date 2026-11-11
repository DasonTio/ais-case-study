# ais_patrec: immune-inspired pattern recognition for network intrusion traffic

A two-layer **Artificial Immune System (AIS)** that reads network connection records,
*recognises* known attack patterns (DoS, Probe, R2L, U2R), and *flags* patterns it has never
seen as `NOVEL`. Analysts can then teach it a new pattern from a few labelled examples without
retraining.

The experiments are on **NSL-KDD** (Tavallaee et al., 2009). Its test set contains 17 attack
types that never appear in the training set, which gives a built-in zero-day test.

```
connection record ─► shape-space (log, one-hot, PCA-12, [0,1])
                         │
                         ▼
        ┌──────── Layer 1: B-cell memory (clonal selection) ────────┐
        │  nearest memory cell + affinity threshold                 │
        │  recognised? ──yes──► label: normal / DoS / Probe / R2L / U2R
        └──────────────┬────────────────────────────────────────────┘
                       │ no (unfamiliar pattern)
                       ▼
        ┌──────── Layer 2: T-cell tolerance (negative selection) ───┐
        │  V-Detector trained on normal traffic only                │
        │  inside a detector? ──yes──► NOVEL (send to an analyst)    │
        │                     ──no───► normal (tolerance policy)     │
        └───────────────────────────────────────────────────────────┘
                       │ analyst labels a few NOVEL records
                       ▼
        learn_new_pattern(): new memory cells, old memory unchanged ("secondary response")
```

## The immune system in one paragraph each

**Negative selection (T-cells, `VDetector`).** The thymus works like a boot camp. It makes
recruits with random receptors, shows each one samples of *your own* body, and destroys any
recruit that reacts. The survivors react only to what is *not you*, so the body can raise an
alarm for an invader it has never met. In code, a detector is a random point in feature space.
It survives only if it stays clear of every normal sample, and it grows until it touches normal
traffic (Forrest et al., 1994; Ji & Dasgupta, 2004).

**Clonal selection (B-cells, `ClonalSelectionClassifier`).** Think of a locksmith with a box of
keys. Shown a new lock, he takes the closest keys and cuts many slightly different copies of
each: few careful cuts for a key that nearly fits, many rough cuts for a poor one. He keeps any
copy that fits better. The keys he keeps are *memory cells*, so the second time the same lock
appears he opens it immediately (de Castro & Von Zuben, 2002).

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q                                      # 10 unit tests, no download needed
python experiments/run_experiments.py --quick  # smoke test, about 2 minutes
python experiments/run_experiments.py          # full run, about 10 minutes on 2 CPUs
python experiments/make_tables.py              # results/metrics.json -> results/summary.md
python experiments/make_report_figures.py      # report figures from results/metrics.json
```

The dataset (about 22 MB) downloads automatically into `data/` the first time you run it.

```python
from ais_patrec import load_nsl_kdd, ShapeSpace, ImmuneRecognizer, VDetector, ClonalSelectionClassifier

train, test = load_nsl_kdd("data")
space = ShapeSpace(n_components=12).fit(train)
X, y = space.transform(train), train["family"].to_numpy(dtype=object)

ais = ImmuneRecognizer(
    vdetector=VDetector(self_radius=0.01, candidate_sampler="self_proximal", proximity_sigma=0.25),
    clonal=ClonalSelectionClassifier(n_memory_per_class=150, memory_allocation="sqrt",
                                     recognition_quantile=0.99),
)
ais.fit(X, y)
signals = ais.explain(space.transform(test))   # final label, memory label, affinity, non-self flag
```

## Project layout

| Path | What it is |
|---|---|
| `ais_patrec/data.py` | NSL-KDD loader, attack-family mapping, `ShapeSpace` preprocessing |
| `ais_patrec/negative_selection.py` | `VDetector` (uniform and self-proximal receptor generation), `NearestSelfDetector` baseline |
| `ais_patrec/clonal_selection.py` | `ClonalSelectionClassifier`: selection, clonal expansion, hypermutation, negative-selection operator, receptor editing, affinity thresholds, `learn_new_pattern` |
| `ais_patrec/immune_recognizer.py` | `ImmuneRecognizer`: the two-layer system |
| `ais_patrec/viz.py` | Plot style with a colour-vision-deficiency-checked palette |
| `experiments/run_experiments.py` | Experiments E0 to E4: every number and figure in the report |
| `experiments/make_tables.py` | Markdown tables from `results/metrics.json` |
| `results/` | `metrics.json`, `summary.md`, `figures/*.png` |
| `experiments/make_report_figures.py` | Figures and equation images used in the report (`report/figures/`) |
| `report/` | The written report (`.docx`, imports straight into Google Docs) and an HTML version |
| `tests/` | Unit tests for the immune guarantees (no detector covers self, memory frozen during secondary response, and so on) |

## Evaluation protocol

* KDDTrain+ is split 80/20 (stratified) into FIT and VALIDATION. Every hyper-parameter and
  threshold is chosen on VALIDATION. KDDTest+ is used only for the final numbers.
* Stochastic models run over 5 seeds (2 to 3 for the slower experiments). Tables report mean ± std.
* Baselines: Isolation Forest, One-class SVM, LOF, a nearest-self rule (Stibor et al., 2005),
  1-NN on all data, Random Forest (plain and class-balanced), and an ablation that picks
  random prototypes without affinity maturation.

See `results/summary.md` for every table. The report discusses the results, including where the
AIS loses.

## Design decisions worth arguing about

1. **Self-proximal receptors.** Classic V-Detector draws candidate detectors uniformly in the
   unit cube. Above roughly 8 dimensions, the coverage estimate fills with empty corners of the
   cube and detection collapses (experiment E1b). Here, candidates are drawn around self samples
   instead, following density-guided NSAs (Yang, Chen & Li, 2017).
2. **PCA shape-space.** Negative selection needs a low-dimensional, meaningful geometry, so the
   ~120 raw features (after one-hot encoding) are projected to 12 components (78% of variance).
3. **Recognition threshold.** A memory cell "binds" only within a per-class radius. Without this
   threshold, a nearest-prototype classifier gives a confident wrong label to every unfamiliar
   pattern. The radius also accounts for the spacing between training examples, so a class
   learned from 5 examples can still be recognised.
4. **`sqrt` memory allocation.** Equal memory per class over-represents rare attacks and raises
   false alarms on normal traffic. A budget proportional to √(class size) was selected on
   validation.

## References

* Forrest, Perelson, Allen & Cherukuri (1994). *Self-nonself discrimination in a computer.* IEEE S&P. doi:10.1109/risp.1994.296580
* Kim & Bentley (2001). *Towards an AIS for network intrusion detection: clonal selection with a negative selection operator.* CEC. doi:10.1109/cec.2001.934333
* de Castro & Von Zuben (2002). *Learning and optimization using the clonal selection principle.* IEEE TEVC. doi:10.1109/tevc.2002.1011539
* Watkins & Boggess (2002). *A resource limited artificial immune classifier (AIRS).* CEC. doi:10.1109/cec.2002.1007049
* Ji & Dasgupta (2004). *Real-valued negative selection algorithm with variable-sized detectors.* GECCO. doi:10.1007/978-3-540-24854-5_30
* Ji & Dasgupta (2009). *V-detector: an efficient negative selection algorithm with "probably adequate" detector coverage.* Inf. Sci. doi:10.1016/j.ins.2008.12.015
* Stibor, Mohr, Timmis & Eckert (2005). *Is negative selection appropriate for anomaly detection?* GECCO. doi:10.1145/1068009.1068061
* Stibor, Timmis & Eckert (2005). *A comparative study of real-valued negative selection to statistical anomaly detection techniques.* ICARIS. doi:10.1007/11536444_20
* Tavallaee, Bagheri, Lu & Ghorbani (2009). *A detailed analysis of the KDD CUP 99 data set.* CISDA. doi:10.1109/cisda.2009.5356528
* Yang, Chen & Li (2017). *An antigen space density based real-value negative selection algorithm.* Applied Soft Computing. doi:10.1016/j.asoc.2017.09.005
* Greensmith, Aickelin & Cayzer (2005). *Introducing dendritic cells as a novel immune-inspired algorithm for anomaly detection.* ICARIS. doi:10.48550/arxiv.1004.3196
