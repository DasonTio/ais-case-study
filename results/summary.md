# Results summary

Generated from `metrics.json`. Values are mean ± std over seeds where the model is stochastic. DR = detection rate, FAR = false-alarm rate.

## E1 - Anomaly detection (trained on normal traffic only), KDDTest+

| Model | AUC | DR (native rule) | FAR (native rule) | Novel-type DR | Detectors |
|---|---|---|---|---|---|
| V-Detector (uniform) | 0.695 ± 0.195 | 3.2 ± 1.5% | 0.2 ± 0.1% | 4.0 ± 4.1% | 18 ± 7 |
| V-Detector (self-proximal) | 0.825 ± 0.028 | 60.1 ± 1.4% | 2.4 ± 0.3% | 37.6 ± 4.7% | 275 ± 14 |
| Nearest-self (positive selection) | 0.953 | 62.0% | 2.7% | 43.9% | - |
| Isolation Forest | 0.922 | 60.9% | 4.9% | 60.6% | - |
| One-class SVM | 0.622 | 50.8% | 1.4% | 22.4% | - |
| LOF (novelty) | 0.911 | 76.9% | 11.6% | 51.5% | - |

## E1b - V-Detector vs. shape-space dimension (validation)

| Dimensions k | Sampler | Val. DR | Val. FAR | Detectors |
|---|---|---|---|---|
| 2 | uniform | 81.4% | 1.4% | 3000 |
| 2 | self_proximal | 81.4% | 1.3% | 3000 |
| 4 | uniform | 81.4% | 0.4% | 80 |
| 4 | self_proximal | 69.1% | 0.9% | 436 |
| 8 | uniform | 28.5% | 0.4% | 80 |
| 8 | self_proximal | 89.0% | 0.9% | 365 |
| 12 | uniform | 12.9% | 0.1% | 18 |
| 12 | self_proximal | 84.3% | 0.6% | 331 |
| 16 | uniform | 19.4% | 0.1% | 38 |
| 16 | self_proximal | 92.6% | 0.9% | 929 |
| 24 | uniform | 25.5% | 0.5% | 117 |
| 24 | self_proximal | 96.9% | 2.9% | 3000 |

## E2 - Attack-pattern recognition (5 families), KDDTest+

| Model | Accuracy | Macro-F1 | Recall normal | Recall DoS | Recall Probe | Recall R2L | Recall U2R | Model size | Fit (s) |
|---|---|---|---|---|---|---|---|---|---|
| CLONALG memory (AIS) | 75.7 ± 0.4% | 0.619 ± 0.026 | 92.0 ± 0.3% | 75.0 ± 0.8% | 76.0 ± 1.9% | 23.4 ± 3.2% | 39.1 ± 6.0% | 863 ± 0 | 2.13 ± 0.25 |
| Random prototypes (ablation) | 74.9 ± 1.4% | 0.576 ± 0.025 | 91.9 ± 0.3% | 76.3 ± 2.6% | 74.5 ± 2.5% | 14.3 ± 6.4% | 56.7 ± 5.7% | 863 ± 0 | 0.00 ± 0.00 |
| 1-NN (all training data) | 72.3% | 0.534 | 92.9% | 75.9% | 60.6% | 4.8% | 22.4% | 100778 | 0.30 |
| Random Forest | 74.5% | 0.494 | 97.3% | 78.1% | 61.9% | 0.2% | 6.0% | 115058 | 14.73 |
| Random Forest (balanced) | 74.5% | 0.541 | 97.4% | 77.8% | 56.1% | 5.5% | 17.9% | 109532 | 9.72 |

## E3 - Two-layer immune recognizer, KDDTest+

| Configuration | Val. F1 | Test DR | Test FAR | Novel-type DR | Novel types labelled NOVEL |
|---|---|---|---|---|---|
| q=0.9|tolerance | 0.968 ± 0.001 | 65.1 ± 0.9% | 4.7 ± 1.0% | 39.3 ± 3.9% | 30.6 ± 3.7% |
| q=0.9|alarm_union | 0.973 ± 0.000 | 74.1 ± 1.1% | 8.0 ± 0.2% | 47.3 ± 4.3% | 30.6 ± 3.7% |
| q=0.95|tolerance | 0.974 ± 0.000 | 68.9 ± 1.7% | 7.3 ± 0.1% | 40.7 ± 3.9% | 29.5 ± 3.7% |
| q=0.95|alarm_union | 0.973 ± 0.000 | 74.1 ± 1.0% | 8.0 ± 0.2% | 47.2 ± 4.5% | 29.5 ± 3.7% |
| q=0.99|tolerance (selected) | 0.974 ± 0.001 | 72.1 ± 1.3% | 7.9 ± 0.1% | 43.2 ± 5.5% | 17.6 ± 3.2% |
| q=0.99|alarm_union | 0.974 ± 0.000 | 73.2 ± 0.9% | 7.9 ± 0.1% | 44.7 ± 5.9% | 17.6 ± 3.2% |
| Random Forest (reference) | - | 58.9% | 2.7% | 19.4% | n/a |

## E4a - Zero-day hold-out: detection of a family never seen in training

| Held-out family | Immune: detected | Immune: labelled NOVEL | RF: detected | Immune FAR | RF FAR |
|---|---|---|---|---|---|
| Probe | 60.2 ± 1.2% | 50.0 ± 2.7% | 36.9 ± 1.4% | 6.6 ± 0.3% | 0.8 ± 0.0% |
| DoS | 71.8 ± 0.9% | 55.3 ± 3.0% | 2.1 ± 0.3% | 3.9 ± 0.1% | 2.1 ± 0.0% |
| R2L | 23.3 ± 0.0% | 8.5 ± 0.0% | 0.8 ± 0.0% | 7.5 ± 0.1% | 2.7 ± 0.0% |

## E4b - Secondary response: learning the held-out family from a few labelled examples

| Held-out family | Shots | Immune recall | RF recall | Immune update (s) | RF retrain (s) |
|---|---|---|---|---|---|
| Probe | 5 | 52.0 ± 2.7% | 11.7 ± 11.0% | 0.003 ± 0.001 | 14.3 ± 0.5 |
| Probe | 20 | 59.7 ± 3.7% | 34.2 ± 4.5% | 0.007 ± 0.001 | 13.7 ± 0.3 |
| Probe | 80 | 64.1 ± 0.5% | 40.5 ± 1.4% | 0.030 ± 0.002 | 13.6 ± 0.7 |
| DoS | 5 | 41.8 ± 19.6% | 33.4 ± 14.8% | 0.002 ± 0.001 | 9.5 ± 0.3 |
| DoS | 20 | 64.1 ± 2.3% | 59.8 ± 0.2% | 0.006 ± 0.000 | 10.0 ± 0.2 |
| DoS | 80 | 69.1 ± 1.4% | 63.2 ± 0.1% | 0.023 ± 0.003 | 9.7 ± 0.8 |
| R2L | 5 | 1.6 ± 0.3% | 0.0 ± 0.0% | 0.002 ± 0.000 | 14.8 ± 0.6 |
| R2L | 20 | 9.4 ± 6.6% | 0.0 ± 0.0% | 0.009 ± 0.001 | 14.5 ± 0.2 |
| R2L | 80 | 10.7 ± 3.0% | 0.0 ± 0.0% | 0.032 ± 0.000 | 14.4 ± 0.6 |
