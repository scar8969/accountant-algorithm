# h1_power_300 — experiment report

Generated 2026-10-18T12:47:54+00:00 from `results/results.json`.

| field | value |
|---|---|
| config hash | 4cf1fb8e03e77757 |
| random seed | 20261018 |
| code revision | unknown |
| feature schema | 1.0.0 |
| simulation version | 1.0.0 |
| dataset version | synthetic-1.0.0-n300-seed20261018 |
| python | 3.14.7 |

## 1. What this report does and does not show

**Synthetic data only.** Outcomes were produced by the simulator documented in `backend/experiments/simulation.py` (v1.0.0). Beating a baseline here shows the pipeline, splits and metrics work. It is not evidence that the model predicts human attraction.

Reported interest is a self-report proxy, not attraction itself. Scores are on an experimental scale and are not probabilities of a successful relationship, nor psychological assessments.

## 2. Dataset

| quantity | value |
|---|---|
| profiles | 300 |
| candidate pairs generated | 7909 |
| directional interest observations | 9000 |
| mutual-interest observations | 1091 |
| interaction-quality observations | 175 |
| unknown interest directions | 6818 |
| test positives (directional) | 310 |
| test positives (mutual) | 20 |

Unshown or unanswered directions are recorded as **unknown**, never as negative labels.

## 3. Protocol and leakage checks

* Primary protocol: participant_grouped (unseen people), test fraction 0.3, seed 20261018.
* Train pairs 3975, test pairs 694, dropped 3240 pairs that straddled the split.
* Shared participants across the split: 0; shared pairs: 0.
* Temporal protocol: train on first 70% of pairs by timestamp; 300 participants reappear by design.

## 4. Primary comparison (mutual interest)

| model | PR-AUC | 95% CI | ROC-AUC | log loss | ECE | NDCG@5 |
|---|---|---|---|---|---|---|
| random | 0.212 | [0.131, 0.342] | 0.460 | – | – | 0.426 |
| structured | 0.228 | [0.131, 0.373] | 0.497 | – | – | 0.393 |
| text | 0.247 | [0.118, 0.412] | 0.363 | – | – | 0.347 |
| structured_text | 0.232 | [0.132, 0.398] | 0.488 | – | – | 0.405 |
| mutual_independence | 0.331 | [0.211, 0.532] | 0.639 | 0.505 | 0.057 | 0.440 |
| mutual_logistic | 0.370 | [0.215, 0.509] | 0.565 | 0.522 | 0.070 | 0.436 |
| mutual_gbm | 0.293 | [0.179, 0.489] | 0.613 | 0.546 | 0.060 | 0.439 |

> With 20 positive mutual outcomes in the held-out split, the 95% interval for the best model (mutual_logistic) spans 0.29 PR-AUC. Differences smaller than that width are indistinguishable in this sample.

Mutual interest and interaction quality are *pair-level* outcomes: each pair contributes one held-out row, not one per direction. Directional interest legitimately contributes two rows because the two directions have different labels.

Primary metric **pr_auc**: best `mutual_logistic` = 0.370; Δ vs random +0.158; Δ vs structured +0.142.

Intervals overlap heavily at this sample size. Ordering is suggestive, not established.

## 5. Directional interest (A → B)

| model | PR-AUC | 95% CI | ROC-AUC | NDCG@5 | coverage |
|---|---|---|---|---|---|
| random | 0.389 | [0.350, 0.440] | 0.492 | 0.509 | 1.000 |
| structured | 0.484 | [0.432, 0.543] | 0.616 | 0.578 | 0.946 |
| text | 0.408 | [0.366, 0.459] | 0.498 | 0.441 | 0.717 |
| structured_text | 0.484 | [0.433, 0.543] | 0.616 | 0.572 | 0.946 |
| directional_logistic | 0.495 | [0.440, 0.554] | 0.630 | 0.592 | 1.000 |

## 6. Ablation: does text add anything? (H2)

* structured only: 0.228
* structured + lexical embedding: 0.232
* difference: +0.004 (same test pairs)

The embedding here is a deterministic lexical hashing encoder, not a semantic model. This ablation tests whether shared wording helps; it does not test semantic compatibility.

## 7. Calibration

Calibration blocks are reported per probabilistic model in `results.json` (`metrics.calibration.bins`). A calibrated probability means: of pairs scored p, about p were positive in this sample. With few positives the bins are noisy.

The interpretable score additionally carries a **weight-sensitivity band**: the same pair rescored under ±25% log-normal perturbation of the pre-registered weights (200 draws, deterministic per pair). It shows how much the number depends on the weight choice. It is not a confidence interval and says nothing about sampling error.

## 8. Temporal protocol (known people, later observations)

| task | model | PR-AUC | ROC-AUC |
|---|---|---|---|
| directional | random | 0.396 | 0.491 |
| directional | structured | 0.466 | 0.608 |
| directional | text | 0.423 | 0.522 |
| directional | directional_logistic | 0.482 | 0.624 |
| mutual | random | 0.224 | 0.493 |
| mutual | structured | 0.234 | 0.553 |
| mutual | text | 0.254 | 0.571 |
| mutual | mutual_independence | 0.289 | 0.672 |
| mutual | mutual_logistic | 0.300 | 0.658 |

## 9. Interaction quality (Task C)

| model | MAE | RMSE | n |
|---|---|---|---|
| interaction_mean | 0.455 | 0.487 | 16 |
| interaction_ridge | 0.472 | 0.534 | 16 |

Quality labels exist only where both sides consented and both reported interest, so this comparison is underpowered in the synthetic run.

## 10. Failure cases

Scored by `structured` so each case can be explained feature by feature; the run's best learned model may be a different one.

| pair | observed mutual | score | coverage | top positive | top conflict |
|---|---|---|---|---|---|
| p0072 ↔ p0161 | False | 100.0 | 0.81 | family | routine_structure |
| p0160 ↔ p0252 | False | 93.9 | 0.74 | ambition | routine_structure |
| p0140 ↔ p0216 | False | 93.5 | 0.81 | family | conflict_approach |
| p0091 ↔ p0219 | False | 92.4 | 0.76 | ambition | routine_structure |
| p0106 ↔ p0169 | False | 91.4 | 0.74 | family | conflict_approach |
| p0136 ↔ p0157 | False | 90.0 | 1.00 | family | goal |
| p0131 ↔ p0138 | False | 89.5 | 1.00 | family | routine_structure |
| p0041 ↔ p0061 | False | 88.3 | 0.65 | independence | frequency |
| p0069 ↔ p0270 | True | – | 0.58 | independence | goal |

## 11. Limitations

* Synthetic outcomes are generated by a documented simulator; outperforming a baseline here validates the pipeline, not the theory of attraction.
* Reported interest is a self-report proxy; it is not attraction itself.
* Predictions are not psychological assessments or diagnoses.
* Few positives make PR-AUC and calibration bins noisy; intervals are reported.
* Exposure is affinity-weighted in the simulator; a real platform's exposure policy would shift prevalence and ranking metrics.
* Synthetic profiles are not a substitute for participant diversity.

## 12. Reproduction

```
python scripts/run_benchmark.py \
  --config configs/experiments/h1_structured_vs_baselines.yaml
python scripts/make_report.py
```

Every headline number above is regenerated from the checked-in configuration and seed. Notebooks are not part of this pipeline.
