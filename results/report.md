# h1_structured_vs_baselines — experiment report

Generated 2026-10-18T12:47:54+00:00 from `results/results.json`.

| field | value |
|---|---|
| config hash | 082951ce029c7bab |
| random seed | 20261018 |
| code revision | unknown |
| feature schema | 1.0.0 |
| simulation version | 1.0.0 |
| dataset version | synthetic-1.0.0-n100-seed20261018 |
| python | 3.14.7 |

## 1. What this report does and does not show

**Synthetic data only.** Outcomes were produced by the simulator documented in `backend/experiments/simulation.py` (v1.0.0). Beating a baseline here shows the pipeline, splits and metrics work. It is not evidence that the model predicts human attraction.

Reported interest is a self-report proxy, not attraction itself. Scores are on an experimental scale and are not probabilities of a successful relationship, nor psychological assessments.

## 2. Dataset

| quantity | value |
|---|---|
| profiles | 100 |
| candidate pairs generated | 1484 |
| directional interest observations | 1800 |
| mutual-interest observations | 316 |
| interaction-quality observations | 45 |
| unknown interest directions | 1168 |
| test positives (directional) | 83 |
| test positives (mutual) | 8 |

Unshown or unanswered directions are recorded as **unknown**, never as negative labels.

## 3. Protocol and leakage checks

* Primary protocol: participant_grouped (unseen people), test fraction 0.3, seed 20261018.
* Train pairs 729, test pairs 157, dropped 598 pairs that straddled the split.
* Shared participants across the split: 0; shared pairs: 0.
* Temporal protocol: train on first 70% of pairs by timestamp; 100 participants reappear by design.

## 4. Primary comparison (mutual interest)

| model | PR-AUC | 95% CI | ROC-AUC | log loss | ECE | NDCG@5 |
|---|---|---|---|---|---|---|
| random | 0.186 | [0.114, 0.361] | 0.312 | – | – | 0.345 |
| structured | 0.293 | [0.144, 0.554] | 0.507 | – | – | 0.371 |
| text | 0.229 | [0.131, 0.462] | 0.454 | – | – | 0.338 |
| structured_text | 0.306 | [0.142, 0.599] | 0.538 | – | – | 0.385 |
| mutual_independence | 0.394 | [0.165, 0.776] | 0.562 | 0.566 | 0.082 | 0.362 |
| mutual_logistic | 0.483 | [0.172, 0.826] | 0.587 | 0.563 | 0.146 | 0.346 |
| mutual_gbm | 0.325 | [0.139, 0.636] | 0.452 | 0.713 | 0.193 | 0.358 |

> With 8 positive mutual outcomes in the held-out split, the 95% interval for the best model (mutual_logistic) spans 0.65 PR-AUC. Differences smaller than that width are indistinguishable in this sample.

Mutual interest and interaction quality are *pair-level* outcomes: each pair contributes one held-out row, not one per direction. Directional interest legitimately contributes two rows because the two directions have different labels.

Primary metric **pr_auc**: best `mutual_logistic` = 0.483; Δ vs random +0.297; Δ vs structured +0.189.

Intervals overlap heavily at this sample size. Ordering is suggestive, not established.

### 4.1 Seed stability

The same configuration was rerun under 5 seeds (20261018, 20261019, 20261020, 20261021, 20261022). A model that only wins on one split is not a result.

**mutual PR-AUC across seeds**

| model | mean | sd | min | max | seeds won |
|---|---|---|---|---|---|
| mutual_gbm | 0.367 | 0.141 | 0.143 | 0.487 | 1 |
| mutual_independence | 0.356 | 0.159 | 0.125 | 0.566 | 1 |
| mutual_logistic | 0.350 | 0.177 | 0.077 | 0.526 | 1 |
| structured_text | 0.290 | 0.174 | 0.050 | 0.515 | 1 |
| structured | 0.288 | 0.173 | 0.048 | 0.503 | 0 |
| text | 0.256 | 0.045 | 0.200 | 0.320 | 1 |
| random | 0.174 | 0.078 | 0.048 | 0.241 | 0 |

**directional PR-AUC across seeds**

| model | mean | sd | min | max | seeds won |
|---|---|---|---|---|---|
| structured_text | 0.448 | 0.066 | 0.334 | 0.488 | 2 |
| structured | 0.448 | 0.069 | 0.330 | 0.490 | 1 |
| directional_logistic | 0.439 | 0.092 | 0.277 | 0.500 | 2 |
| text | 0.416 | 0.084 | 0.282 | 0.485 | 0 |
| random | 0.392 | 0.067 | 0.322 | 0.493 | 0 |

**The single-seed winner changed across seeds (mutual_gbm, mutual_independence, mutual_logistic, structured_text, text).** Differences between models are smaller than seed-to-seed variance, so this sample cannot rank them. Increasing participants is the only fix; more synthetic pairs would not be evidence either.

## 5. Directional interest (A → B)

| model | PR-AUC | 95% CI | ROC-AUC | NDCG@5 | coverage |
|---|---|---|---|---|---|
| random | 0.411 | [0.331, 0.527] | 0.487 | 0.604 | 1.000 |
| structured | 0.490 | [0.404, 0.594] | 0.528 | 0.630 | 0.885 |
| text | 0.485 | [0.419, 0.589] | 0.592 | 0.588 | 0.696 |
| structured_text | 0.482 | [0.398, 0.588] | 0.525 | 0.618 | 0.885 |
| directional_logistic | 0.498 | [0.422, 0.603] | 0.575 | 0.659 | 1.000 |

## 6. Ablation: does text add anything? (H2)

* structured only: 0.293
* structured + lexical embedding: 0.306
* difference: +0.013 (same test pairs)

The embedding here is a deterministic lexical hashing encoder, not a semantic model. This ablation tests whether shared wording helps; it does not test semantic compatibility.

## 7. Calibration

Calibration blocks are reported per probabilistic model in `results.json` (`metrics.calibration.bins`). A calibrated probability means: of pairs scored p, about p were positive in this sample. With few positives the bins are noisy.

The interpretable score additionally carries a **weight-sensitivity band**: the same pair rescored under ±25% log-normal perturbation of the pre-registered weights (200 draws, deterministic per pair). It shows how much the number depends on the weight choice. It is not a confidence interval and says nothing about sampling error.

## 8. Temporal protocol (known people, later observations)

| task | model | PR-AUC | ROC-AUC |
|---|---|---|---|
| directional | random | 0.383 | 0.476 |
| directional | structured | 0.409 | 0.560 |
| directional | text | 0.397 | 0.539 |
| directional | directional_logistic | 0.441 | 0.597 |
| mutual | random | 0.195 | 0.368 |
| mutual | structured | 0.233 | 0.466 |
| mutual | text | 0.262 | 0.539 |
| mutual | mutual_independence | 0.276 | 0.588 |
| mutual | mutual_logistic | 0.321 | 0.649 |

## 9. Interaction quality (Task C)

| model | MAE | RMSE | n |
|---|---|---|---|
| interaction_mean | 0.727 | 0.776 | 7 |
| interaction_ridge | 0.727 | 0.776 | 7 |

Quality labels exist only where both sides consented and both reported interest, so this comparison is underpowered in the synthetic run.

## 10. Failure cases

Scored by `structured` so each case can be explained feature by feature; the run's best learned model may be a different one.

| pair | observed mutual | score | coverage | top positive | top conflict |
|---|---|---|---|---|---|
| p0082 ↔ p0092 | False | 93.2 | 0.69 | family | frequency |
| p0092 ↔ p0100 | False | 90.0 | 0.89 | independence | routine_structure |
| p0016 ↔ p0088 | False | 88.5 | 0.68 | family | routine_structure |
| p0041 ↔ p0088 | False | 87.7 | 0.68 | family | routine_structure |
| p0020 ↔ p0051 | False | 85.9 | 0.84 | ambition | directness |
| p0068 ↔ p0070 | False | 85.3 | 1.00 | ambition | conflict_approach |
| p0046 ↔ p0068 | False | 84.7 | 1.00 | ambition | conflict_approach |
| p0016 ↔ p0068 | False | 83.4 | 0.81 | family | routine_structure |
| p0014 ↔ p0060 | True | – | 0.55 | family | routine_structure |
| p0051 ↔ p0076 | True | – | 0.47 | family | goal |

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
