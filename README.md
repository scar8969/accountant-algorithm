# The Accountant Algorithm

[![ci](https://github.com/scar8969/accountant-algorithm/actions/workflows/ci.yml/badge.svg)](https://github.com/scar8969/accountant-algorithm/actions/workflows/ci.yml)
![tests](https://img.shields.io/badge/tests-90%20passing-2f6f4f?style=flat-square)
![python](https://img.shields.io/badge/python-3.11%2B-31506f?style=flat-square)
![license](https://img.shields.io/badge/license-MIT-31506f?style=flat-square)
![status](https://img.shields.io/badge/status-research%20prototype-6b4f17?style=flat-square)

**Can a structured compatibility model rank promising matches better than a random
baseline — and can it tell "we look alike on paper" apart from "we would actually get
along"?**

This repository is a research harness built to answer that question honestly. It is a
feature engine, a benchmark runner, an inspection dashboard and a consent-aware API. It is
**not** a dating app: no messaging, no swiping, no discovery feed, no photo scoring.

> [!IMPORTANT]
> Everything here runs on **synthetic data**. That proves the pipeline works. It does not
> prove anything about people. The score is a 0–100 experimental scale — not a probability
> of a relationship, and not a psychological assessment.

| | |
|---|---|
| **Question** | Do structured compatibility features rank mutually interested pairs above chance? |
| **Approach** | 7 baselines + learned models, compared on held-out participants with bootstrap intervals |
| **Answer so far** | The pipeline is ready; the evidence is not. At 300 synthetic profiles a learned pair model beats random by +0.16 PR-AUC, but intervals still overlap |
| **Size** | 5,875 lines of Python, 90 tests, 9 tables, 11 models, 0 JS in the UI |

**Jump to:** [Run it](#run-it) · [Read the results](#the-results) · [How it works](#how-it-works) ·
[Use the API](#using-the-api) · [What the score means](#what-the-score-means) ·
[Consent & privacy](#data-consent-and-deletion) · [Reference](#reference)

---

## Run it

Three commands, about two minutes on a laptop.

```bash
python -m pip install -e ".[dev]"       # install
python -m alembic upgrade head          # create the database
python scripts/run_benchmark.py \       # run the benchmark (~50 s)
  --config configs/experiments/h1_structured_vs_baselines.yaml
```

Then look at it:

```bash
python -m uvicorn backend.api.main:app --port 8775
# http://127.0.0.1:8775/         experiment report
# http://127.0.0.1:8775/compare  pair explorer
# http://127.0.0.1:8775/docs     API reference
```

Optional, if you want the fuller picture:

```bash
python scripts/sweep_seeds.py --seeds 5    # is the winner just a lucky split?
python scripts/make_report.py              # results/report.md
python scripts/seed_demo.py --n 100        # synthetic profiles for the API
python -m pytest -q                        # 90 tests, ~45 s
```

<details>
<summary>Larger replication (300 profiles, 7,909 pairs, ~2 minutes)</summary>

```bash
python scripts/run_benchmark.py --config configs/experiments/h1_power_300.yaml \
  --out results/power
python scripts/make_report.py --results results/power/results.json \
  --out results/power/report.md
```
</details>

---

## The results

Screenshots are from the actual run; every number is in
[`results/report.md`](results/report.md).

### 300-profile run — the one worth reading

88 held-out pairs, 20 positives. PR-AUC with 95% intervals from a bootstrap over
**participants** (not rows):

| model | PR-AUC | 95% interval |
|---|---|---|
| random ranking | 0.212 | 0.131 – 0.342 |
| structured baseline (hand-weighted) | 0.228 | 0.131 – 0.373 |
| text only (lexical) | 0.247 | 0.118 – 0.412 |
| structured + text | 0.232 | 0.132 – 0.398 |
| independence product `p_AB · p_BA` | 0.331 | 0.211 – 0.532 |
| **mutual-interest logistic** | **0.370** | 0.215 – 0.509 |
| mutual GBM | 0.293 | 0.179 – 0.489 |

**How to read this:** the learned pair model beats random by +0.16 and beats the
hand-weighted baseline by +0.14, but its interval overlaps the independence product's. The
ordering is plausible; it is not established. Directional interest (A → B) is steadier:
structured 0.484 vs random 0.389.

### 100-profile run — why the honest answer is "not yet"

Only 34 test pairs and 8 positives. The best model reaches 0.483, but its interval is
**0.65 wide** — wider than every gap between models. Rerunning the identical config on five
seeds produces **five different winners**:

| model | mean PR-AUC | sd | seeds won |
|---|---|---|---|
| mutual GBM | 0.367 | 0.141 | 1 |
| independence product | 0.356 | 0.159 | 1 |
| mutual logistic | 0.350 | 0.177 | 1 |
| structured + text | 0.290 | 0.174 | 1 |
| structured | 0.288 | 0.173 | 0 |
| text only | 0.256 | 0.045 | 1 |
| random | 0.174 | 0.078 | 0 |

At 40 profiles the **random baseline wins**. That is the point of the harness: it prints the
interval width and the seed spread next to every headline number, so a lucky split cannot
be mistaken for a finding. More synthetic pairs would not fix this — only consented
participants would.

### What the interface looks like

![Experiment report](docs/dashboard.png)

The report leads with the primary metric and its interval, then calibration, seed
stability, protocol checks and the pairs the model got most wrong.

![Primary result panel](docs/result.png)

![Seed stability panel](docs/stability.png)

The pair explorer decomposes one score into weighted feature contributions, hard
constraints, missing answers and a sensitivity band:

![Compatibility explorer](docs/explorer.png)

---

## How it works

One process, no queue, no vector database, no microservices.

![System map](docs/architecture.svg)

| Stage | What happens | Where |
|---|---|---|
| **Profile** | 1–5 self-reports, explicit preferences, optional text, hard deal-breakers | `backend/features/` |
| **Score** | Weighted similarity + preference fit; missing answers masked, never imputed; below 60% coverage the score is withheld | `backend/features/structured.py` |
| **Rank** | 11 registered models (B0 random → B4 learned pair models) behind one interface | `backend/algorithms/` |
| **Evaluate** | Participant-grouped and temporal splits, leakage assertions, PR-AUC/ROC-AUC/calibration/NDCG | `backend/experiments/` |
| **Report** | JSON, CSV, markdown and a zero-JS dashboard with inline SVG charts | `backend/api/` |

Three outcomes are modelled separately, never collapsed into one opaque number:

1. **Directional interest** — would A say yes to B?
2. **Mutual interest** — would *both* say yes? (Not the average of two directions; the
   product `p_AB · p_BA` is only a baseline.)
3. **Interaction quality** — among pairs who both said yes and talked, how was it?

<details>
<summary>Why "leakage" gets its own test suite</summary>

Pairs share participants, so a random row split leaks one person across train and test —
the model memorises individuals and offline scores look excellent. The harness splits by
participant and **drops** pairs that straddle the split, verifies the split inside every run,
and fails the run (not just the review) if participants or pairs overlap. A temporal
protocol covers the other question: does it improve for people already seen?
</details>

---

## Using the API

```bash
curl -s localhost:8775/api/v1/compatibility/score \
  -H "X-API-Key: $ACCOUNTANT_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"profile_a_id":"...","profile_b_id":"...","model_name":"structured_baseline"}'
```

```json
{
  "score": 61.4,
  "scale": "0-100",
  "sufficient_data": true,
  "coverage": 0.94,
  "evidence": "high",
  "uncertainty": {"method": "weight-perturbation", "p05": 58.1, "p50": 61.2, "p95": 64.0},
  "dimensions": {"values": 0.62, "communication": 0.71, "lifestyle": 0.48},
  "explanation": {
    "contributions": [{"feature": "goal", "points": 10.5}],
    "conflicts": [{"feature": "social_energy", "points": 2.1}],
    "missing": ["text"]
  },
  "limitations": ["Score is on an experimental 0-100 scale; it is not a probability ..."]
}
```

Endpoints: participants, consent, profiles (+ versions), score, rank, feedback, experiments,
runs, export, withdraw, health. Full list and schemas at `/docs`.

<details>
<summary>Auth, limits and other operational details</summary>

* `X-API-Key` for research operations (production refuses to start without
  `ACCOUNTANT_API_KEY`); `X-Participant-Token` for a participant's own data.
* Rate limits per IP: participants 200/min, scoring 240/min, feedback 120/min
  (`ACCOUNTANT_RL_*`; `0/60` disables).
* Input caps on text, interests, maps, feedback payloads and experiment configs.
* `/health` does a database round-trip; every request gets an `X-Request-ID` and a log line.
* `/compatibility/score` returns 409 for a profile stored under an older feature schema.
</details>

---

## What the score means

A score of 61 means 61 on a fixed 0–100 scale built from pre-registered weights. It does
**not** mean a 61% chance of a relationship.

* Every score ships with an **evidence level** (coverage and completeness) and a
  **weight-sensitivity band** — the same pair rescored under ±25% perturbation of the
  weights. That is sensitivity to the weights, not sampling uncertainty.
* **Hard constraints** (deal-breakers, `"hard": true` ranges) gate a direction entirely;
  they are not folded into the number.
* **Missing answers are masked**, never imputed. Too much missing data and the score is
  withheld rather than invented.

### Glossary

| term | plain meaning |
|---|---|
| **PR-AUC** | Of the pairs the model ranks highest, how many are actually positive? Random ranking scores the positive rate (here ~0.21); higher is better |
| **Prevalence** | How many positives exist at all — PR-AUC must be read against it, not against 0.5 |
| **Cluster bootstrap** | Resampling *people* instead of rows, so correlated pairs don't fake precision |
| **Interval width** | How much this sample can distinguish. Gaps smaller than the width are noise |
| **Calibration / ECE** | Do "70% confident" predictions come true ~70% of the time? |
| **NDCG@5** | Quality of the top-5 ordering, discounted by position |
| **Leakage** | The same person appearing on both sides of a split |

---

## Data, consent and deletion

The system is built to be defensible before it is ever pointed at a real person.

* **Consent is enforced, not stored.** No profile can be created without an active consent
  record matching the supported study version and purpose. Consent withdrawal deletes data.
* **Match responses never expose the other person's answers** — only aggregate dimensions,
  contributions, gates and evidence.
* **Export gives you your own data**, including your side of each pair outcome and never the
  other person's.
* **Withdrawal** scrubs profile answers, deletes pair outcomes, predictions and feedback,
  invalidates the token, and keeps only aggregate run metrics (no personal data).
* **Pseudonyms are not anonymisation.** Preferences plus interaction records can still
  identify someone; keep identity mapping separate. The schema stores no names, phone
  numbers or exact locations.

---

## Project facts

Measured on this repository, not estimated:

| quantity | value |
|---|---|
| Python | 45 files · 5,875 lines (4,438 application, 1,136 tests, rest scripts) |
| tests | 90 passing in ~45 s |
| models | 11 registered, 3 prediction tasks |
| database | 9 tables, Alembic migrations |
| dashboard | 3 templates + 353 lines of CSS, 0 JS, 0 CDN requests |
| 100-profile run | 1,484 pairs, 157 test pairs, 50 s end to end |
| 300-profile run | 7,909 pairs, 694 test pairs, 124 s end to end |
| mutual rows | 34 → 88 (one per pair, never one per direction) |
| interval width | 0.65 → 0.29 as the sample grows |

---

## Quality

| Check | Result |
|---|---|
| Tests | 90 passing in ~45 s — features, metrics, simulation, API, leakage, rendering |
| Leakage | 0 shared participants, 0 shared pairs, asserted inside every run |
| CI | tests + migrations on a fresh DB + benchmark smoke run on every push |
| Determinism | same config → same numbers; all seeds recorded with each run |
| Failure visibility | failed runs persist with status and error instead of vanishing |
| Config validation | unknown keys/models, bad splits and impossible sample sizes fail before running |

The full incident-style list of what was found and fixed is in [CHANGELOG.md](CHANGELOG.md).

---

## Reference

<details>
<summary><b>Feature schema 1.0.0 and scoring</b></summary>

| dimension | features | weight |
|---|---|---|
| values | ambition, family, independence | 0.42 |
| communication | directness, conflict approach, contact frequency | 0.26 |
| lifestyle | social energy, routine structure | 0.17 |
| relationship goals | category match table | 0.10 |
| text | optional embedding cosine | 0.05 |

Similarity-desirable features use `s_i = max(0, 1 − |x_A − x_B| / R_i)`; complementary
features use a preference-satisfaction function. Aggregation is a weighted mean over
*answered* features. The directional score blends symmetric compatibility with the viewer's
stated-preference fit, weighted by how many preferences they answered. The questionnaire is
an experimental inventory, not a validated psychometric instrument.
</details>

<details>
<summary><b>Models</b></summary>

| id | model | notes |
|---|---|---|
| B0 | `random` | seeded hash of the pair — sanity check |
| B1 | `structured` | pre-registered weights + hard gates |
| B1+text | `structured_text` | ablation: adds the embedding leaf |
| B2 | `text` | cosine over self-descriptions (lexical hashing, not semantics) |
| B3 | `directional_logistic`, `mutual_independence` | logistic on pair features; product of two directional probabilities |
| B4 | `mutual_logistic`, `mutual_gbm` | direct pair model; regularised tree model |
| C | `interaction_mean`, `interaction_ridge` | rating baselines |

Swap `HashingEmbedding` for a sentence encoder (anything with `.embed()`) and the same
ablation tests semantic text.
</details>

<details>
<summary><b>Evaluation protocol</b></summary>

Metrics: PR-AUC (primary, with prevalence), ROC-AUC, log loss, Brier, ECE + reliability
bins, precision/recall at a threshold, NDCG@5, Recall@5, MRR, coverage, goal diversity.
Models are compared on identical held-out pairs, with a cluster bootstrap over participants.
Every run emits a `power_note` that translates interval width into "differences smaller than
this are indistinguishable".
</details>

<details>
<summary><b>Synthetic data generator</b></summary>

Three latents (warmth, structure, drive) generate the 1–5 answers. Interest is a logit over
stated-preference fit, latent affinity, topical overlap, candidate completeness, viewer
choosiness, a shared pair-level shock (why mutual ≠ product of independent probabilities) and
noise; deal-breaker violations are suppressed. Exposure is affinity-weighted, so exposure
bias exists by construction, and labels exist only for exposed directions. The full model is
recorded in every `results.json`.
</details>

<details>
<summary><b>Deliberate deviations from the original blueprint</b></summary>

| Blueprint | Here | Why |
|---|---|---|
| Next.js dashboard | server-rendered Jinja2 + inline SVG | one process, no build step; the JSON API is unchanged, so a Next.js client can replace it |
| PostgreSQL | PostgreSQL *or* SQLite | zero-setup local runs; Docker Compose brings Postgres |
| Plotly | hand-rolled SVG | offline, no CDN, no JS |
| MLflow | run rows + results JSON with config hash, seed, code revision | enough provenance at this stage |

</details>

<details>
<summary><b>Definition of done (blueprint §11)</b></summary>

- [x] Synthetic profiles reproducible from a seed
- [x] At least three baselines on the same held-out split (seven)
- [x] Every prediction records model + feature-schema version
- [x] Directional feedback preserved; unknown labels never negative
- [x] Participant-grouped evaluation with a leakage suite that fails the run
- [x] Dashboard reports ranking metrics, uncertainty and baseline comparisons
- [x] Consent withdrawal and deletion workflows tested
- [x] Report separates synthetic results from real participant evidence
- [x] Config validation, migrations, rate limits, schema guards, seed sweep

Not done, on purpose: messaging, swiping, discovery, photo scoring, live recommendations,
and any claim that the score predicts relationship success.

</details>

<details>
<summary><b>Repository map</b></summary>

```
backend/features/     schema, scoring, hard constraints, embeddings
backend/algorithms/   baselines, learned models, registry
backend/experiments/  simulation, splits + leakage asserts, runner, report
backend/evaluation/   classification, ranking, calibration, cluster bootstrap
backend/api/          routes, schemas, app, charts, templates, static/
backend/core/         security, consent, rate limiting
backend/db/           models, session, Alembic migrations
tests/                unit/, integration/, leakage/
configs/              h1_structured_vs_baselines.yaml, h1_power_300.yaml
scripts/              run_benchmark, sweep_seeds, make_report, seed_demo
docs/                 screenshots + architecture map
```
</details>

---

## Roadmap

1. **Before any pilot** — consent-screen review, retention job, deletion checks on backups,
   proxy rate limits, and a power calculation for how many consented pairs H1 needs.
2. **Semantic embeddings** — swap the lexical encoder through the existing provider interface.
3. **Pilot** — usability, completion rate and whether explanations are understandable, not
   prediction accuracy.
4. **Confirmatory** — pre-register metric and split, freeze the test set, evaluate once.
5. **Only then** heavier models. A complex model that cannot beat `structured` on held-out
   people is not a result.

## License

MIT — see [LICENSE](LICENSE). Research prototype: not a dating product, not a diagnostic tool.
