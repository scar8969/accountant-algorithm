# The Accountant Algorithm

A research platform for one question: **can a structured compatibility model rank
promising matches better than simple baselines — and can it separate profile
attraction from interpersonal compatibility?**

It is not a dating app. There is no messaging, no swiping, no discovery feed, no
photo scoring. It is a benchmark harness, a feature engine, an inspection UI and a
consent-aware API for running small, pre-registered experiments.

Two claims are deliberately kept apart:

* **Reported interest** and **observed interaction outcomes** are measurable.
* **Attraction** is not an objective universal quantity. Scores on the 0–100 scale
  are not probabilities of a relationship, and nothing here is a psychological
  assessment or diagnosis.

The first release is validated on synthetic data only. Synthetic results prove the
pipeline works. They are not evidence about people.

---

## Quickstart

```bash
cd accountant-algorithm
python -m pip install -e ".[dev]"          # fastapi, sqlalchemy, numpy, sklearn, pytest

# 1. run the benchmark (writes results/results.json + results.csv)
python scripts/run_benchmark.py --config configs/experiments/h1_structured_vs_baselines.yaml

# 2. seed-variance check (5 seeds; writes results/sweep.json)
python scripts/sweep_seeds.py --seeds 5

# 3. markdown report
python scripts/make_report.py              # results/report.md

# 4. dashboard + API
python -m uvicorn backend.api.main:app --port 8775
#    http://127.0.0.1:8775/           results dashboard
#    http://127.0.0.1:8775/compare    compatibility explorer
#    http://127.0.0.1:8775/docs       OpenAPI

# 5. tests
python -m pytest -q                        # 63 tests: unit, integration, leakage
```

Optional demo data for the API: `python scripts/seed_demo.py --n 100` creates
synthetic participants (marked `source="synthetic"`).

---

## What the first run actually shows

Run `h1_structured_vs_baselines`, 100 synthetic profiles, 1,484 exposed pairs,
participant-grouped split (729 train / 157 test pairs, 598 straddling pairs dropped,
0 shared participants).

Mutual interest, held-out participants, PR-AUC with cluster-bootstrap 95% CI:

| model | PR-AUC | 95% CI | NDCG@5 |
|---|---|---|---|
| random | 0.262 | 0.156–0.487 | 0.345 |
| structured (B1) | 0.293 | 0.184–0.502 | 0.371 |
| text only (B2) | 0.229 | 0.141–0.390 | 0.338 |
| structured + text | 0.306 | 0.189–0.519 | 0.385 |
| independence product | 0.295 | 0.191–0.477 | 0.386 |
| **mutual logistic (B3)** | **0.375** | 0.206–0.589 | 0.364 |
| mutual GBM | 0.242 | 0.148–0.455 | 0.314 |

Directional interest (A → B): random 0.411, structured 0.485, text 0.485,
structured+text 0.477, directional logistic 0.498.

**The single-seed winner is not reproducible.** Rerunning the identical config on
five seeds gives five different winners:

| model | mean PR-AUC | sd | seeds won |
|---|---|---|---|
| mutual GBM | 0.326 | 0.123 | 1 |
| mutual logistic | 0.290 | 0.149 | 2 |
| structured + text | 0.290 | 0.174 | 1 |
| structured | 0.288 | 0.173 | 0 |
| text only | 0.256 | 0.045 | 1 |
| independence product | 0.252 | 0.084 | 0 |
| random | 0.200 | 0.083 | 0 |

The gap between models is smaller than seed-to-seed variance, so this sample cannot
rank them. That is the honest result, and it is reported in `results/report.md`
and at the top of the dashboard instead of hiding behind the best seed. The
conclusion for the research plan: **the pipeline is ready; the evidence is not.**
Adding synthetic pairs would not fix it — only consented participants would.

---

## Architecture

Modular monolith, one process, no queue, no vector database, no microservices.

```
backend/
  features/     schema.py (weights, goals, deal-breakers), structured.py (scoring),
                embeddings.py (provider interface + lexical hashing encoder)
  algorithms/   base.py (TaskModel), baseline.py (B0/B1/B2), supervised.py (B3/B4),
                registry.py (one place experiment configs resolve model names)
  experiments/  simulation.py (latent-factor generator + outcome model),
                splits.py (grouped + temporal), runner.py, report.py
  evaluation/   classification.py, ranking.py, calibration.py, stats.py (cluster CI)
  api/          routes.py, schemas.py, main.py, charts.py, templates/
  core/         security.py (research key, participant tokens), consent.py
  db/           models.py (9 tables), session.py
tests/          unit/, integration/, leakage/
configs/        experiments/h1_structured_vs_baselines.yaml
scripts/        run_benchmark.py, sweep_seeds.py, make_report.py, seed_demo.py
```

### Deviations from the blueprint (deliberate, documented)

| Blueprint | Here | Why |
|---|---|---|
| Next.js dashboard | server-rendered Jinja2 + inline SVG | one process, no build step; the versioned JSON API is unchanged, so a Next.js client can replace it later without backend work |
| PostgreSQL | PostgreSQL *or* SQLite via `ACCOUNTANT_DB_URL` | the whole thing runs locally with zero setup; Docker Compose brings Postgres for deployment |
| Alembic migrations | `create_all()` for the MVP | the schema is young; migrations land before any pilot data exists (see Roadmap) |
| Plotly | hand-rolled SVG | works offline, no CDN, no JS dependency |
| MLflow | results JSON + run rows in Postgres | model configs, seeds and dataset versions are recorded per run; MLflow adds nothing at this stage |

---

## The three prediction tasks

They are never collapsed into one opaque score.

* **A — directional interest**: P(A is interested in B | A's stated preferences, B's
  profile). Scored on PR-AUC/ROC-AUC.
* **B — mutual interest**: P(both independently say yes). A pair-level outcome, not
  the average of two directional scores. The independence product
  `p_AB · p_BA` is one *baseline*; a direct pair model is another.
* **C — interaction quality**: conversation rating, only among pairs who both
  consented and both reported interest. MAE/RMSE.

## Feature inventory (schema 1.0.0)

1–5 self-reports, optional text, explicit deal-breakers. Not a validated
psychometric instrument; described as an experimental inventory everywhere.

| dimension | features | weight |
|---|---|---|
| values | ambition, family, independence | 0.42 |
| communication | directness, conflict approach, contact frequency (preference function) | 0.26 |
| lifestyle | social energy, routine structure | 0.17 |
| relationship goals | category match table | 0.10 |
| text | optional embedding cosine | 0.05 |

Similarity-desirable features use `s_i = max(0, 1 − |x_A − x_B| / R_i)` with
`R_i = 4`; complementary features (e.g. A's acceptable contact frequency vs B's
actual) use a preference-satisfaction function. Aggregation is the blueprint's
weighted mean over *answered* features. **Missing answers get mask 0 — never an
imputed value.** Below 60% weighted coverage the score is withheld
(`score: null`, `sufficient_data: false`) rather than invented. Deal-breakers are
hard directional gates applied before ranking.

The directional score is `0.5 · symmetric compatibility + 0.5 · preference fit`, so
"how similar you are" and "how well they match what you asked for" stay separable
and inspectable.

## Baselines and learned models

| id | model | notes |
|---|---|---|
| B0 | `random` | seeded hash of the pair — sanity check |
| B1 | `structured` | pre-registered weights + hard gates |
| B1+text | `structured_text` | ablation: adds the embedding leaf |
| B2 | `text` | cosine over optional self-descriptions |
| B3 | `directional_logistic`, `mutual_independence` | logistic on pair features; product of two directional probabilities |
| B4 | `mutual_logistic`, `mutual_gbm` | direct pair model; regularised tree model |
| C | `interaction_mean`, `interaction_ridge` | rating baselines |

The text encoder is a deterministic signed-hashing TF encoder: it captures **shared
wording, not semantics**. Swap the provider (any object with `.embed()`), bump its
version, and the H2 ablation tests a real semantic encoder.

## Evaluation, and the leakage trap

Matching data has a special failure mode: many pairs share a participant, so a
random row split puts the same person on both sides. Two protocols are implemented:

* **unseen people** (primary) — split participants, drop pairs that straddle the
  split. `tests/leakage` asserts zero shared participants and zero shared pairs.
* **known people / temporal** — train on earlier observations, test on later ones.

Also enforced by tests: features never read labels or post-interaction feedback
(changing a row's label does not change its feature vector or its score), outcomes
without exposure stay `null` and are never treated as negatives, and mutual labels
require both directions.

Metrics: PR-AUC (primary, with positive prevalence reported), ROC-AUC, log loss,
Brier, ECE + reliability bins, precision/recall at threshold, NDCG@5, Recall@5, MRR,
coverage and goal diversity. Uncertainty is a **cluster bootstrap over participants**,
not rows. Models are compared on identical held-out pairs.

## Consent, privacy and deletion

* `POST /api/v1/participants` issues a random token, stored only as a SHA-256 hash.
* A profile cannot be created without an active consent record — enforced in the
  application, not just stored (`tests/integration/test_api.py`).
* Match responses return aggregate dimensions and feature contributions, never the
  other person's answers, interests or text.
* Feedback is directional and idempotent; mutual interest is only "recorded" when
  both sides have answered.
* `POST /participants/{id}/withdraw` scrubs profiles, deletes pair outcomes,
  predictions and feedback, invalidates the token, and marks the participant
  deleted. Aggregate run metrics (no personal data) are retained for reproducibility,
  and the consent text says so.
* `POST /participants/{id}/export` returns that participant's own data as JSON.
* UUIDs are pseudonyms, not anonymisation: preferences plus interaction records can
  still identify someone. Keep identity mapping separate; the schema never stores
  names, phone numbers or exact locations.

API auth: `X-API-Key` for research operations (set `ACCOUNTANT_API_KEY`;
production refuses to start without it), `X-Participant-Token` for a participant's
own data. The dashboard is unauthenticated and must be bound to localhost.

## API v1

`POST /participants` · `POST /participants/{id}/consent` ·
`POST /participants/{id}/export` · `POST /participants/{id}/withdraw` ·
`POST /profiles` · `POST /profiles/{id}/versions` · `GET /profiles/{id}` ·
`POST /compatibility/score` · `POST /compatibility/rank` · `POST /feedback` ·
`POST /experiments` · `POST /experiments/{id}/runs` · `GET /runs/{id}` ·
`GET /runs/{id}/results` · `GET /models` · `GET /health`

```bash
curl -s localhost:8775/api/v1/compatibility/score \
  -H 'X-API-Key: dev-research-key' -H 'Content-Type: application/json' \
  -d '{"profile_a_id":"...","profile_b_id":"...","model_name":"structured_baseline"}'
```

```json
{
  "task": "compatibility_score",
  "score": 61.4,
  "scale": "0-100",
  "probability": null,
  "sufficient_data": true,
  "coverage": 0.94,
  "dimensions": {"values": 0.62, "communication": 0.71, "lifestyle": 0.48,
                 "relationship_goals": 0.8},
  "explanation": {"contributions": [{"feature": "goal", "points": 10.5}],
                  "conflicts": [{"feature": "social_energy", "points": 2.1}],
                  "missing": ["text"]},
  "limitations": ["Score is on an experimental 0-100 scale; it is not a probability ..."]
}
```

## Synthetic data generation (why the numbers are what they are)

Three latents (warmth, structure, drive) generate the 1–5 answers. Interest is a
logit over stated-preference fit, latent affinity, topical overlap, the candidate's
profile completeness, the viewer's choosiness, a **shared pair-level shock** (which
is why mutual interest is not the product of two independent probabilities) and
noise; deal-breaker violations are suppressed. Exposure is affinity-weighted, so
exposure bias exists in the data by construction. Interest labels exist only for
exposed directions. The full model is written into every `results.json` under
`simulation`, and into `results/report.md`.

## Roadmap

1. **Before any pilot**: Alembic migrations, rate limiting on participant creation,
   a real consent screen and retention job, and a power calculation for how many
   consented pairs the H1 comparison actually needs.
2. **Semantic embeddings**: swap `HashingEmbedding` for a sentence encoder through
   the existing provider interface; rerun the H2 ablation.
3. **Pilot (Phase B)**: usability, completion rate, whether explanations are
   understandable — not prediction accuracy.
4. **Confirmatory (Phase C)**: pre-register H1's metric and split, freeze the test
   set, then evaluate once. Do not tune against it.
5. **Only then**: heavier models. A complex model that cannot beat `structured` on
   held-out participants is not a result.

## Definition of done (blueprint §11)

- [x] Synthetic profiles reproducible from a seed (tested)
- [x] At least three baselines on the same held-out split (seven)
- [x] Every prediction records model + feature-schema version
- [x] Directional feedback preserved; unknown labels never negative
- [x] Participant-grouped evaluation with a leakage test suite
- [x] Dashboard reports ranking metrics, uncertainty and baseline comparisons
- [x] Consent withdrawal and deletion workflows tested
- [x] Report separates synthetic results from real participant evidence

Not done, on purpose: messaging, swiping, discovery, photo scoring, live
recommendations, and any claim that the score predicts relationship success.
