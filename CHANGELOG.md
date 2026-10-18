# Changelog

## 0.2.0 — robustness pass

Audit of the first release found a statistical defect, two state-handling bugs and a
set of missing guards. Everything below is covered by a test.

### Fixed

* **Pair-level outcomes were double counted.** Mutual-interest and interaction rows
  were built from both orderings of every pair, so a 34-pair test set reported n=68.
  The apparent variance was halved, the positive count doubled and the cluster
  bootstrap resampled the same pair twice. Mutual/interaction tasks now use one row
  per pair (`pair_rows`), and the supervised mutual and interaction models dedupe their
  training labels too. Reported mutual PR-AUCs changed as a result — the README and
  reports carry the corrected numbers.
* **Experiment seeds were ignored.** `POST /experiments` stored `random_seed`, but the
  merged default config always supplied its own seed, so every API run silently used
  the default. The experiment seed now wins unless the protocol or request overrides it.
* **Failed runs vanished.** A run that raised was marked failed and then rolled back by
  the request session, leaving no trace. Failures are committed before the error is
  returned, and `GET /experiments/{id}/runs` lists them.
* **Feedback idempotency had a race and a NULL hole.** The old unique index
  `(participant, prediction_id, type)` does not dedupe rows where `prediction_id` is
  NULL, which is the common case. Added a `dedupe_key` column with a unique
  `(participant, dedupe_key)` index and a caught `IntegrityError` path.
* **Scoring ignored the stored feature schema.** A profile saved under an older schema
  was scored with current rules. `/compatibility/score` now returns 409 and ranking
  skips those candidates.
* **Export was incomplete.** It returned profiles and feedback but not pair outcomes.
  It now returns the participant's own side of each pair, never the other person's.
* **Dashboard crashed on a partial results file.** `comparison` was assumed to contain
  every task; missing tasks now default to empty blocks.
* **CI whiskers could draw outside the plot area** when an interval exceeded the axis
  maximum; they are clamped.
* **Failure cases were mislabelled.** They were described as coming from the best
  model but were computed from the structured score. They now say `scored_by:
  structured`, and the dashboard explains why.

### Hardened

* **Config validation** (`validate_config`): unknown top-level keys, unknown model
  names, oversized/undersized samples, invalid split fractions and bootstrap settings
  fail before a run starts. `load_config` validates on load.
* **Split guards**: `grouped_pair_split` refuses fewer than 8 participants or 12 pairs
  and refuses to emit an empty side; `assert_no_leakage` and `assert_temporal_order`
  run inside `run_experiment` and raise instead of reporting.
* **Rate limiting**: in-process sliding window per client IP on participant creation,
  scoring and feedback, configurable via `ACCOUNTANT_RL_*` (`0/60` disables).
* **Input caps**: interests (50 × 60 chars), trait/preference maps (20 entries),
  feedback payload (4 KB), experiment/run configs (64 KB), experiment name and
  hypothesis lengths.
* **Consent validation**: unknown study version (409) or purpose (400) is rejected
  instead of being stored as an unusable consent record.
* **Auth ordering**: research endpoints authenticate as a dependency, so an
  unauthorised caller gets 401 before body validation.
* **Health** now performs a database round-trip and reports participant count;
  **request logging** adds `X-Request-ID` plus method/path/status/duration.
* **Alembic migrations** replace "create_all and hope": baseline revision, env that
  reads `ACCOUNTANT_DB_URL`, `make migrate`, `make db-reset`, CI check on a fresh DB.
* **Run provenance**: `config_hash` and `seed` are stored with each run and returned by
  `GET /runs/{id}`.

### Added (research quality)

* **Hard range constraints**: `"hard": true` on a preference becomes a directional
  gate (`ambition:hard_range`), alongside named deal-breakers.
* **Preference shrinkage**: the directional score's preference half is weighted by how
  many preferences the viewer answered, so a single stated range cannot dominate.
* **Disclosed uncertainty**: every score carries an evidence level and a deterministic
  weight-sensitivity band (±25% log-normal weight perturbation, 200 draws), explicitly
  labelled as sensitivity to the weights — not a confidence interval.
* **Precision statement**: every run emits `power_note` translating the best model's
  interval width into the difference the sample can actually resolve.
* **Larger replication config** `h1_power_300.yaml` (300 profiles): interval width for
  mutual PR-AUC drops from 0.65 to 0.29, and the model ordering becomes readable.
* **90 tests** (was 63), including a dedicated `tests/unit/test_robustness.py`, new
  integration tests for consent versions, rate limits, schema guards, export contents,
  seed handling and failed-run persistence, and render tests for the report, explorer
  and the no-artifact setup page.
* **GitHub Actions CI**: tests, migrations on a fresh database, benchmark smoke run.

### Changed numbers

The mutual-interest metrics in the README/report were regenerated after the
double-counting fix: the 100-profile test set is 34 pairs (8 positives), so its
intervals are wider and no model can be ranked from it. The 300-profile replication is
now the primary evidence about the pipeline's behaviour.
