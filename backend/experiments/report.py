"""Markdown report generator.

The report is deliberately blunt about what the numbers do and do not support:
synthetic results validate the pipeline; they are not evidence about attraction.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


def _f(value, digits: int = 3) -> str:
    if value is None:
        return "–"
    return f"{value:.{digits}f}"


def _signed(value, digits: int = 3) -> str:
    if value is None:
        return "–"
    return f"{value:+.{digits}f}"


def _table(rows: list[list[str]], header: list[str]) -> str:
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(out)


def _sweep_block(sweep: dict) -> list[str]:
    lines = ["### 4.1 Seed stability", ""]
    seeds = sweep.get("seeds", [])
    lines.append(f"The same configuration was rerun under {len(seeds)} seeds "
                 f"({', '.join(str(s) for s in seeds)}). A model that only wins on one "
                 "split is not a result.")
    lines.append("")
    for task in ("mutual", "directional"):
        rows = []
        for row in sweep.get(task, []):
            rows.append([row["model"], _f(row["mean"]), _f(row["sd"]),
                         _f(row["min"]), _f(row["max"]), row.get("wins", 0)])
        if rows:
            lines.append(f"**{task} PR-AUC across seeds**")
            lines.append("")
            lines.append(_table(rows, ["model", "mean", "sd", "min", "max",
                                       "seeds won"]))
            lines.append("")
    winners = {row.get("best_mutual") for row in sweep.get("per_seed", [])}
    if len(winners) > 1:
        lines.append(f"**The single-seed winner changed across seeds "
                     f"({', '.join(sorted(str(w) for w in winners))}).** Differences "
                     "between models are smaller than seed-to-seed variance, so this "
                     "sample cannot rank them. Increasing participants is the only "
                     "fix; more synthetic pairs would not be evidence either.")
        lines.append("")
    return lines


def build_report(results: dict, sweep: dict | None = None) -> str:
    exp = results["experiment"]
    data = results["dataset"]
    protocol = results["protocol"]
    leakage = protocol["leakage"]
    comparison = results.get("comparison", {})
    temporal = protocol.get("temporal", {})

    lines: list[str] = []
    lines.append(f"# {exp['name']} — experiment report")
    lines.append("")
    lines.append(f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} "
                 f"from `results/results.json`.")
    lines.append("")
    lines.append(_table([
        ["config hash", exp["config_hash"]],
        ["random seed", str(exp["seed"])],
        ["code revision", exp["code_revision"]],
        ["feature schema", exp["feature_schema_version"]],
        ["simulation version", exp["simulation_version"]],
        ["dataset version", data["version"]],
        ["python", exp["python"]],
    ], ["field", "value"]))
    lines.append("")
    lines.append("## 1. What this report does and does not show")
    lines.append("")
    lines.append("**Synthetic data only.** Outcomes were produced by the simulator "
                 f"documented in `backend/experiments/simulation.py` "
                 f"(v{exp['simulation_version']}). Beating a baseline here shows the "
                 "pipeline, splits and metrics work. It is not evidence that the model "
                 "predicts human attraction.")
    lines.append("")
    lines.append("Reported interest is a self-report proxy, not attraction itself. "
                 "Scores are on an experimental scale and are not probabilities of a "
                 "successful relationship, nor psychological assessments.")
    lines.append("")
    lines.append("## 2. Dataset")
    lines.append("")
    lines.append(_table([
        ["profiles", data["n_profiles"]],
        ["candidate pairs generated", data["n_pairs"]],
        ["directional interest observations", data["n_directional_observations"]],
        ["mutual-interest observations", data["n_mutual_observations"]],
        ["interaction-quality observations", data["n_interaction_observations"]],
        ["unknown interest directions", data["unknown_interest_directions"]],
        ["test positives (directional)", data["test_positives"]["directional"]],
        ["test positives (mutual)", data["test_positives"]["mutual"]],
    ], ["quantity", "value"]))
    lines.append("")
    lines.append("Unshown or unanswered directions are recorded as **unknown**, never as "
                 "negative labels.")
    lines.append("")
    lines.append("## 3. Protocol and leakage checks")
    lines.append("")
    lines.append(f"* Primary protocol: {protocol['primary']}, test fraction "
                 f"{protocol['test_frac']}, seed {protocol['seed']}.")
    lines.append(f"* Train pairs {leakage['train_pairs']}, test pairs "
                 f"{leakage['test_pairs']}, dropped {protocol['dropped_straddling_pairs']} "
                 "pairs that straddled the split.")
    lines.append(f"* Shared participants across the split: "
                 f"{len(leakage['shared_participants'])}; shared pairs: "
                 f"{leakage['shared_pairs']}.")
    lines.append(f"* Temporal protocol: train on first "
                 f"{temporal.get('train_frac', 0) * 100:.0f}% of pairs by timestamp; "
                 f"{temporal.get('leakage', {}).get('shared_participants', []) and len(temporal['leakage']['shared_participants']) or 0} "
                 "participants reappear by design.")
    lines.append("")
    lines.append("## 4. Primary comparison (mutual interest)")
    lines.append("")
    comp = comparison.get("mutual", {})
    rows = []
    for task_block, task_name in ((results["results"].get("mutual", {}), "mutual"),):
        for name, payload in task_block.items():
            m = payload.get("metrics", {})
            ci = m.get("ci") or {}
            rows.append([name,
                         _f(m.get("pr_auc")),
                         (f"[{_f(ci.get('lo'))}, {_f(ci.get('hi'))}]"
                          if ci.get("lo") is not None else "–"),
                         _f(m.get("roc_auc")),
                         _f(m.get("log_loss")),
                         _f(m.get("ece")),
                         _f(payload.get("ranking", {}).get("ndcg@5"))])
    lines.append(_table(rows, ["model", "PR-AUC", "95% CI", "ROC-AUC", "log loss",
                               "ECE", "NDCG@5"]))
    lines.append("")
    if comp.get("best_model"):
        lines.append(f"Primary metric **{comp['primary_metric']}**: best "
                     f"`{comp['best_model']}` = {_f(comp['best_value'])}; "
                     f"Δ vs random {_signed(comp.get('delta_vs_random'))}; "
                     f"Δ vs structured {_signed(comp.get('delta_vs_structured'))}.")
        lines.append("")
        lines.append("Intervals overlap heavily at this sample size. Ordering is "
                     "suggestive, not established.")
        lines.append("")
    if sweep:
        lines.extend(_sweep_block(sweep))
    lines.append("## 5. Directional interest (A → B)")
    lines.append("")
    rows = []
    for name, payload in results["results"].get("directional", {}).items():
        m = payload.get("metrics", {})
        ci = m.get("ci") or {}
        rows.append([name, _f(m.get("pr_auc")),
                     (f"[{_f(ci.get('lo'))}, {_f(ci.get('hi'))}]"
                      if ci.get("lo") is not None else "–"),
                     _f(m.get("roc_auc")),
                     _f(payload.get("ranking", {}).get("ndcg@5")),
                     _f(payload.get("ranking", {}).get("coverage"))])
    lines.append(_table(rows, ["model", "PR-AUC", "95% CI", "ROC-AUC", "NDCG@5",
                               "coverage"]))
    lines.append("")
    lines.append("## 6. Ablation: does text add anything? (H2)")
    lines.append("")
    mutual = results["results"].get("mutual", {})
    base = mutual.get("structured", {}).get("metrics", {}).get("pr_auc")
    plus = mutual.get("structured_text", {}).get("metrics", {}).get("pr_auc")
    lines.append(f"* structured only: {_f(base)}")
    lines.append(f"* structured + lexical embedding: {_f(plus)}")
    if base is not None and plus is not None:
        lines.append(f"* difference: {_signed(plus - base)} (same test pairs)")
    lines.append("")
    lines.append("The embedding here is a deterministic lexical hashing encoder, not a "
                 "semantic model. This ablation tests whether shared wording helps; it "
                 "does not test semantic compatibility.")
    lines.append("")
    lines.append("## 7. Calibration")
    lines.append("")
    lines.append("Calibration blocks are reported per probabilistic model in "
                 "`results.json` (`metrics.calibration.bins`). A calibrated probability "
                 "means: of pairs scored p, about p were positive in this sample. "
                 "With few positives the bins are noisy.")
    lines.append("")
    lines.append("## 8. Temporal protocol (known people, later observations)")
    lines.append("")
    rows = []
    for task in ("directional", "mutual"):
        for name, payload in temporal.get("results", {}).get(task, {}).items():
            m = payload.get("metrics", {})
            rows.append([task, name, _f(m.get("pr_auc")), _f(m.get("roc_auc"))])
    lines.append(_table(rows, ["task", "model", "PR-AUC", "ROC-AUC"]))
    lines.append("")
    lines.append("## 9. Interaction quality (Task C)")
    lines.append("")
    rows = []
    for name, payload in results["results"].get("interaction", {}).items():
        m = payload.get("metrics", {})
        rows.append([name, _f(m.get("mae")), _f(m.get("rmse")), m.get("n")])
    lines.append(_table(rows, ["model", "MAE", "RMSE", "n"]))
    lines.append("")
    lines.append("Quality labels exist only where both sides consented and both "
                 "reported interest, so this comparison is underpowered in the "
                 "synthetic run.")
    lines.append("")
    lines.append("## 10. Failure cases")
    lines.append("")
    cases = results.get("failure_cases", [])
    if cases:
        rows = []
        for case in cases[:10]:
            pos = case["top_positive"][0]["feature"] if case["top_positive"] else "–"
            con = case["top_conflict"][0]["feature"] if case["top_conflict"] else "–"
            rows.append([f"{case['pair'][0]} ↔ {case['pair'][1]}",
                         case["observed_mutual"], _f(case["structured_score"], 1),
                         _f(case["coverage"], 2), pos, con])
        lines.append(_table(rows, ["pair", "observed mutual", "score", "coverage",
                                   "top positive", "top conflict"]))
    else:
        lines.append("No failure cases recorded.")
    lines.append("")
    lines.append("## 11. Limitations")
    lines.append("")
    for item in results.get("limitations", []):
        lines.append(f"* {item}")
    lines.append("* Exposure is affinity-weighted in the simulator; a real platform's "
                 "exposure policy would shift prevalence and ranking metrics.")
    lines.append("* Synthetic profiles are not a substitute for participant diversity.")
    lines.append("")
    lines.append("## 12. Reproduction")
    lines.append("")
    lines.append("```")
    lines.append("python scripts/run_benchmark.py \\")
    lines.append("  --config configs/experiments/h1_structured_vs_baselines.yaml")
    lines.append("python scripts/make_report.py")
    lines.append("```")
    lines.append("")
    lines.append("Every headline number above is regenerated from the checked-in "
                 "configuration and seed. Notebooks are not part of this pipeline.")
    lines.append("")
    return "\n".join(lines)


def write_report(results: dict, path: str | Path, sweep: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_report(results, sweep), encoding="utf-8")
    return path
