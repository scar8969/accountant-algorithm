#!/usr/bin/env python
"""Run a checked-in experiment configuration end to end.

    python scripts/run_benchmark.py --config configs/experiments/h1_structured_vs_baselines.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.experiments.runner import load_config, run_experiment  # noqa: E402


def fmt(x, width=7):
    if x is None:
        return "-".rjust(width)
    if isinstance(x, float):
        return f"{x:.3f}".rjust(width)
    return str(x).rjust(width)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(
        ROOT / "configs" / "experiments" / "h1_structured_vs_baselines.yaml"))
    ap.add_argument("--out", default=None, help="results directory")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--profiles", type=int, default=None)
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.profiles is not None:
        cfg["synthetic"]["n_profiles"] = args.profiles

    results = run_experiment(cfg, out_dir=args.out, write_artifacts=not args.no_write)

    exp = results["experiment"]
    data = results["dataset"]
    leak = results["protocol"]["leakage"]

    print()
    print(f"experiment   {exp['name']}  (config {exp['config_hash']}, seed {exp['seed']})")
    print(f"dataset      {data['version']}")
    print(f"             profiles={data['n_profiles']} pairs={data['n_pairs']} "
          f"directional_obs={data['n_directional_observations']} "
          f"mutual_obs={data['n_mutual_observations']} "
          f"unknown_directions={data['unknown_interest_directions']}")
    print(f"split        grouped by participant  train_pairs={leak['train_pairs']} "
          f"test_pairs={leak['test_pairs']} dropped={results['protocol']['dropped_straddling_pairs']}")
    print(f"leakage      shared_participants={len(leak['shared_participants'])} "
          f"shared_pairs={leak['shared_pairs']}  -> "
          f"{'CLEAN' if leak['disjoint_participants'] and leak['disjoint_pairs'] else 'LEAK'}")

    for task in ("directional", "mutual"):
        block = results["results"].get(task, {})
        if not block:
            continue
        print()
        print(f"{task} prediction (held-out participants, n="
              f"{block[next(iter(block))]['metrics']['n']})")
        print(f"  {'model':<24}{'pr_auc':>8}{'95% CI':>18}{'roc_auc':>9}"
              f"{'ndcg@5':>8}{'recall@5':>10}{'coverage':>10}")
        for name, payload in block.items():
            m, r = payload["metrics"], payload.get("ranking", {})
            ci = m.get("ci") or {}
            interval = (f"[{fmt(ci.get('lo'), 5).strip()},{fmt(ci.get('hi'), 5).strip()}]"
                        if ci.get("lo") is not None else "-")
            print(f"  {name:<24}{fmt(m.get('pr_auc'), 8)}{interval:>18}"
                  f"{fmt(m.get('roc_auc'), 9)}{fmt(r.get('ndcg@5'), 8)}"
                  f"{fmt(r.get('recall@5'), 10)}{fmt(r.get('coverage'), 10)}")

    inter = results["results"].get("interaction")
    if inter:
        print()
        print("interaction quality (regression)")
        for name, payload in inter.items():
            m = payload["metrics"]
            print(f"  {name:<24} MAE={m.get('mae')} RMSE={m.get('rmse')} n={m.get('n')}")

    comp = results["comparison"]["mutual"]
    if comp.get("best_model"):
        print()
        print(f"primary metric ({comp['primary_metric']}) for mutual interest: "
              f"best={comp['best_model']} {comp['best_value']} "
              f"delta_vs_random={comp['delta_vs_random']} "
              f"delta_vs_structured={comp['delta_vs_structured']}")

    print()
    print("REMINDER: synthetic outcomes validate the pipeline, not human attraction.")
    if not args.no_write:
        out = Path(args.out) if args.out else ROOT / "results"
        print(f"artifacts: {out / 'results.json'}  {out / 'results.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
