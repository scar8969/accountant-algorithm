#!/usr/bin/env python
"""Seed-variance sweep: is the model ordering just one lucky split?

Runs the same configuration under several random seeds and reports mean ± sd of
the primary metric per model. A model that only 'wins' on one seed is not a
result.

    python scripts/sweep_seeds.py --seeds 5
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.experiments.runner import load_config, run_experiment  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(
        ROOT / "configs" / "experiments" / "h1_structured_vs_baselines.yaml"))
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--start", type=int, default=None)
    ap.add_argument("--out", default=str(ROOT / "results" / "sweep.json"))
    ap.add_argument("--profiles", type=int, default=None)
    args = ap.parse_args()

    base = load_config(args.config)
    start = base["seed"] if args.start is None else args.start
    seeds = [start + i for i in range(args.seeds)]
    if args.profiles:
        base["synthetic"]["n_profiles"] = args.profiles
    base["protocol"]["bootstrap_iters"] = min(base["protocol"]["bootstrap_iters"], 200)

    per_seed: list[dict] = []
    for seed in seeds:
        cfg = json.loads(json.dumps(base))
        cfg["seed"] = seed
        results = run_experiment(cfg, write_artifacts=False)
        per_seed.append({
            "seed": seed,
            "mutual": {name: payload["metrics"].get("pr_auc")
                       for name, payload in results["results"].get("mutual", {}).items()},
            "directional": {name: payload["metrics"].get("pr_auc")
                            for name, payload in results["results"].get(
                                "directional", {}).items()},
            "best_mutual": results["comparison"]["mutual"].get("best_model"),
            "best_directional": results["comparison"]["directional"].get("best_model"),
            "mutual_positives": results["dataset"]["n_mutual_observations"],
        })
        print(f"seed {seed}: best mutual = {per_seed[-1]['best_mutual']} "
              f"(mutual obs {per_seed[-1]['mutual_positives']})")

    def summarise(task: str) -> list[dict]:
        models = sorted({m for row in per_seed for m in row[task] if row[task][m] is not None})
        out = []
        for model in models:
            values = [row[task][model] for row in per_seed if row[task].get(model) is not None]
            out.append({"model": model, "n_seeds": len(values),
                        "mean": round(statistics.fmean(values), 4),
                        "sd": round(statistics.stdev(values), 4) if len(values) > 1 else 0.0,
                        "min": round(min(values), 4), "max": round(max(values), 4),
                        "wins": sum(1 for row in per_seed
                                    if row[f"best_{task}"] == model)})
        out.sort(key=lambda row: -row["mean"])
        return out

    summary = {"seeds": seeds, "per_seed": per_seed,
               "mutual": summarise("mutual"),
               "directional": summarise("directional")}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    for task in ("mutual", "directional"):
        print()
        print(f"{task}: PR-AUC across {len(seeds)} seeds")
        print(f"  {'model':<24}{'mean':>8}{'sd':>8}{'min':>8}{'max':>8}{'best':>6}")
        for row in summary[task]:
            print(f"  {row['model']:<24}{row['mean']:>8.3f}{row['sd']:>8.3f}"
                  f"{row['min']:>8.3f}{row['max']:>8.3f}{row['wins']:>6}")
    print()
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
