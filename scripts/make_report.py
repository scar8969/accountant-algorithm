#!/usr/bin/env python
"""Render results/results.json as a markdown report.

    python scripts/make_report.py [--results results/results.json] [--out results/report.md]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.experiments.report import write_report  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=str(ROOT / "results" / "results.json"))
    ap.add_argument("--out", default=str(ROOT / "results" / "report.md"))
    args = ap.parse_args()

    path = Path(args.results)
    if not path.exists():
        print(f"no results at {path}; run scripts/run_benchmark.py first")
        return 1
    results = json.loads(path.read_text(encoding="utf-8"))
    sweep_path = path.parent / "sweep.json"
    sweep = json.loads(sweep_path.read_text(encoding="utf-8")) if sweep_path.exists() else None
    out = write_report(results, args.out, sweep)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
