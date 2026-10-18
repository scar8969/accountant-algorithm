"""FastAPI application + server-rendered research dashboard.

The dashboard reads the latest benchmark artifact (results/results.json) and
renders it with inline SVG. It is intentionally not a Next.js app: one process
and zero build step keeps the MVP reproducible. The JSON API below is versioned
independently, so a richer frontend can be added later without touching it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from ..db.session import init_db
from ..experiments.runner import DEFAULT_CONFIG, load_config
from ..features.embeddings import get_default_provider
from ..features.structured import explanation, score_pair
from ..experiments.simulation import generate_profiles
from . import charts
from .routes import router

ROOT = Path(__file__).resolve().parents[2]
TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


def _results_path() -> Path:
    return Path(os.environ.get("ACCOUNTANT_RESULTS", ROOT / "results" / "results.json"))


def _load_sweep() -> dict | None:
    path = _results_path().parent / "sweep.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _load_results() -> dict | None:
    path = _results_path()
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _metric_rows(task_block: dict, best: str | None) -> list[dict]:
    rows = []
    for name, payload in task_block.items():
        m = payload.get("metrics", {})
        r = payload.get("ranking", {})
        ci = m.get("ci") or {}
        rows.append({
            "model": name,
            "pr_auc": m.get("pr_auc"),
            "ci": (f"{ci['lo']:.3f}–{ci['hi']:.3f}"
                   if ci.get("lo") is not None else "–"),
            "roc_auc": m.get("roc_auc"),
            "log_loss": m.get("log_loss"),
            "ece": m.get("ece"),
            "ndcg": r.get("ndcg@5"),
            "recall": r.get("recall@5"),
            "coverage": r.get("coverage"),
            "highlight": name == best,
        })
    rows.sort(key=lambda row: -(row["pr_auc"] if row["pr_auc"] is not None else -1))
    return rows


def _bar_items(task_block: dict, best: str | None) -> list[dict]:
    items = []
    for name, payload in task_block.items():
        m = payload.get("metrics", {})
        ci = m.get("ci") or {}
        items.append({"label": name, "value": m.get("pr_auc"),
                      "lo": ci.get("lo"), "hi": ci.get("hi"),
                      "highlight": name == best})
    return sorted(items, key=lambda i: -(i["value"] or 0))


def _calibration_source(task_block: dict) -> tuple[str, list[dict]] | tuple[None, None]:
    candidates = [(name, payload.get("metrics", {}).get("calibration"))
                  for name, payload in task_block.items()
                  if payload.get("metrics", {}).get("calibration")]
    if not candidates:
        return None, None
    preferred = [c for c in candidates if "logistic" in c[0]] or candidates
    name, block = preferred[0]
    return name, block.get("bins", [])


def build_view(results: dict) -> dict:
    exp = results["experiment"]
    data = results["dataset"]
    protocol = results["protocol"]
    mutual = results.get("results", {}).get("mutual", {})
    directional = results.get("results", {}).get("directional", {})
    comparison = results.get("comparison", {})

    best_mutual = comparison.get("mutual", {}).get("best_model")
    best_directional = comparison.get("directional", {}).get("best_model")

    calib_name, calib_bins = _calibration_source(mutual)

    temporal = protocol.get("temporal", {})
    temporal_comparison = {
        task: _metric_rows(temporal.get("results", {}).get(task, {}), None)
        for task in ("directional", "mutual")
    }

    return {
        "experiment": exp,
        "dataset": data,
        "protocol": protocol,
        "simulation": results.get("simulation", {}),
        "mutual_rows": _metric_rows(mutual, best_mutual),
        "directional_rows": _metric_rows(directional, best_directional),
        "mutual_chart": charts.bar_chart(_bar_items(mutual, best_mutual)),
        "directional_chart": charts.bar_chart(_bar_items(directional, best_directional)),
        "calibration_name": calib_name,
        "calibration_chart": (charts.calibration_chart(calib_bins)
                              if calib_bins else ""),
        "comparison": comparison,
        "failure_cases": results.get("failure_cases", []),
        "limitations": results.get("limitations", []),
        "temporal": {"comparison": temporal_comparison,
                     "dataset": temporal.get("dataset", {}),
                     "leakage": temporal.get("leakage", {}),
                     "train_frac": temporal.get("train_frac")},
        "interaction": results.get("results", {}).get("interaction", {}),
        "sweep": _load_sweep(),
        "sweep_winners": sorted({row.get("best_mutual")
                                 for row in (_load_sweep() or {}).get("per_seed", [])
                                 if row.get("best_mutual")}),
        "generated": exp.get("created_at", ""),
        "results_path": str(_results_path()),
    }


def _demo_profiles() -> tuple[list[dict], dict]:
    results = _load_results()
    cfg = DEFAULT_CONFIG
    if results:
        cfg = results["experiment"].get("config", DEFAULT_CONFIG)
    profiles = generate_profiles(cfg["synthetic"]["n_profiles"], seed=cfg["seed"],
                                 missing_rate=cfg["synthetic"].get("missing_rate", 0.1))
    return profiles, cfg


def create_app() -> FastAPI:
    init_db()
    app = FastAPI(title="Accountant Algorithm — research platform", version="0.1.0")
    app.include_router(router, prefix="/api/v1")

    @app.get("/", response_class=HTMLResponse)
    def dashboard(request: Request):
        results = _load_results()
        if results is None:
            return TEMPLATES.TemplateResponse(request, "empty.html", {
                "title": "Accountant Algorithm",
                "results_path": str(_results_path()),
            })
        return TEMPLATES.TemplateResponse(request, "dashboard.html",
                                          {"title": "Accountant Algorithm — results",
                                           "view": build_view(results)})

    @app.get("/compare", response_class=HTMLResponse)
    def compare(request: Request, a: str | None = Query(default=None),
                b: str | None = Query(default=None)):
        profiles, cfg = _demo_profiles()
        by_id = {p["participant_id"]: p for p in profiles}
        a = a or profiles[0]["participant_id"]
        b = b or profiles[1]["participant_id"]
        result = None
        if a in by_id and b in by_id and a != b:
            ps = score_pair(by_id[a], by_id[b])
            result = {"score": ps.score, "coverage": ps.coverage,
                      "sufficient": not ps.insufficient,
                      "dimensions": ps.dimensions,
                      "gate_a": ps.gate_a, "gate_b": ps.gate_b,
                      "limitations": ps.limitations,
                      "explanation": explanation(ps)}
        return TEMPLATES.TemplateResponse(request, "compare.html", {
            "title": "Compatibility explorer",
            "profiles": profiles, "a": a, "b": b, "result": result,
            "seed": cfg["seed"], "synthetic": True,
            "embed": get_default_provider().__class__.__name__,
        })

    return app


app = create_app()
