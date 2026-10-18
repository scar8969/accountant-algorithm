"""Uncertainty: cluster bootstrap so resampling happens over people, not rows."""

from __future__ import annotations

import numpy as np


def cluster_bootstrap(items: list, groups: list[str], metric_fn, *,
                      iters: int = 500, seed: int = 0, alpha: float = 0.05) -> dict:
    """Percentile bootstrap CI of `metric_fn` resampling whole clusters.

    Pairs sharing a participant are correlated; resampling rows independently
    would understate uncertainty. Clusters are participant ids.
    """
    if not items:
        return {"estimate": None, "lo": None, "hi": None, "n_valid": 0}
    rng = np.random.default_rng(seed)
    by_group: dict[str, list[int]] = {}
    for idx, g in enumerate(groups):
        by_group.setdefault(g, []).append(idx)
    keys = list(by_group)

    estimate = metric_fn(items)
    values: list[float] = []
    for _ in range(iters):
        pick = rng.choice(len(keys), size=len(keys), replace=True)
        sample: list = []
        for j in pick:
            sample.extend(items[i] for i in by_group[keys[j]])
        try:
            v = metric_fn(sample)
        except Exception:
            v = None
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            values.append(float(v))

    if not values:
        return {"estimate": _r(estimate), "lo": None, "hi": None, "n_valid": 0}
    lo, hi = np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"estimate": _r(estimate), "lo": _r(lo), "hi": _r(hi),
            "n_valid": len(values), "iters": iters}


def _r(x):
    if x is None:
        return None
    x = float(x)
    return None if np.isnan(x) else round(x, 4)
