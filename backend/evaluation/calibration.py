"""Calibration diagnostics: reliability curve, ECE, Brier decomposition note.

Interpretation guard: a calibrated probability means "of pairs scored p, about
p were positive in this sample" — nothing about relationship success.
"""

from __future__ import annotations

import numpy as np

from .classification import brier, ece, reliability


def calibration_block(y: np.ndarray, p: np.ndarray, bins: int = 10) -> dict:
    return {
        "n": int(len(y)),
        "brier": None if len(y) == 0 else round(brier(y, p), 4),
        "ece": None if len(y) == 0 else round(ece(y, p, bins), 4),
        "bins": reliability(y, p, bins),
        "note": ("Probability calibration on held-out pairs. With few positives "
                 "these bins are noisy; read them with the reported intervals."),
    }
