"""Synthetic profiles and outcome simulation.

Why this exists: with no real participants yet, the only honest use of synthetic
data is to prove the *pipeline* works — scoring, splits, metrics, leakage guards,
interface. Synthetic results are NOT evidence about human attraction, and every
report generated from this module says so.

The generator is a documented latent-factor model. Three latent dimensions
(warmth, structure, drive) drive the structured answers; interest is then a
function of stated-preference fit, latent affinity, topical overlap, the
candidate's profile completeness, the viewer's choosiness, and a shared
pair-level shock (which is exactly why mutual interest is not the product of two
independent probabilities).
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import numpy as np

from ..features import schema as S
from ..features.structured import answer_completeness, preference_mean

SIMULATION_VERSION = "1.0.0"

INTEREST_TOPICS = (
    "hiking", "cooking", "gaming", "reading", "live music", "travel",
    "fitness", "painting", "films", "technology", "gardening", "photography",
    "cycling", "coffee", "dogs", "cats", "volunteering", "startups",
    "history", "science", "dancing", "climbing", "podcasts", "board games",
)

GOAL_PHRASES = {
    "long_term": "something long-term and serious",
    "long_term_open": "a committed relationship, open to where it goes",
    "friendship_first": "friendship first, then see",
    "casual": "something casual and low-pressure",
}

STYLE_PHRASES = {
    "low": "quiet and slow to open up",
    "mid": "warm once I know someone",
    "high": "very direct about what I think",
}

# trait -> loading on (warmth, structure, drive)
LOADINGS = {
    "ambition": (0.10, 0.35, 0.85),
    "family": (0.75, 0.20, -0.05),
    "independence": (-0.30, 0.10, 0.75),
    "directness": (0.55, -0.15, 0.55),
    "conflict_approach": (0.10, -0.20, 0.80),
    "social_energy": (0.90, 0.05, 0.10),
    "routine_structure": (-0.05, 0.90, 0.10),
    "frequency": (0.45, 0.10, 0.35),
}


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _clip_trait(x: float) -> int:
    return int(min(S.TRAIT_MAX, max(S.TRAIT_MIN, round(float(x)))))


def _jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _latent_affinity(za: np.ndarray, zb: np.ndarray) -> float:
    """1.0 for identical latent vectors, 0.0 at the far edge of the space."""
    distance = float(np.abs(za - zb).sum())
    return max(0.0, 1.0 - distance / (2.0 * len(za)))


def generate_profiles(n: int = 100, seed: int = 20261018,
                      missing_rate: float = 0.10) -> list[dict]:
    rng = np.random.default_rng(seed)
    latents = rng.normal(0.0, 1.0, size=(n, 3))
    now = datetime(2026, 10, 1, tzinfo=timezone.utc)
    profiles: list[dict] = []

    for i in range(n):
        z = latents[i]
        traits: dict[str, int] = {}
        for name, load in LOADINGS.items():
            value = 3.0 + 1.05 * float(np.dot(load, z)) + rng.normal(0, 0.55)
            if rng.random() >= missing_rate:
                traits[name] = _clip_trait(value)

        prefs: dict[str, dict] = {}
        for name in S.TRAITS:
            if name not in traits or rng.random() < missing_rate:
                continue
            width = int(rng.integers(0, 3))
            center = traits[name] + rng.normal(0, 0.9)
            lo = int(min(S.TRAIT_MAX, max(S.TRAIT_MIN, round(center - width))))
            hi = int(min(S.TRAIT_MAX, max(S.TRAIT_MIN, round(center + width))))
            prefs[name] = {"range": [lo, max(lo, hi)],
                           "weight": int(rng.integers(1, 6))}

        goal = None
        if rng.random() >= 0.05:
            probs = np.array([0.42, 0.30, 0.16, 0.12])
            probs = probs * np.array([1.0 + 0.15 * z[1], 1.0, 1.0, 1.0 - 0.1 * z[1]])
            goal = str(rng.choice(list(S.GOALS), p=probs / probs.sum()))

        flags = {
            "smoker": bool(rng.random() < 0.16),
            "drinks_regularly": bool(rng.random() < 0.32),
            "wants_children": bool(rng.random() < _sigmoid(0.8 + 0.6 * z[0])),
            "has_pets": bool(rng.random() < 0.42),
        }
        dealbreakers: list[str] = []
        for flag in S.FLAGS:
            if not flags[flag] and rng.random() < 0.13:
                dealbreakers.append(flag)
        dealbreakers = dealbreakers[:2]

        interests: list[str] = []
        if rng.random() >= 0.05:
            n_topics = int(rng.integers(3, 7))
            # topic preference is weakly tied to the latent dimensions so that
            # text/interest overlap carries a little real signal
            weights = np.array([
                1.0 + 0.35 * (z[0] if k % 3 == 0 else z[1] if k % 3 == 1 else z[2])
                for k in range(len(INTEREST_TOPICS))
            ]).clip(0.15)
            weights = weights / weights.sum()
            idx = rng.choice(len(INTEREST_TOPICS), size=n_topics, replace=False, p=weights)
            interests = sorted(INTEREST_TOPICS[j] for j in idx)

        style_key = "high" if traits.get("directness", 3) >= 4 else (
            "low" if traits.get("directness", 3) <= 2 else "mid")
        text = None
        if interests and rng.random() >= 0.15:
            text = (f"I care about {goal.replace('_', ' ') if goal else 'people'} and "
                    f"good conversation. Weekends are usually {', '.join(interests[:3])}. "
                    f"Friends would say I am {STYLE_PHRASES[style_key]}.")

        updated = now - timedelta(days=float(rng.uniform(0, 240)))
        pid = f"p{i + 1:04d}"
        profiles.append({
            "participant_id": pid,
            "profile_id": f"prof-{pid}",
            "version": 1,
            "feature_schema_version": S.FEATURE_SCHEMA_VERSION,
            "traits": traits,
            "preferences": prefs,
            "goal": goal,
            "flags": flags,
            "dealbreakers": dealbreakers,
            "interests": interests,
            "text": text,
            "updated_at": updated.isoformat(),
            "_latent": [round(float(v), 4) for v in z],
            "_simulated": True,
        })
    return profiles


def simulate_pairs(profiles: list[dict], seed: int = 20261018,
                   exposure_k: int = 12) -> list[dict]:
    """Generate exposed pairs and their (synthetic) outcomes."""
    rng = np.random.default_rng(seed + 1)
    n = len(profiles)
    if n < 2:
        return []

    base = datetime(2026, 10, 1, tzinfo=timezone.utc)
    latents = [np.array(p["_latent"], dtype=float) for p in profiles]
    completeness = [answer_completeness(p) for p in profiles]

    exposure: dict[tuple[int, int], dict] = {}
    for i in range(n):
        affinity = np.array([_latent_affinity(latents[i], latents[j]) for j in range(n)])
        noise = rng.random(n)
        pick_score = np.where(np.arange(n) == i, -1.0, 0.55 * affinity + 0.45 * noise)
        k = min(exposure_k, n - 1)
        chosen = np.argsort(-pick_score)[:k]
        for j in chosen:
            key = (int(min(i, j)), int(max(i, j)))
            entry = exposure.setdefault(key, {"a": key[0], "b": key[1],
                                              "exposed": [False, False]})
            entry["exposed"][0 if i == key[0] else 1] = True

    pairs: list[dict] = []
    for key in sorted(exposure):
        ei, ej = exposure[key]["a"], exposure[key]["b"]
        pa, pb = profiles[ei], profiles[ej]
        za, zb = latents[ei], latents[ej]

        affinity = _latent_affinity(za, zb)
        overlap = _jaccard(pa["interests"], pb["interests"])
        pref_ab = preference_mean(pa, pb) or 0.0
        pref_ba = preference_mean(pb, pa) or 0.0
        pop_a, pop_b = completeness[ei], completeness[ej]
        choosy_a = abs(float(rng.normal(0, 0.5)))
        choosy_b = abs(float(rng.normal(0, 0.5)))
        shock = float(rng.normal(0, 0.7))  # shared by both directions

        gate_ab = any((pb.get("flags") or {}).get(d) is True for d in pa["dealbreakers"])
        gate_ba = any((pa.get("flags") or {}).get(d) is True for d in pb["dealbreakers"])

        def draw(pref, pop_other, choosy, gate):
            logit = (-3.75 + 2.20 * pref + 1.40 * affinity + 1.60 * overlap
                     + 0.80 * pop_other - 0.80 * choosy + shock
                     + float(rng.normal(0, 0.35)))
            p = _sigmoid(logit)
            if gate:
                p *= 0.06
            return bool(rng.random() < p), p

        ia, p_ab = draw(pref_ab, pop_b, choosy_a, gate_ab)
        ib, p_ba = draw(pref_ba, pop_a, choosy_b, gate_ba)

        exposed_a, exposed_b = exposure[key]["exposed"]
        interest_a = ia if exposed_a else None
        interest_b = ib if exposed_b else None
        mutual = (ia and ib) if (exposed_a and exposed_b) else None

        quality_a = quality_b = cont_a = cont_b = None
        if mutual:
            if rng.random() < 0.85:
                qa = 1.0 + 4.0 * (0.55 * affinity + 0.45 * pref_ab) + rng.normal(0, 0.6)
                qb = 1.0 + 4.0 * (0.55 * affinity + 0.45 * pref_ba) + rng.normal(0, 0.6)
                quality_a = int(min(5, max(1, round(qa))))
                quality_b = int(min(5, max(1, round(qb))))
                cont_a = bool(rng.random() < _sigmoid(0.4 + 2.0 * (quality_a - 3) / 2))
                cont_b = bool(rng.random() < _sigmoid(0.4 + 2.0 * (quality_b - 3) / 2))

        pairs.append({
            "a": pa, "b": pb,
            "exposed_a": exposed_a, "exposed_b": exposed_b,
            "interest_a": interest_a, "interest_b": interest_b,
            "mutual": mutual,
            "quality_a": quality_a, "quality_b": quality_b,
            "continued_a": cont_a, "continued_b": cont_b,
            "ts_a": (base + timedelta(days=float(rng.uniform(0, 60)))).isoformat(),
            "ts_b": (base + timedelta(days=float(rng.uniform(0, 60)))).isoformat(),
            "sim_p_interest_a": round(p_ab, 6),
            "sim_p_interest_b": round(p_ba, 6),
            "_synthetic": True,
        })
    return pairs


def simulation_manifest() -> dict:
    return {
        "version": SIMULATION_VERSION,
        "latent_dimensions": ["warmth", "structure", "drive"],
        "interest_model": (
            "logit = -1.15 + 2.35*pref_fit + 1.30*latent_affinity + 1.10*topic_overlap "
            "+ 1.05*candidate_completeness - 1.00*viewer_choosiness + pair_shock + noise; "
            "deal-breaker violations are multiplied by 0.06 inside the logit->p step."),
        "exposure_model": (
            "Each profile reviews k candidates ranked by 0.55*latent_affinity + "
            "0.45*uniform noise, so exposure is correlated with unobserved affinity "
            "(a stand-in for platform exposure bias). Interest labels exist only for "
            "exposed directions; everything else is unknown, never negative."),
        "interaction_model": (
            "Conversation happens with p=0.85 when both sides are interested; quality "
            "1-5 is a noisy function of affinity and stated-preference fit."),
        "warning": ("Synthetic outcomes are generated from a known model. Beating a "
                    "baseline here validates the pipeline, not the theory."),
    }
