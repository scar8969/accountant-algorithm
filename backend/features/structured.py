"""Structured feature engine: validation, trait vectors, pair scoring.

The scoring function follows section 4.2 of the blueprint:

    s_i(A,B) = max(0, 1 - |x_Ai - x_Bi| / R_i)

for features where similarity is desirable, and a *preference satisfaction*
function for complementary features (A's stated acceptable range vs B's answer).

Missing answers get mask m_i = 0. If weighted coverage falls below
MIN_COVERAGE the pair is reported as insufficient data rather than scored.
Deal-breakers are applied as hard directional gates, before ranking.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone

from . import schema as S
from .embeddings import cosine


# --------------------------------------------------------------------------- #
# primitives
# --------------------------------------------------------------------------- #
def similarity(x: float, y: float, r: float = S.TRAIT_RANGE) -> float:
    """Similarity for features where closeness is desirable. Always in [0, 1]."""
    return max(0.0, 1.0 - abs(float(x) - float(y)) / r)


def pref_satisfaction(acceptable: tuple[float, float] | list[float], x: float,
                      r: float = S.TRAIT_RANGE) -> float:
    """How well an answer x fits a stated acceptable range.

    1.0 inside the range, then linear decay, 0.0 when `r` away from the boundary.
    """
    lo, hi = float(acceptable[0]), float(acceptable[1])
    x = float(x)
    if lo <= x <= hi:
        return 1.0
    dist = min(abs(x - lo), abs(x - hi))
    return max(0.0, 1.0 - dist / r)


def _mean(values: list[float]) -> float | None:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
def validate_profile(profile: dict) -> list[str]:
    """Return a list of schema errors. Empty list means valid."""
    errors: list[str] = []
    if not isinstance(profile, dict):
        return ["profile must be an object"]

    traits = profile.get("traits") or {}
    for name, value in traits.items():
        if name not in S.TRAITS:
            errors.append(f"unknown trait: {name}")
            continue
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            errors.append(f"{name} must be numeric")
        elif not (S.TRAIT_MIN <= float(value) <= S.TRAIT_MAX):
            errors.append(f"{name} out of range [{S.TRAIT_MIN}, {S.TRAIT_MAX}]")

    prefs = profile.get("preferences") or {}
    for name, pref in prefs.items():
        if name not in S.TRAITS:
            errors.append(f"unknown preference target: {name}")
            continue
        rng = (pref or {}).get("range")
        if not rng:
            errors.append(f"preference {name} missing range")
            continue
        if len(rng) != 2:
            errors.append(f"preference {name} range must be [lo, hi]")
            continue
        lo, hi = rng
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in rng):
            errors.append(f"preference {name} range must be numeric")
        elif not (S.TRAIT_MIN <= lo <= hi <= S.TRAIT_MAX):
            errors.append(f"preference {name} range must satisfy 1 <= lo <= hi <= 5")
        w = (pref or {}).get("weight", 3)
        if not isinstance(w, (int, float)) or not (1 <= w <= 5):
            errors.append(f"preference {name} weight must be in [1, 5]")

    goal = profile.get("goal")
    if goal is not None and goal not in S.GOALS:
        errors.append(f"unknown goal: {goal}")

    for key, value in (profile.get("flags") or {}).items():
        if key not in S.FLAGS:
            errors.append(f"unknown flag: {key}")
        elif not isinstance(value, bool):
            errors.append(f"flag {key} must be boolean")

    for db in profile.get("dealbreakers") or []:
        if db not in S.FLAGS:
            errors.append(f"unknown deal-breaker: {db}")

    interests = profile.get("interests") or []
    if not isinstance(interests, list) or not all(isinstance(i, str) for i in interests):
        errors.append("interests must be a list of strings")

    if profile.get("text") is not None and not isinstance(profile["text"], str):
        errors.append("text must be a string or null")

    return errors


# --------------------------------------------------------------------------- #
# confidence
# --------------------------------------------------------------------------- #
def answer_completeness(profile: dict) -> float:
    """Fraction of expected instrument fields that were answered."""
    answered = 0
    traits = profile.get("traits") or {}
    for name in S.TRAITS:
        if traits.get(name) is not None:
            answered += 1
    if profile.get("goal") is not None:
        answered += 1
    if profile.get("interests"):
        answered += 1
    return answered / S.EXPECTED_ANSWERS


def recency_factor(updated_at: str | None, reference_time: datetime | None = None,
                   half_life_days: float = 365.0) -> float:
    """Exponential decay of answer recency. 1.0 when no timestamp is available."""
    if not updated_at or reference_time is None:
        return 1.0
    try:
        ts = datetime.fromisoformat(str(updated_at).replace("Z", "+00:00"))
    except ValueError:
        return 1.0
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    age_days = max(0.0, (reference_time - ts).total_seconds() / 86400.0)
    return math.exp(-math.log(2) * age_days / half_life_days)


def profile_confidence(profile: dict, reference_time: datetime | None = None) -> float:
    return answer_completeness(profile) * recency_factor(profile.get("updated_at"),
                                                        reference_time)


# --------------------------------------------------------------------------- #
# pair scoring
# --------------------------------------------------------------------------- #
@dataclass
class PairScore:
    a_id: str
    b_id: str
    score: float | None                    # 0-100, None when insufficient
    coverage: float
    insufficient: bool
    dimensions: dict[str, float] = field(default_factory=dict)
    leaves: dict[str, dict] = field(default_factory=dict)
    gate_a: list[str] = field(default_factory=list)   # A's deal-breakers hit by B
    gate_b: list[str] = field(default_factory=list)
    eligible_a: bool = True                # A may rank B
    eligible_b: bool = True
    confidence_a: float = 0.0
    confidence_b: float = 0.0
    limitations: list[str] = field(default_factory=list)

    @property
    def gated(self) -> bool:
        return bool(self.gate_a or self.gate_b)


def _dealbreakers_hit(owner: dict, other: dict) -> list[str]:
    other_flags = other.get("flags") or {}
    return [db for db in (owner.get("dealbreakers") or []) if other_flags.get(db) is True]


def score_pair(a: dict, b: dict, embed=None,
               reference_time: datetime | None = None) -> PairScore:
    """Interpretable 0-100 compatibility score for an ordered pair."""
    a_traits = a.get("traits") or {}
    b_traits = b.get("traits") or {}
    a_prefs = a.get("preferences") or {}
    b_prefs = b.get("preferences") or {}

    leaves: dict[str, dict] = {}

    for name, (dim, weight) in S.SIMILARITY_FEATURES.items():
        x, y = a_traits.get(name), b_traits.get(name)
        s = similarity(x, y) if (x is not None and y is not None) else None
        leaves[name] = {"kind": "similarity", "dim": dim, "w": weight, "s": s,
                        "a": x, "b": y}

    # Complementary: stated frequency preference vs the other's actual answer.
    f_a, f_b = a_traits.get("frequency"), b_traits.get("frequency")
    rng_a = (a_prefs.get("frequency") or {}).get("range")
    rng_b = (b_prefs.get("frequency") or {}).get("range")
    s_ab = pref_satisfaction(rng_a, f_b) if (rng_a and f_b is not None) else None
    s_ba = pref_satisfaction(rng_b, f_a) if (rng_b and f_a is not None) else None
    dim, weight = S.COMPLEMENTARY_FEATURES["frequency"]
    leaves["frequency"] = {
        "kind": "complementary", "dim": dim, "w": weight,
        "s": _mean([s_ab, s_ba]), "s_a_to_b": s_ab, "s_b_to_a": s_ba,
        "a": f_a, "b": f_b,
    }

    goal_s = S.goal_similarity(a.get("goal"), b.get("goal"))
    leaves["goal"] = {"kind": "categorical", "dim": "relationship_goals",
                      "w": S.GOAL_WEIGHT, "s": goal_s,
                      "a": a.get("goal"), "b": b.get("goal")}

    # Optional text similarity: experimental feature, weight only counted when
    # both descriptions exist.
    text_s = None
    if embed is not None and a.get("text") and b.get("text"):
        text_s = max(0.0, cosine(embed.embed(a["text"]), embed.embed(b["text"])))
        leaves["text"] = {"kind": "embedding", "dim": "text",
                          "w": S.TEXT_WEIGHT, "s": text_s}

    core_weight = sum(v["w"] for k, v in leaves.items() if k != "text")
    present_core = sum(v["w"] for k, v in leaves.items()
                       if k != "text" and v["s"] is not None)
    coverage = present_core / core_weight if core_weight else 0.0

    numerator = sum(v["w"] * v["s"] for v in leaves.values() if v["s"] is not None)
    denominator = sum(v["w"] for v in leaves.values() if v["s"] is not None)

    insufficient = coverage < S.MIN_COVERAGE or denominator == 0.0
    score = None if insufficient else round(100.0 * numerator / denominator, 2)

    # Dimension means (weighted by feature weight, present features only).
    dims: dict[str, float] = {}
    for dim_name in S.DIMENSIONS:
        vals = [(v["w"], v["s"]) for v in leaves.values()
                if v["dim"] == dim_name and v["s"] is not None]
        if vals:
            dims[dim_name] = round(sum(w * s for w, s in vals) / sum(w for w, _ in vals), 4)

    gate_a = _dealbreakers_hit(a, b)
    gate_b = _dealbreakers_hit(b, a)

    limitations = [
        "Score is on an experimental 0-100 scale; it is not a probability of a "
        "successful relationship.",
        "No claim about long-term relationship success is supported by these features.",
    ]
    if insufficient:
        limitations.append(
            f"Withheld: weighted coverage {coverage:.0%} is below the "
            f"{S.MIN_COVERAGE:.0%} threshold.")
    if text_s is None:
        limitations.append("Text similarity not computed (no embeddings or no text).")
    if gate_a or gate_b:
        limitations.append("A deal-breaker applies; the pair is ineligible for ranking "
                           "in the gated direction.")

    return PairScore(
        a_id=str(a.get("participant_id") or a.get("profile_id") or ""),
        b_id=str(b.get("participant_id") or b.get("profile_id") or ""),
        score=score, coverage=round(coverage, 4), insufficient=insufficient,
        dimensions=dims, leaves=leaves, gate_a=gate_a, gate_b=gate_b,
        eligible_a=not gate_a and not insufficient,
        eligible_b=not gate_b and not insufficient,
        confidence_a=round(profile_confidence(a, reference_time), 4),
        confidence_b=round(profile_confidence(b, reference_time), 4),
        limitations=limitations,
    )


def preference_mean(owner: dict, other: dict) -> float | None:
    """Weighted mean satisfaction of *owner's* stated preferences by *other*."""
    owner_prefs = owner.get("preferences") or {}
    other_traits = other.get("traits") or {}
    num = den = 0.0
    for name in S.TRAITS:
        pref = owner_prefs.get(name) or {}
        rng = pref.get("range")
        value = other_traits.get(name)
        if not rng or value is None:
            continue
        w = float(pref.get("weight", 3))
        num += w * pref_satisfaction(rng, value)
        den += w
    return None if den == 0 else num / den


def directional_score(a: dict, b: dict, ps: PairScore | None = None,
                      embed=None) -> float | None:
    """Score for 'how much would A be interested in B', in [0, 100].

    50/50 blend of similarity-based compatibility and A's stated preferences.
    Returns 0.0 when A's deal-breakers are violated, None when insufficient.
    """
    ps = ps or score_pair(a, b, embed=embed)
    if ps.score is None:
        return None
    if ps.gate_a:
        return 0.0
    pref = preference_mean(a, b)
    if pref is None:
        return ps.score
    blended = (S.DIRECTIONAL_SIM_WEIGHT * (ps.score / 100.0)
               + (1 - S.DIRECTIONAL_SIM_WEIGHT) * pref)
    return round(100.0 * blended, 2)


def explanation(ps: PairScore, top: int = 4) -> dict:
    """Positive signals, conflicts and missing data for one pair."""
    present = [(k, v) for k, v in ps.leaves.items() if v["s"] is not None]
    denom = sum(v["w"] for _, v in present) or 1.0

    scored = sorted(
        ((k, v["s"], v["w"] / denom * 100.0, v["w"] / denom * 100.0 * v["s"])
         for k, v in present),
        key=lambda t: t[3],
    )
    points = [{"feature": k, "similarity": round(s, 4),
               "weight_pct": round(w, 2), "points": round(p, 2)}
              for k, s, w, p in scored]
    return {
        "contributions": points[-top:][::-1],
        "conflicts": points[:top],
        "missing": sorted(k for k, v in ps.leaves.items() if v["s"] is None),
        "dimensions": ps.dimensions,
        "gate_a": ps.gate_a,
        "gate_b": ps.gate_b,
        "coverage": ps.coverage,
    }


