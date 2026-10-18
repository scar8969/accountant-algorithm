"""Feature schema v1 for the compatibility research pipeline.

This is an *experimental feature inventory*, not a validated psychometric
instrument. Nothing here measures a stable psychological construct; it records
self-reported answers on a 1-5 scale plus optional text.

All weights are pre-registered constants. Changing them means bumping
FEATURE_SCHEMA_VERSION, because stored experiments reference this string.
"""

from __future__ import annotations

FEATURE_SCHEMA_VERSION = "1.0.0"

# Trait scale -----------------------------------------------------------------
TRAIT_MIN = 1.0
TRAIT_MAX = 5.0
TRAIT_RANGE = TRAIT_MAX - TRAIT_MIN  # this is R_i in the blueprint formula

DIMENSIONS = (
    "values",
    "communication",
    "lifestyle",
    "relationship_goals",
    "text",
)

# Symmetric similarity features: name -> (dimension, pre-registered weight)
# Similarity is meaningful here: two people who differ by less are scored higher.
SIMILARITY_FEATURES: dict[str, tuple[str, float]] = {
    "ambition": ("values", 0.15),
    "family": ("values", 0.15),
    "independence": ("values", 0.12),
    "directness": ("communication", 0.10),
    "conflict_approach": ("communication", 0.08),
    "social_energy": ("lifestyle", 0.10),
    "routine_structure": ("lifestyle", 0.07),
}

# Complementary feature: one person's *stated preference* against the other's
# actual answer. Handled with a preference function, not a similarity function.
COMPLEMENTARY_FEATURES: dict[str, tuple[str, float]] = {
    "frequency": ("communication", 0.08),
}

GOAL_WEIGHT = 0.10
TEXT_WEIGHT = 0.05  # experimental; ablated in the benchmark

GOALS = ("long_term", "long_term_open", "friendship_first", "casual")

# Symmetric compatibility between stated relationship goals.
GOAL_COMPAT: dict[tuple[str, str], float] = {
    ("long_term", "long_term"): 1.00,
    ("long_term", "long_term_open"): 0.80,
    ("long_term", "friendship_first"): 0.45,
    ("long_term", "casual"): 0.10,
    ("long_term_open", "long_term_open"): 1.00,
    ("long_term_open", "friendship_first"): 0.55,
    ("long_term_open", "casual"): 0.35,
    ("friendship_first", "friendship_first"): 1.00,
    ("friendship_first", "casual"): 0.50,
    ("casual", "casual"): 1.00,
}

# Binary lifestyle flags that can be named as deal-breakers.
FLAGS = ("smoker", "drinks_regularly", "wants_children", "has_pets")

TRAITS = tuple(SIMILARITY_FEATURES) + tuple(COMPLEMENTARY_FEATURES)

# Fields counted toward answer completeness (text is optional and excluded).
ANSWER_FIELDS = TRAITS + ("goal", "interests")
EXPECTED_ANSWERS = len(ANSWER_FIELDS)

# Below this weighted coverage the pair score is withheld entirely.
MIN_COVERAGE = 0.60

# Directional score = mean of (similarity-based compatibility, stated-preference fit)
DIRECTIONAL_SIM_WEIGHT = 0.5


def goal_similarity(a: str | None, b: str | None) -> float | None:
    if a not in GOALS or b not in GOALS:
        return None
    if a == b:
        return 1.0
    key = (a, b) if (a, b) in GOAL_COMPAT else (b, a)
    return GOAL_COMPAT.get(key)


__all__ = [
    "FEATURE_SCHEMA_VERSION",
    "TRAIT_MIN",
    "TRAIT_MAX",
    "TRAIT_RANGE",
    "DIMENSIONS",
    "SIMILARITY_FEATURES",
    "COMPLEMENTARY_FEATURES",
    "GOAL_WEIGHT",
    "TEXT_WEIGHT",
    "GOALS",
    "GOAL_COMPAT",
    "FLAGS",
    "TRAITS",
    "ANSWER_FIELDS",
    "EXPECTED_ANSWERS",
    "MIN_COVERAGE",
    "DIRECTIONAL_SIM_WEIGHT",
    "goal_similarity",
]
