"""Evaluation protocols.

Two protocols, because matching data has two different generalisation questions:

1. Unseen-person (participant-grouped): every participant appears in exactly one
   side. Pairs straddling the split are dropped, since one endpoint's answers
   would otherwise be seen in training. This tests generalisation to new people.
2. Known-person (temporal): train on earlier observations, test on later ones,
   allowing the same participants on both sides. This tests whether the model
   improves for people it has already observed.

Random row-level splits are never used: with pair data they leak one person's
outcomes across both sides of the split.
"""

from __future__ import annotations

from datetime import datetime
from itertools import combinations


def participant_ids(pair: dict) -> tuple[str, str]:
    return (pair["a"]["participant_id"], pair["b"]["participant_id"])


def pair_key(pair: dict) -> tuple[str, str]:
    a, b = participant_ids(pair)
    return (a, b) if a <= b else (b, a)


def pair_time(pair: dict) -> datetime:
    ts_a = datetime.fromisoformat(str(pair.get("ts_a")))
    ts_b = datetime.fromisoformat(str(pair.get("ts_b") or pair.get("ts_a")))
    return max(ts_a, ts_b)


def grouped_pair_split(pairs: list[dict], test_frac: float = 0.3,
                       seed: int = 0) -> tuple[list[dict], list[dict], list[dict]]:
    """Split by participant; drop pairs that straddle the two sides.

    Refuses to produce a split that cannot support evaluation: a silent empty test
    set would make every downstream metric meaningless.
    """
    people = sorted({p for pair in pairs for p in participant_ids(pair)})
    if len(people) < 8:
        raise ValueError(f"need at least 8 participants for a grouped split, got "
                         f"{len(people)}")
    if len(pairs) < 12:
        raise ValueError(f"need at least 12 pairs for a grouped split, got {len(pairs)}")

    import numpy as np
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(people))
    n_test = max(1, min(len(people) - 1, int(round(len(people) * test_frac))))
    test_ids = {people[i] for i in order[:n_test]}
    train_ids = set(people) - test_ids

    train, test, dropped = [], [], []
    for pair in pairs:
        a, b = participant_ids(pair)
        if a in train_ids and b in train_ids:
            train.append(pair)
        elif a in test_ids and b in test_ids:
            test.append(pair)
        else:
            dropped.append(pair)
    if not test or not train:
        raise ValueError("grouped split produced an empty train or test set")
    return train, test, dropped


def temporal_pair_split(pairs: list[dict],
                        train_frac: float = 0.7) -> tuple[list[dict], list[dict]]:
    ordered = sorted(pairs, key=pair_time)
    cut = int(round(len(ordered) * train_frac))
    cut = max(1, min(len(ordered) - 1, cut)) if len(ordered) > 1 else len(ordered)
    return ordered[:cut], ordered[cut:]


def assert_no_leakage(train: list[dict], test: list[dict], *,
                      require_disjoint_people: bool = True) -> None:
    """Hard failure rather than a report: a leaking split must stop the run.

    The temporal protocol deliberately allows the same person on both sides, so
    it is checked for pair overlap and ordering instead (see
    `assert_temporal_order`).
    """
    report = leakage_report(train, test)
    problems = []
    if require_disjoint_people and not report["disjoint_participants"]:
        problems.append(f"{len(report['shared_participants'])} shared participants")
    if not report["disjoint_pairs"]:
        problems.append(f"{report['shared_pairs']} shared pairs")
    if problems:
        raise AssertionError("split leakage detected — " + "; ".join(problems))


def assert_temporal_order(train: list[dict], test: list[dict]) -> None:
    """Train observations must all precede test observations; pairs must not repeat."""
    if not train or not test:
        raise ValueError("temporal split produced an empty train or test set")
    latest_train = max(pair_time(p) for p in train)
    earliest_test = min(pair_time(p) for p in test)
    if latest_train > earliest_test:
        raise AssertionError(
            f"temporal leakage: train extends to {latest_train.isoformat()} but test "
            f"starts at {earliest_test.isoformat()}")
    if {pair_key(p) for p in train} & {pair_key(p) for p in test}:
        raise AssertionError("temporal split shares pairs between train and test")


def leakage_report(train: list[dict], test: list[dict]) -> dict:
    train_people = {p for pair in train for p in participant_ids(pair)}
    test_people = {p for pair in test for p in participant_ids(pair)}
    train_keys = {pair_key(p) for p in train}
    test_keys = {pair_key(p) for p in test}
    return {
        "train_pairs": len(train),
        "test_pairs": len(test),
        "train_participants": len(train_people),
        "test_participants": len(test_people),
        "shared_participants": sorted(train_people & test_people),
        "shared_pairs": len(train_keys & test_keys),
        "disjoint_participants": not (train_people & test_people),
        "disjoint_pairs": not (train_keys & test_keys),
    }


def candidate_sets(pairs: list[dict]) -> dict[str, list[dict]]:
    """Group pairs into per-query candidate lists (both directions)."""
    out: dict[str, list[dict]] = {}
    for pair in pairs:
        for query, candidate in ((pair["a"], pair["b"]), (pair["b"], pair["a"])):
            qid = query["participant_id"]
            cid = candidate["participant_id"]
            if pair["interest_a"] is None and pair["interest_b"] is None:
                continue
            interest = pair["interest_a"] if query is pair["a"] else pair["interest_b"]
            if interest is None:
                continue
            out.setdefault(qid, []).append({
                "query": query, "candidate": candidate,
                "interest": bool(interest),
                "mutual": pair.get("mutual"),
                "pair": pair,
            })
    return out


def overlapping_pairs(pairs_a: list[dict], pairs_b: list[dict]) -> int:
    keys = {pair_key(p) for p in pairs_a}
    return sum(1 for p in pairs_b if pair_key(p) in keys)


__all__ = ["grouped_pair_split", "temporal_pair_split", "leakage_report",
           "assert_no_leakage", "assert_temporal_order", "candidate_sets",
           "pair_key", "pair_time", "participant_ids", "overlapping_pairs"]
