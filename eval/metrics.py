"""Retrieval metrics, written out so the definitions are visible.

Every one of these takes a ranked list of ad ids and the set that should have
come back, and nothing else. Keeping them free of the retriever is what lets
the same numbers compare four different retrievers.
"""

from __future__ import annotations

import math


def recall_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    """Share of the correct ads that made the top k."""
    if not gold:
        return 0.0
    return len(set(ranked[:k]) & gold) / len(gold)


def precision_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    return len(set(ranked[:k]) & gold) / k


def hit_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    """1 if anything correct is in the top k. The known-item question."""
    return 1.0 if set(ranked[:k]) & gold else 0.0


def reciprocal_rank(ranked: list[str], gold: set[str]) -> float:
    """1/rank of the first correct ad, 0 if none. Averaged this is MRR.

    Rewards getting one right answer to the very top, which is what matters
    when a person reads three results and stops.
    """
    for i, ad_id in enumerate(ranked, start=1):
        if ad_id in gold:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    """Binary-gain nDCG. Unlike recall it cares where in the top k they land."""
    if not gold:
        return 0.0
    dcg = sum(1.0 / math.log2(i + 1)
              for i, ad_id in enumerate(ranked[:k], start=1) if ad_id in gold)
    ideal = sum(1.0 / math.log2(i + 1)
                for i in range(1, min(len(gold), k) + 1))
    return dcg / ideal if ideal else 0.0


def evaluate(results: list[tuple[list[str], set[str]]],
             ks: tuple[int, ...] = (1, 3, 5, 10)) -> dict[str, float]:
    """Average every metric over a list of (ranked ids, gold ids) pairs."""
    if not results:
        return {}
    n = len(results)
    out: dict[str, float] = {}
    for k in ks:
        out["recall@%d" % k] = sum(
            recall_at_k(r, g, k) for r, g in results) / n
        out["hit@%d" % k] = sum(hit_at_k(r, g, k) for r, g in results) / n
    out["mrr"] = sum(reciprocal_rank(r, g) for r, g in results) / n
    out["ndcg@10"] = sum(ndcg_at_k(r, g, 10) for r, g in results) / n
    out["n"] = float(n)
    return out
