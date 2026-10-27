"""Tests for the retrieval metrics.

Every number in results.md comes out of these four functions, so a quiet bug
here would not look like a bug. It would look like a finding.
"""

from __future__ import annotations

import math

from eval.metrics import (
    evaluate,
    hit_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

RANKED = ["a", "b", "c", "d", "e"]


def test_recall_counts_gold_found_not_results_returned():
    assert recall_at_k(RANKED, {"a", "z"}, 5) == 0.5
    assert recall_at_k(RANKED, {"a", "c"}, 5) == 1.0


def test_recall_respects_the_cutoff():
    assert recall_at_k(RANKED, {"d"}, 3) == 0.0
    assert recall_at_k(RANKED, {"d"}, 4) == 1.0


def test_precision_divides_by_k_not_by_gold():
    assert precision_at_k(RANKED, {"a"}, 5) == 0.2
    assert precision_at_k(RANKED, {"a"}, 1) == 1.0


def test_hit_is_all_or_nothing():
    assert hit_at_k(RANKED, {"c"}, 3) == 1.0
    assert hit_at_k(RANKED, {"c"}, 2) == 0.0


def test_reciprocal_rank_uses_the_first_hit_only():
    assert reciprocal_rank(RANKED, {"c", "d"}) == 1 / 3
    assert reciprocal_rank(RANKED, {"a"}) == 1.0
    assert reciprocal_rank(RANKED, {"zzz"}) == 0.0


def test_ndcg_is_one_when_the_gold_fills_the_top():
    assert ndcg_at_k(["a", "b", "c"], {"a", "b"}, 10) == 1.0


def test_ndcg_rewards_rank_where_recall_cannot_tell_the_difference():
    """Both have the one correct answer inside the top 5, so recall@5 is the
    same. Position is the only difference, and ndcg is what sees it."""
    early, late = ["a", "x", "y"], ["x", "y", "a"]
    gold = {"a"}
    assert recall_at_k(early, gold, 5) == recall_at_k(late, gold, 5)
    assert ndcg_at_k(early, gold, 10) > ndcg_at_k(late, gold, 10)


def test_ndcg_discount_matches_the_definition():
    assert math.isclose(ndcg_at_k(["x", "a"], {"a"}, 10), 1 / math.log2(3))


def test_empty_gold_scores_zero_rather_than_dividing_by_zero():
    for fn in (recall_at_k, hit_at_k, ndcg_at_k):
        assert fn(RANKED, set(), 5) == 0.0
    assert reciprocal_rank(RANKED, set()) == 0.0


def test_nothing_retrieved_scores_zero():
    assert recall_at_k([], {"a"}, 5) == 0.0
    assert reciprocal_rank([], {"a"}) == 0.0


def test_evaluate_averages_over_queries():
    perfect = (["a"], {"a"})
    missed = (["z"], {"a"})
    out = evaluate([perfect, missed])
    assert out["n"] == 2
    assert out["mrr"] == 0.5
    assert out["recall@1"] == 0.5


def test_evaluate_on_nothing_returns_nothing():
    assert evaluate([]) == {}
