"""Tests for the evaluation set itself.

The queries and the labels are the thing every number is measured against. If
they drift away from the frozen corpus, the numbers stay plausible and stop
being true, which is worse than a crash.
"""

from __future__ import annotations

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read(name):
    with open(os.path.join(ROOT, name), encoding="utf-8") as f:
        return json.load(f)


def ads():
    path = os.path.join(ROOT, "data", "eval_corpus.jsonl")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_every_ad_in_the_frozen_corpus_is_labelled():
    labels = read("eval/labels.json")["labels"]
    assert {a["id"] for a in ads()} == set(labels)


def test_the_label_counts_match_the_labels():
    doc = read("eval/labels.json")
    for k, n in doc["counts"].items():
        assert sum(1 for v in doc["labels"].values() if v == k) == n


def test_every_gold_answer_is_an_ad_that_exists():
    """A gold id that is not in the corpus is unreachable, so it would silently
    cap recall below 1.0 and look like a retrieval failure."""
    have = {a["id"] for a in ads()}
    for q in read("eval/queries.json")["queries"]:
        missing = set(q["gold"]) - have
        assert not missing, "%r points at %s" % (q["query"], missing)


def test_every_query_has_at_least_one_gold_answer():
    for q in read("eval/queries.json")["queries"]:
        assert q["gold"], q["query"]


def test_no_gold_answer_is_an_ad_labelled_a_different_job():
    labels = read("eval/labels.json")["labels"]
    for q in read("eval/queries.json")["queries"]:
        for g in q["gold"]:
            assert labels[g] in ("yes", "borderline"), (q["query"], g)


def test_all_three_families_are_present_and_populated():
    counts = read("eval/queries.json")["counts"]
    assert set(counts) == {"known_item", "keyword_topical", "semantic_topical"}
    assert all(v >= 8 for v in counts.values())


def test_a_known_item_query_does_not_contain_its_answers_title():
    """The query is built from a requirement line with the title words taken
    out. Leave them in and every retriever scores well for the wrong reason."""
    by_id = {a["id"]: a for a in ads()}
    for q in read("eval/queries.json")["queries"]:
        if q["family"] != "known_item":
            continue
        title_words = {w for w in by_id[q["gold"][0]]["title"].lower().split()
                       if len(w) > 3}
        overlap = title_words & set(q["query"].split())
        assert not overlap, "%r leaks %s" % (q["query"], overlap)


def test_a_known_item_query_appears_once_with_all_its_answers():
    """Some of these ads are one template with the job title swapped, so two
    can yield the same query. Splitting that across two entries would mark a
    retriever wrong for returning the other correct ad."""
    queries = read("eval/queries.json")["queries"]
    known = [q["query"] for q in queries if q["family"] == "known_item"]
    assert len(known) == len(set(known))


def test_no_known_item_query_is_boilerplate():
    """The longest line in a lot of ads is the equal opportunity notice, which
    every ad carries, so asking for it back has no single right answer."""
    for q in read("eval/queries.json")["queries"]:
        if q["family"] != "known_item":
            continue
        for leak in ("qualified applicants", "without regard",
                     "discriminate", "religion sex"):
            assert leak not in q["query"], q["query"]


def test_the_semantic_queries_share_few_words_with_their_answers():
    """This family exists to test meaning rather than word overlap. If a query
    repeats the ad's own vocabulary, bm25 can win it and the family stops
    measuring what it is for."""
    by_id = {a["id"]: a for a in ads()}
    for q in read("eval/queries.json")["queries"]:
        if q["family"] != "semantic_topical":
            continue
        words = {w for w in q["query"].lower().split() if len(w) > 4}
        for g in q["gold"]:
            title = by_id[g]["title"].lower()
            shared = {w for w in words if w in title}
            assert len(shared) <= 1, "%r vs %r shares %s" % (
                q["query"], by_id[g]["title"], shared)
