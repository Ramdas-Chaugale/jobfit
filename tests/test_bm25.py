"""Tests for the lexical baseline.

BM25 is here to keep the embeddings honest, so it has to be right. A broken
baseline that scores badly would make the vector index look good for free.
"""

from __future__ import annotations

import math

from jobfit.bm25 import BM25, tokens

DOCS = [
    "We need a machine learning engineer with PyTorch and computer vision.",
    "Account executive, SaaS sales, quota carrying, DACH region.",
    "Data engineer building pipelines with Spark, Airflow and dbt on AWS.",
    "Senior machine learning engineer, PyTorch, Kubernetes, model serving.",
]


def test_tokeniser_keeps_technology_names_intact():
    """c++ and node.js are the whole query sometimes. Splitting them loses it."""
    out = tokens("Experience with C++, C#, Node.js and .NET (5+ years)")
    assert "c++" in out
    assert "c#" in out
    assert "node.js" in out
    assert "5+" in out


def test_tokeniser_lowercases_and_drops_punctuation():
    assert tokens("PyTorch, TensorFlow.") == ["pytorch", "tensorflow"]


def test_the_most_relevant_document_ranks_first():
    top = BM25(DOCS).top("spark airflow dbt pipelines")
    assert top
    assert top[0][0] == 2


def test_a_term_in_no_document_scores_nothing():
    assert BM25(DOCS).top("kubernetes golang rust elixir haskell ocaml") != []
    assert BM25(DOCS).top("zzzzqqq") == []


def test_an_empty_query_returns_nothing():
    assert BM25(DOCS).top("") == []


def test_idf_never_goes_negative():
    """A word in every document must not score below zero, or a document is
    ranked down for containing a word the query asked for. The older idf
    formula does exactly that, which is why this one is not it."""
    b = BM25(["common term here", "common term there", "common term everywhere"])
    assert all(v >= 0 for v in b.idf.values())
    assert b.idf["common"] < b.idf["here"]


def test_a_rare_term_outweighs_a_common_one():
    b = BM25(DOCS)
    assert b.idf["spark"] > b.idf["machine"]


def test_length_normalisation_does_not_simply_favour_long_documents():
    """Same single match, very different lengths. The long one must not win on
    having more words around the match."""
    short = "pytorch"
    long = "pytorch " + ("filler words about the company and its mission " * 30)
    b = BM25([short, long])
    sc = b.scores("pytorch")
    assert sc[0] > sc[1]


def test_an_empty_corpus_does_not_divide_by_zero():
    b = BM25([])
    assert b.scores("anything") == []
    assert not math.isnan(b.avg_len)
