"""The retrievers being compared, all over the same frozen corpus.

Each one takes a query and returns ad ids, best first. That is the whole
interface, which is what lets one set of metrics score all of them.

Scoring note. The production index is a faiss IndexFlatIP over L2 normalised
vectors, which is an exact inner product search. Here the same vectors are held
in a numpy array and searched with a dot product, which is the identical
arithmetic, so the ranking is the same. Doing it this way means a variant can
reweight scores before ranking, which is the point of the exercise, and it
means the evaluation runs without faiss.

Vectors are cached in data/eval_vectors.npy and committed, so the evaluation
reproduces on a clean clone without downloading a model.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from jobfit.bm25 import BM25  # noqa: E402
from jobfit.chunking import chunk_all  # noqa: E402
from jobfit.corpus import Ad  # noqa: E402
from jobfit.relevance import is_boilerplate  # noqa: E402
from jobfit.relevance import is_relevant as new_filter  # noqa: E402

CORPUS = os.path.join(ROOT, "data", "eval_corpus.jsonl")
VECTORS = os.path.join(ROOT, "data", "eval_vectors.npy")
QUERY_VECTORS = os.path.join(ROOT, "data", "eval_query_vectors.npz")

RRF_K = 60          # the usual constant in reciprocal rank fusion
BOILER_WEIGHT = 0.5  # what a chunk with no requirement signal keeps


def load_ads() -> list[Ad]:
    with open(CORPUS, encoding="utf-8") as f:
        return [Ad(**json.loads(line)) for line in f if line.strip()]


class Corpus:
    """Chunks, their vectors and a BM25 index, built once and shared."""

    def __init__(self, filtered: bool = False):
        ads = load_ads()
        self.kept_ads = (
            [a for a in ads if new_filter(a.title, a.text)]
            if filtered else ads)
        kept = {a.id for a in self.kept_ads}

        all_chunks = chunk_all(ads)
        self.all_ids = [c.chunk_id for c in all_chunks]
        vectors = self._vectors(all_chunks)

        pick = [i for i, c in enumerate(all_chunks) if c.ad_id in kept]
        self.chunks = [all_chunks[i] for i in pick]
        self.vectors = vectors[pick]
        self.bm25 = BM25([c.for_embedding() for c in self.chunks])
        self.boiler = np.array(
            [BOILER_WEIGHT if is_boilerplate(c.text) else 1.0
             for c in self.chunks], dtype="float32")

    def _vectors(self, chunks) -> np.ndarray:
        """Load the cached vectors, or embed and cache them."""
        if os.path.exists(VECTORS):
            v = np.load(VECTORS)
            if v.shape[0] == len(chunks):
                return v
            print("  cached vectors are stale, re-embedding")
        from jobfit.index import DEFAULT_MODEL, embed
        print("  embedding %d chunks with %s" % (len(chunks), DEFAULT_MODEL))
        v = embed([c.for_embedding() for c in chunks], DEFAULT_MODEL,
                  progress=True)
        np.save(VECTORS, v)
        return v

    def query_vector(self, query: str) -> np.ndarray:
        return query_vector(query)


# Query vectors are cached and committed for the same reason the chunk vectors
# are: six variants over sixty six queries would otherwise re-embed the same
# text hundreds of times, and a clean clone could not run the evaluation at all
# without downloading the model.
_QCACHE: dict[str, np.ndarray] = {}
if os.path.exists(QUERY_VECTORS):
    with np.load(QUERY_VECTORS) as _z:
        _QCACHE = {str(k): _z[k] for k in _z.files}


def query_vector(query: str) -> np.ndarray:
    if query not in _QCACHE:
        from jobfit.index import DEFAULT_MODEL, embed
        _QCACHE[query] = embed([query], DEFAULT_MODEL)[0]
        np.savez(QUERY_VECTORS, **_QCACHE)
    return _QCACHE[query]


def _roll_up(scored: list[tuple[int, float]], corpus: Corpus,
             k: int) -> list[str]:
    """Chunks to ads. An ad takes its best chunk's score, once."""
    best: dict[str, float] = {}
    for i, s in scored:
        ad_id = corpus.chunks[i].ad_id
        if s > best.get(ad_id, float("-inf")):
            best[ad_id] = s
    return [a for a, _ in sorted(best.items(), key=lambda kv: -kv[1])][:k]


def dense(corpus: Corpus, query: str, k: int = 10,
          boiler: bool = False) -> list[str]:
    q = corpus.query_vector(query)
    scores = corpus.vectors @ q
    if boiler:
        scores = scores * corpus.boiler
    order = np.argsort(-scores)[:k * 8]
    return _roll_up([(int(i), float(scores[i])) for i in order], corpus, k)


def lexical(corpus: Corpus, query: str, k: int = 10) -> list[str]:
    return _roll_up(corpus.bm25.top(query, k * 8), corpus, k)


def hybrid(corpus: Corpus, query: str, k: int = 10,
           boiler: bool = False) -> list[str]:
    """Reciprocal rank fusion of the two.

    Fusing ranks rather than scores sidesteps the fact that a cosine
    similarity and a BM25 score are not on the same scale and cannot be added
    without a normalisation that would itself need tuning.
    """
    pool = k * 8
    q = corpus.query_vector(query)
    d = corpus.vectors @ q
    if boiler:
        d = d * corpus.boiler
    dense_rank = list(np.argsort(-d)[:pool])
    lex_rank = [i for i, _ in corpus.bm25.top(query, pool)]

    fused: dict[int, float] = {}
    for ranking in (dense_rank, lex_rank):
        for rank, i in enumerate(ranking, start=1):
            fused[int(i)] = fused.get(int(i), 0.0) + 1.0 / (RRF_K + rank)
    return _roll_up(sorted(fused.items(), key=lambda kv: -kv[1]), corpus, k)


# name -> (needs filtered corpus, callable)
VARIANTS = {
    "dense (baseline)": (False, lambda c, q, k: dense(c, q, k)),
    "bm25 only": (False, lambda c, q, k: lexical(c, q, k)),
    "hybrid rrf": (False, lambda c, q, k: hybrid(c, q, k)),
    "hybrid + boilerplate": (False, lambda c, q, k: hybrid(c, q, k,
                                                           boiler=True)),
    "hybrid + relevance filter": (True, lambda c, q, k: hybrid(c, q, k)),
    "hybrid + filter + boilerplate": (True, lambda c, q, k: hybrid(
        c, q, k, boiler=True)),
}
