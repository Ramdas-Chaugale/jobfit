"""Search service behind the web interface.

It imports the retrievers from eval/ rather than reimplementing them, so the
thing a visitor types into is the same code the numbers in results.md came
from. A demo that quietly uses a different code path to the evaluation is worse
than no demo.

The one thing added here is the justifying chunk. The evaluation only needs to
know which ad came back; a person needs to see the sentence that explains why.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from eval.retrievers import RRF_K, Corpus, load_ads  # noqa: E402
from jobfit.relevance import is_boilerplate, why  # noqa: E402

POOL = 80
QUERIES = os.path.join(ROOT, "eval", "queries.json")
QUERY_VECTORS = os.path.join(ROOT, "data", "eval_query_vectors.npz")

# name -> (needs the filtered corpus, method, downweight boilerplate)
# Same six the evaluation reports, in the same order.
VARIANTS: dict[str, tuple[bool, str, bool]] = {
    "dense (baseline)": (False, "dense", False),
    "bm25 only": (False, "bm25", False),
    "hybrid rrf": (False, "hybrid", False),
    "hybrid + boilerplate": (False, "hybrid", True),
    "hybrid + relevance filter": (True, "hybrid", False),
    "hybrid + filter + boilerplate": (True, "hybrid", True),
}

DEFAULT_VARIANT = "hybrid + filter + boilerplate"


@dataclass
class Hit:
    ad_id: str
    title: str
    company: str
    location: str
    url: str
    score: float
    evidence: str          # the chunk that put this ad here
    boilerplate: bool      # the chunk carries no requirement signal
    why_relevant: str      # what the corpus filter saw in the title

    def as_dict(self) -> dict:
        return {
            "ad_id": self.ad_id, "title": self.title, "company": self.company,
            "location": self.location, "url": self.url,
            "score": round(self.score, 4), "evidence": self.evidence,
            "boilerplate": self.boilerplate, "why_relevant": self.why_relevant,
        }


class SearchService:
    def __init__(self) -> None:
        self.plain = Corpus(filtered=False)
        self.filtered = Corpus(filtered=True)
        self.ads = {a.id: a for a in load_ads()}
        self._cached = self._cached_queries()

    # --- what a visitor can run without waiting for a model download -------

    @staticmethod
    def _cached_queries() -> set[str]:
        if not os.path.exists(QUERY_VECTORS):
            return set()
        with np.load(QUERY_VECTORS) as z:
            return {str(k) for k in z.files}

    def is_instant(self, query: str) -> bool:
        """True when the query vector is already on disk. A fresh query has to
        load the embedding model first, which is slow the first time."""
        return query in self._cached

    def examples(self) -> list[dict]:
        with open(QUERIES, encoding="utf-8") as f:
            data = json.load(f)
        return [{"family": q["family"], "query": q["query"],
                 "gold": len(q.get("gold", []))} for q in data["queries"]]

    # --- retrieval ---------------------------------------------------------

    def _scored_chunks(self, corpus: Corpus, query: str, method: str,
                       boiler: bool) -> list[tuple[int, float]]:
        if method == "bm25":
            return corpus.bm25.top(query, POOL)

        q = corpus.query_vector(query)
        dense = corpus.vectors @ q
        if boiler:
            dense = dense * corpus.boiler

        if method == "dense":
            order = np.argsort(-dense)[:POOL]
            return [(int(i), float(dense[i])) for i in order]

        # hybrid: reciprocal rank fusion, the same constant the evaluation uses
        dense_rank = [int(i) for i in np.argsort(-dense)[:POOL]]
        lex_rank = [i for i, _ in corpus.bm25.top(query, POOL)]
        fused: dict[int, float] = {}
        for ranking in (dense_rank, lex_rank):
            for rank, i in enumerate(ranking, start=1):
                fused[int(i)] = fused.get(int(i), 0.0) + 1.0 / (RRF_K + rank)
        return sorted(fused.items(), key=lambda kv: -kv[1])

    def search(self, query: str, variant: str = DEFAULT_VARIANT,
               k: int = 10) -> list[Hit]:
        if variant not in VARIANTS:
            raise KeyError(variant)
        use_filtered, method, boiler = VARIANTS[variant]
        corpus = self.filtered if use_filtered else self.plain

        # Roll chunks up to ads, keeping the chunk that won, because that
        # chunk is the answer to "why am I being shown this".
        best: dict[str, tuple[float, int]] = {}
        for i, score in self._scored_chunks(corpus, query, method, boiler):
            ad_id = corpus.chunks[i].ad_id
            if score > best.get(ad_id, (float("-inf"), -1))[0]:
                best[ad_id] = (score, i)

        ranked = sorted(best.items(), key=lambda kv: -kv[1][0])[:k]
        hits = []
        for ad_id, (score, chunk_i) in ranked:
            ad = self.ads.get(ad_id)
            chunk = corpus.chunks[chunk_i]
            if ad is None:
                continue
            hits.append(Hit(
                ad_id=ad_id, title=ad.title, company=ad.company,
                location=ad.location, url=ad.url, score=score,
                evidence=chunk.text.strip()[:420],
                boilerplate=is_boilerplate(chunk.text),
                why_relevant=why(ad.title, ad.text),
            ))
        return hits

    def compare(self, query: str, k: int = 5) -> dict:
        """Every variant on the same query.

        This is the part worth having. The evaluation says bm25 wins keyword
        queries and loses semantic ones. Here you can watch it happen.
        """
        out = {}
        for name in VARIANTS:
            out[name] = [h.as_dict() for h in self.search(query, name, k)]
        return out
