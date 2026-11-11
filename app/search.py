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

from eval.metrics import ndcg_at_k, recall_at_k, reciprocal_rank  # noqa: E402
from eval.retrievers import RRF_K, Corpus, load_ads  # noqa: E402
from jobfit.relevance import is_boilerplate, why  # noqa: E402

POOL = 80
QUERIES = os.path.join(ROOT, "eval", "queries.json")
RESULTS = os.path.join(ROOT, "eval", "results.md")
QUERY_VECTORS = os.path.join(ROOT, "data", "eval_query_vectors.npz")

# name -> (needs the filtered corpus, method, downweight boilerplate)
# Same seven the evaluation reports, in the same order.
VARIANTS: dict[str, tuple[bool, str, bool]] = {
    "dense (baseline)": (False, "dense", False),
    "bm25 only": (False, "bm25", False),
    "hybrid rrf": (False, "hybrid", False),
    "hybrid + boilerplate": (False, "hybrid", True),
    "hybrid + relevance filter": (True, "hybrid", False),
    "hybrid + filter + boilerplate": (True, "hybrid", True),
    "hybrid + filter + boilerplate + rerank": (True, "rerank", True),
}

DEFAULT_VARIANT = "hybrid + filter + boilerplate + rerank"


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
            order = np.argsort(-dense, kind="stable")[:POOL]
            return [(int(i), float(dense[i])) for i in order]

        # hybrid: reciprocal rank fusion, the same constant the evaluation uses
        dense_rank = [int(i) for i in np.argsort(-dense, kind="stable")[:POOL]]
        lex_rank = [i for i, _ in corpus.bm25.top(query, POOL)]
        fused: dict[int, float] = {}
        for ranking in (dense_rank, lex_rank):
            for rank, i in enumerate(ranking, start=1):
                fused[int(i)] = fused.get(int(i), 0.0) + 1.0 / (RRF_K + rank)
        ordered = sorted(fused.items(), key=lambda kv: -kv[1])

        if method != "rerank":
            return ordered

        # The cross-encoder reads query and chunk together, but only for the
        # 61 queries whose scores are committed. A query typed in by hand has
        # none, and the honest answer is to say so rather than quietly serve
        # the un-reranked order under a reranked label.
        from jobfit.rerank import SHORTLIST, RerankUnavailable, score

        shortlist = [i for i, _ in ordered][:SHORTLIST]
        chunks = [corpus.chunks[i] for i in shortlist]
        try:
            ce = score(query, [c.chunk_id for c in chunks],
                       [c.for_embedding() for c in chunks])
        except RerankUnavailable as err:
            raise RerankUnavailable(str(err)) from err
        if boiler:
            ce = ce * np.array([corpus.boiler[i] for i in shortlist], dtype="float32")
        return [(shortlist[j], float(ce[j])) for j in range(len(shortlist))]

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

    # --- the evaluation's own answers --------------------------------------

    def _queries(self) -> dict:
        if not hasattr(self, "_qcache"):
            with open(QUERIES, encoding="utf-8") as f:
                data = json.load(f)
            self._qcache = {q["query"]: q for q in data["queries"]}
            self._families = data["families"]
        return self._qcache

    def gold(self, query: str) -> set:
        """The ads a human marked correct for this query, if it is one of the
        61. Empty for anything typed in, and the interface says so rather than
        implying a score it cannot compute."""
        q = self._queries().get(query)
        return set(q["gold"]) if q else set()

    def family_of(self, query: str) -> str | None:
        q = self._queries().get(query)
        return q["family"] if q else None

    def score_run(self, ranked: list, gold: set, k: int = 10) -> dict:
        if not gold:
            return {}
        return {
            "recall_at_10": round(recall_at_k(ranked, gold, k), 3),
            "mrr": round(reciprocal_rank(ranked, gold), 3),
            "ndcg_at_10": round(ndcg_at_k(ranked, gold, k), 3),
            "found": len([a for a in ranked[:k] if a in gold]),
            "total_gold": len(gold),
        }

    def corpus_stats(self) -> dict:
        boiler = sum(1 for c in self.plain.chunks if is_boilerplate(c.text))
        return {
            "ads": len(self.ads),
            "companies": len({a.company for a in self.ads.values() if a.company}),
            "chunks": len(self.plain.chunks),
            "chunks_after_filter": len(self.filtered.chunks),
            "ads_after_filter": len(self.filtered.kept_ads),
            "boilerplate_chunks": boiler,
            "boilerplate_pct": round(100 * boiler / max(len(self.plain.chunks), 1), 1),
            "queries": len(self._queries()),
            "families": self._families,
        }

    @staticmethod
    def results_tables() -> list[dict]:
        """The committed evaluation tables, parsed out of eval/results.md.

        Read from the file rather than recomputed, so the page cannot show a
        number the repository does not also publish.
        """
        if not os.path.exists(RESULTS):
            return []
        out, rows = [], []
        with open(RESULTS, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("|") and line.endswith("|"):
                    cells = [c.strip() for c in line.strip("|").split("|")]
                    if all(set(c) <= set("-: ") for c in cells):
                        continue
                    rows.append(cells)
                elif rows:
                    out.append({"header": rows[0], "rows": rows[1:]})
                    rows = []
        if rows:
            out.append({"header": rows[0], "rows": rows[1:]})
        return out

    def compare(self, query: str, k: int = 5) -> dict:
        """Every variant on the same query.

        This is the part worth having. The evaluation says bm25 wins keyword
        queries and loses semantic ones. Here you can watch it happen.
        """
        from jobfit.rerank import RerankUnavailable

        gold = self.gold(query)
        out = {}
        for name in VARIANTS:
            # Score over the full ten, then show the top k. Otherwise recall@10
            # would silently become recall@k and disagree with results.md.
            try:
                ranked = [h.ad_id for h in self.search(query, name, 10)]
                hits = [h.as_dict() for h in self.search(query, name, k)]
            except RerankUnavailable:
                out[name] = {"results": [], "metrics": {},
                             "unavailable": "no cross-encoder scores committed for "
                                            "this query, so it is not shown rather "
                                            "than shown wrong"}
                continue
            for h in hits:
                h["correct"] = h["ad_id"] in gold if gold else None
            out[name] = {"results": hits, "metrics": self.score_run(ranked, gold)}
        return out
