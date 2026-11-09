"""Cross-encoder reranking of the top candidates.

Retrieval scores a query against a chunk without the two ever meeting: the
chunk was embedded months before the query existed. A cross-encoder reads both
together, so it can notice that "senior" in the query and "graduate" in the ad
are the same word family pointing opposite ways. It is far too slow to run over
926 chunks, which is why it reranks a shortlist instead.

Scores are cached to disk and committed, for the same reason the chunk vectors
are: the evaluation has to replay in CI with numpy and nothing else. Scoring
needs torch and happens once, here, on a machine with room for it.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = os.path.join(ROOT, "data", "rerank_scores.json")
MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# How many chunks the cross-encoder sees. Past about 40 the extra candidates
# are ones the first stage ranked badly for good reason, and every one costs a
# forward pass.
SHORTLIST = 40


class RerankUnavailable(RuntimeError):
    """Asked to score something not in the cache, with no model to fall back on."""


def _load() -> dict:
    if os.path.exists(CACHE):
        with open(CACHE, encoding="utf-8") as f:
            return json.load(f)
    return {}


_CACHE: dict = _load()


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(MODEL, max_length=512)


def score(query: str, chunk_ids: list[str], texts: list[str],
          allow_model: bool = False) -> np.ndarray:
    """Relevance of each chunk to the query, higher is better.

    Reads the committed cache first. Only touches the model when explicitly
    allowed, so a CI run cannot quietly start downloading weights.
    """
    cached = _CACHE.get(query, {})
    missing = [i for i, c in enumerate(chunk_ids) if c not in cached]

    if missing:
        if not allow_model:
            raise RerankUnavailable(
                f"{len(missing)} chunks for {query!r} are not in "
                f"data/rerank_scores.json. Rebuild it with "
                f"python -m eval.build_rerank_cache"
            )
        pairs = [(query, texts[i]) for i in missing]
        fresh = _model().predict(pairs, show_progress_bar=False)
        cached = dict(cached)
        for i, s in zip(missing, fresh):
            cached[chunk_ids[i]] = float(s)
        _CACHE[query] = cached

    return np.array([cached[c] for c in chunk_ids], dtype="float32")


def save() -> str:
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    with open(CACHE, "w", encoding="utf-8") as f:
        json.dump(_CACHE, f, sort_keys=True, separators=(",", ":"))
    return CACHE


def cached_queries() -> int:
    return len(_CACHE)
