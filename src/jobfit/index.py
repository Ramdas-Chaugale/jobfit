"""Embed the chunks and put them in a vector index.

    python -m jobfit.index build
    python -m jobfit.index search "who wants PyTorch and Kubernetes"

The embedding model is a parameter, not a hard-coded choice, because
eval/compare_models.py measures whether a bigger one is actually worth it on
this corpus. Picking one without measuring would be guessing.

Vectors are L2 normalised and the index uses inner product, which makes the
score plain cosine similarity. Exact search, no approximation: at a thousand
chunks an approximate index would add failure modes and save nothing.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict

import numpy as np

from .chunking import Chunk, chunk_all
from .corpus import load as load_corpus

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data")

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def paths(model: str) -> tuple[str, str]:
    """Index files are named after the model, so two can coexist."""
    tag = model.split("/")[-1].replace(".", "-")
    return (os.path.join(DATA, "index_%s.faiss" % tag),
            os.path.join(DATA, "chunks_%s.jsonl" % tag))


_model_cache: dict = {}


def encoder(model: str):
    if model not in _model_cache:
        from sentence_transformers import SentenceTransformer
        _model_cache[model] = SentenceTransformer(model, device="cpu")
    return _model_cache[model]


def embed(texts: list[str], model: str, batch: int = 64,
          progress: bool = False) -> np.ndarray:
    vecs = encoder(model).encode(
        texts, batch_size=batch, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=progress)
    return np.asarray(vecs, dtype="float32")


def build(model: str = DEFAULT_MODEL, quiet: bool = False) -> int:
    import faiss

    ads = load_corpus()
    chunks = chunk_all(ads)
    if not quiet:
        print("%d ads -> %d chunks" % (len(ads), len(chunks)))
        print("embedding with %s" % model)

    t0 = time.time()
    vecs = embed([c.for_embedding() for c in chunks], model,
                 progress=not quiet)
    took = time.time() - t0

    idx = faiss.IndexFlatIP(vecs.shape[1])
    idx.add(vecs)

    index_path, chunks_path = paths(model)
    os.makedirs(DATA, exist_ok=True)
    faiss.write_index(idx, index_path)
    with open(chunks_path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")

    if not quiet:
        print("  %d vectors, %d dimensions" % (idx.ntotal, vecs.shape[1]))
        print("  took %.1fs (%.0f chunks/second)" % (took, len(chunks) / took))
        print("  wrote %s" % os.path.relpath(index_path, ROOT))
    return idx.ntotal


class Index:
    """A built index, ready to search."""

    def __init__(self, model: str = DEFAULT_MODEL):
        import faiss

        index_path, chunks_path = paths(model)
        if not os.path.exists(index_path):
            raise SystemExit(
                "no index for %s yet, run: python -m jobfit.index build" % model)
        self.model = model
        self.index = faiss.read_index(index_path)
        self.chunks: list[Chunk] = []
        with open(chunks_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.chunks.append(Chunk(**json.loads(line)))

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        q = embed([query], self.model)
        scores, ids = self.index.search(q, min(k, len(self.chunks)))
        return [(self.chunks[i], float(s))
                for i, s in zip(ids[0], scores[0]) if i >= 0]

    def search_ads(self, query: str, k: int = 5,
                   pool: int = 25) -> list[tuple[str, float, list[Chunk]]]:
        """Retrieve chunks, then roll them up to one entry per job ad.

        Several chunks of the same ad often rank together. A reader wants a
        list of jobs, not the same job four times, so the ad takes its best
        chunk's score and keeps the chunks behind it for citation.
        """
        hits = self.search(query, pool)
        best: dict[str, list] = {}
        for chunk, score in hits:
            if chunk.ad_id not in best:
                best[chunk.ad_id] = [score, []]
            best[chunk.ad_id][0] = max(best[chunk.ad_id][0], score)
            best[chunk.ad_id][1].append(chunk)
        ranked = sorted(best.items(), key=lambda kv: -kv[1][0])[:k]
        return [(ad_id, sc, cs) for ad_id, (sc, cs) in ranked]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=("build", "search"))
    ap.add_argument("query", nargs="*")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("-k", type=int, default=5)
    args = ap.parse_args()

    if args.command == "build":
        build(args.model)
        return

    q = " ".join(args.query)
    if not q:
        raise SystemExit('give a query, e.g. index.py search "pytorch remote"')
    idx = Index(args.model)
    print('"%s"\n' % q)
    for ad_id, score, chunks in idx.search_ads(q, args.k):
        c = chunks[0]
        print("  %.3f  %s" % (score, c.title))
        print("         %s%s" % (c.company, (" | " + c.location) if c.location else ""))
        print("         %s" % c.text[:150].replace("\n", " "))
        print()


if __name__ == "__main__":
    main()
