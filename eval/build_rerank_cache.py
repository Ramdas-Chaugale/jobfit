"""Score every query against its shortlist once, and commit the result.

Needs torch. Run it after changing the queries, the chunking or the shortlist
size, then commit data/rerank_scores.json so the evaluation keeps replaying in
CI with numpy alone.

    python -m eval.build_rerank_cache
"""

from __future__ import annotations

import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from eval.retrievers import Corpus, hybrid_rerank  # noqa: E402
from jobfit.rerank import cached_queries, save  # noqa: E402


def main() -> None:
    with open(os.path.join(ROOT, "eval", "queries.json"), encoding="utf-8") as f:
        queries = [q["query"] for q in json.load(f)["queries"]]

    # Both corpora, because the filtered one holds a different chunk set and
    # the cross-encoder is keyed by chunk id.
    for filtered in (False, True):
        corpus = Corpus(filtered=filtered)
        label = "filtered" if filtered else "plain"
        start = time.time()
        for n, q in enumerate(queries, 1):
            for boiler in (False, True):
                hybrid_rerank(corpus, q, 10, boiler=boiler, allow_model=True)
            if n % 20 == 0:
                print(f"  {label}: {n}/{len(queries)}  {time.time() - start:.0f}s")
        print(f"  {label}: done in {time.time() - start:.0f}s")

    path = save()
    size = os.path.getsize(path) / 1e6
    print(f"\nwrote {os.path.relpath(path, ROOT)}  "
          f"{cached_queries()} queries, {size:.2f} MB")


if __name__ == "__main__":
    main()
