"""Freeze the interface into a static site.

The server build needs a persistent process: the corpus takes about five
seconds to assemble and the ONNX model nine more, which is fine for a long
running instance and terrible for a serverless function that cold starts on
every visit.

But the 61 evaluation queries are a closed set. Their answers never change, so
they can be computed once here and served as a file. That gives a public URL
with no cold start, no server and no running cost, at the price of being honest
that a visitor cannot type a query of their own.

    python -m app.export_static        # writes site/

Free text search still needs `python -m app.server`, and the page says so.
"""

from __future__ import annotations

import json
import os
import shutil

from app.search import DEFAULT_VARIANT, VARIANTS, SearchService

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SITE = os.path.join(ROOT, "site")
STATIC = os.path.join(ROOT, "app", "static")


def main() -> None:
    svc = SearchService()
    examples = svc.examples()

    bundle = {
        "static": True,
        "variants": list(VARIANTS),
        "default": DEFAULT_VARIANT,
        "examples": examples,
        "evaluation": {"corpus": svc.corpus_stats(), "tables": svc.results_tables()},
        "ads": {},
        "queries": {},
    }

    # Ad titles, companies and urls repeat across every variant and every
    # query. Holding them once and referencing by id roughly halves the file.
    def thin(h: dict, keep_evidence: bool) -> dict:
        bundle["ads"].setdefault(h["ad_id"], {
            "title": h["title"], "company": h["company"],
            "location": h["location"], "url": h["url"],
            "why_relevant": h["why_relevant"],
        })
        out = {"ad_id": h["ad_id"], "score": h["score"], "correct": h["correct"]}
        if keep_evidence:
            out["evidence"] = h["evidence"]
            out["boilerplate"] = h["boilerplate"]
        return out

    for n, ex in enumerate(examples, 1):
        q = ex["query"]
        gold = svc.gold(q)

        # The compare view lists titles only, so the evidence text is dead
        # weight there. The search view shows it, so it is kept there.
        compare = svc.compare(q, 5)
        for v in compare.values():
            v["results"] = [thin(h, False) for h in v["results"]]

        search = {}
        for name in VARIANTS:
            hits = [h.as_dict() for h in svc.search(q, name, 10)]
            for h in hits:
                h["correct"] = h["ad_id"] in gold if gold else None
            search[name] = [thin(h, True) for h in hits]

        bundle["queries"][q] = {
            "info": {"family": svc.family_of(q), "gold": sorted(gold),
                     "scored": bool(gold), "instant": True},
            "compare": compare,
            "search": search,
        }
        if n % 20 == 0:
            print(f"  {n}/{len(examples)} queries")

    os.makedirs(SITE, exist_ok=True)
    shutil.copy(os.path.join(STATIC, "index.html"), os.path.join(SITE, "index.html"))

    path = os.path.join(SITE, "data.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, separators=(",", ":"))

    size = os.path.getsize(path) / 1e6
    print(f"\nwrote site/data.json  {len(bundle['queries'])} queries, {size:.2f} MB")
    print("wrote site/index.html")


if __name__ == "__main__":
    main()
