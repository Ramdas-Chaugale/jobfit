"""Score every retriever on every query family and write the table.

    python -m eval.run                      # writes eval/results.md
    python -m eval.run --inspect            # also show the baseline failures
    python -m eval.run --inspect --check    # fail if the file would change

CI runs this with --check, so the table in results.md and the one quoted in
the README are regenerated and compared on every push. A number cannot drift
away from the code that produced it without the build going red.

Nothing here chooses a winner. It prints the numbers per family, because a
variant that wins one family and loses another is the normal outcome and
averaging that away would hide the only interesting part.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from eval import retrievers  # noqa: E402
from eval.metrics import evaluate  # noqa: E402

QUERIES = os.path.join(ROOT, "eval", "queries.json")
RESULTS = os.path.join(ROOT, "eval", "results.md")
FAMILIES = ("known_item", "keyword_topical", "semantic_topical")


def load_queries() -> list[dict]:
    with open(QUERIES, encoding="utf-8") as f:
        return json.load(f)["queries"]


def score_variant(fn, corpus, queries: list[dict]) -> dict[str, dict]:
    out = {}
    for fam in FAMILIES:
        pairs = []
        for q in (x for x in queries if x["family"] == fam):
            ranked = fn(corpus, q["query"], 10)
            pairs.append((ranked, set(q["gold"])))
        out[fam] = evaluate(pairs)
    return out


def reachable(corpus, queries: list[dict]) -> dict[str, float]:
    """Share of correct answers still in the index after filtering.

    Filtering the corpus cannot raise recall above this, so it is the ceiling
    every number for a filtered variant sits under. Reporting it stops a drop
    in recall from being read as a ranking failure when it is a missing ad.
    """
    kept = {a.id for a in corpus.kept_ads}
    out = {}
    for fam in FAMILIES:
        got = tot = 0
        for q in (x for x in queries if x["family"] == fam):
            tot += len(q["gold"])
            got += len([g for g in q["gold"] if g in kept])
        out[fam] = got / tot if tot else 0.0
    return out


def inspect_baseline(corpus, queries: list[dict], n: int = 4) -> list[str]:
    """The worst semantic queries, with what came back instead."""
    by_ad = {a.id: a for a in retrievers.load_ads()}
    rows = []
    sem = [q for q in queries if q["family"] == "semantic_topical"]
    scored = []
    for q in sem:
        ranked = retrievers.dense(corpus, q["query"], 10)
        hit = len(set(ranked[:5]) & set(q["gold"])) / min(5, len(q["gold"]))
        scored.append((hit, q, ranked))
    for hit, q, ranked in sorted(scored, key=lambda x: x[0])[:n]:
        rows.append("")
        rows.append("**%s**  (%.0f%% of the top 5 correct)"
                    % (q["query"], 100 * hit))
        rows.append("")
        for r in ranked[:4]:
            mark = "ok " if r in set(q["gold"]) else "no "
            rows.append("- `%s` %s" % (mark, by_ad[r].title.strip()[:70]))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--check", action="store_true",
                    help="fail if results.md would change, for CI")
    args = ap.parse_args()

    queries = load_queries()
    counts = {f: sum(1 for q in queries if q["family"] == f) for f in FAMILIES}
    print("%d queries: %s" % (len(queries), ", ".join(
        "%d %s" % (v, k) for k, v in counts.items())))

    print("\nbuilding the unfiltered corpus")
    plain = retrievers.Corpus(filtered=False)
    print("  %d ads, %d chunks" % (len(plain.kept_ads), len(plain.chunks)))
    print("building the filtered corpus")
    filtered = retrievers.Corpus(filtered=True)
    print("  %d ads, %d chunks" % (len(filtered.kept_ads),
                                   len(filtered.chunks)))
    boiler = int(sum(1 for b in plain.boiler if b < 1.0))
    print("  %d of %d chunks carry no requirement signal (%.0f%%)"
          % (boiler, len(plain.chunks), 100 * boiler / len(plain.chunks)))

    ceiling = reachable(filtered, queries)
    print("\nrecall ceiling for filtered variants: " + ", ".join(
        "%s %.2f" % (k.split("_")[0], v) for k, v in ceiling.items()))

    results = {}
    for name, (needs_filter, fn) in retrievers.VARIANTS.items():
        t0 = time.time()
        results[name] = score_variant(fn, filtered if needs_filter else plain,
                                      queries)
        print("  scored %-32s %4.1fs" % (name, time.time() - t0))

    header = ("| variant | known recall@10 | known mrr | keyword recall@10 "
              "| keyword mrr | semantic recall@10 | semantic mrr |")
    sep = "|---|---:|---:|---:|---:|---:|---:|"
    rows = [header, sep]
    for name, per in results.items():
        cells = []
        for f in FAMILIES:
            cells.append("%.3f" % per[f]["recall@10"])
            cells.append("%.3f" % per[f]["mrr"])
        rows.append("| %s | %s |" % (name, " | ".join(cells)))

    print()
    for r in rows:
        print(r)

    base = results["dense (baseline)"]
    best_sem = max(results.items(),
                   key=lambda kv: kv[1]["semantic_topical"]["recall@10"])

    out = [
        "# Results",
        "",
        "Produced by `python -m eval.run` over the frozen snapshot in "
        "`data/eval_corpus.jsonl`, which is 135 ads. Rebuilding the live "
        "corpus changes the ads, so the snapshot is committed and the "
        "numbers below refer only to it.",
        "",
        "## The corpus filter",
        "",
        "Measured in `python -m eval.run_filter` against 135 hand labels in "
        "`eval/labels.json`.",
        "",
        "| filter | ads kept | precision | recall |",
        "|---|---:|---:|---:|",
        "| substring match, the first version | 135 | 0.21 | 1.00 |",
        "| title decides, body breaks ties | 37 | 0.76 | 1.00 |",
        "",
        "Precision and recall are against the 28 ads labelled plainly an AI, "
        "ML or data role. The first filter kept everything, because a single "
        "mention of AI passed an ad and 121 of the 135 mention it. It also "
        "matched `llm` inside the word fulfillment.",
        "",
        "## Retrieval",
        "",
        "%d queries in three families. Every retriever returns ten ads, "
        "scored the same way." % len(queries),
        "",
    ] + rows + [
        "",
        "Recall ceiling for the two filtered variants, since filtering removes "
        "three ads that are correct answers: " + ", ".join(
            "%s %.2f" % (k.split("_")[0], v) for k, v in ceiling.items())
        + ". Their recall cannot pass that and a shortfall against the "
          "unfiltered variants is partly a missing ad, not a worse ranking.",
        "",
        "## Reading it",
        "",
        "- **bm25 beats the embeddings on the keyword family.** It would be "
        "strange if it did not: the correct answers for that family are "
        "defined by a keyword. That family is a floor, not a verdict.",
        "- **the semantic family is the real test**, and the baseline scores "
        "%.3f recall@10 there against %.3f for %s."
        % (base["semantic_topical"]["recall@10"],
           best_sem[1]["semantic_topical"]["recall@10"], best_sem[0]),
        "- **%d%% of chunks carry no requirement signal**, which is the About "
        "Us problem the first README described. Downweighting them is the "
        "cheapest of the fixes tried here." % round(100 * boiler
                                                    / len(plain.chunks)),
        "",
        "## Still missing",
        "",
        "Answer generation, faithfulness of a generated answer against its "
        "citations, a reranker, and anything served over HTTP. The numbers "
        "here cover retrieval only.",
        "",
    ]

    if args.inspect:
        out += ["## What the baseline gets wrong", ""] + inspect_baseline(
            plain, queries)

    fresh = "\n".join(out) + "\n"
    if args.check:
        with open(RESULTS, encoding="utf-8") as f:
            committed = f.read()
        if committed != fresh:
            print("\n%s does not match a fresh run."
                  % os.path.relpath(RESULTS, ROOT))
            print("Regenerate it with: python -m eval.run --inspect")
            raise SystemExit(1)
        print("\n%s matches a fresh run" % os.path.relpath(RESULTS, ROOT))
        return

    with open(RESULTS, "w", encoding="utf-8") as f:
        f.write(fresh)
    print("\nwrote %s" % os.path.relpath(RESULTS, ROOT))


if __name__ == "__main__":
    main()
