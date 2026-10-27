"""Measure the relevance filter against the hand labels.

    python -m eval.run_filter

Sweeps the threshold and prints precision and recall at each value, then lists
every ad the filter and the labels disagree about, with the filter's reason.
Reading the disagreements is the point: a number alone does not say whether a
rule is wrong or a label is.
"""

from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from jobfit import relevance  # noqa: E402
from jobfit.corpus import is_relevant as old_filter  # noqa: E402

CORPUS = os.path.join(ROOT, "data", "eval_corpus.jsonl")
LABELS = os.path.join(ROOT, "eval", "labels.json")


def load():
    with open(CORPUS, encoding="utf-8") as f:
        ads = [json.loads(line) for line in f if line.strip()]
    with open(LABELS, encoding="utf-8") as f:
        labels = json.load(f)["labels"]
    return ads, labels


def prf(kept: list, wanted: set) -> tuple[float, float, float]:
    k = {a["id"] for a in kept}
    tp = len(k & wanted)
    prec = tp / len(k) if k else 0.0
    rec = tp / len(wanted) if wanted else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def main() -> None:
    ads, labels = load()
    strict = {i for i, v in labels.items() if v == "yes"}
    lenient = {i for i, v in labels.items() if v in ("yes", "borderline")}

    print("%d ads in the frozen snapshot" % len(ads))
    print("hand labels: %d yes, %d borderline, %d no\n"
          % (len(strict), len(lenient) - len(strict), len(ads) - len(lenient)))

    rows = []
    old_kept = [a for a in ads if old_filter(a["title"], a["text"])]
    rows.append(("substring filter", old_kept))
    for th in (1, 2, 3, 4):
        kept = [a for a in ads
                if relevance.is_relevant(a["title"], a["text"], th)]
        rows.append(("title+body, threshold %d" % th, kept))

    print("%-26s %5s %18s %18s" % ("", "kept", "strict (yes only)",
                                   "lenient (+borderline)"))
    print("%-26s %5s %6s %6s %5s %6s %6s %5s"
          % ("", "", "prec", "rec", "f1", "prec", "rec", "f1"))
    for name, kept in rows:
        ps, rs, fs = prf(kept, strict)
        pl, rl, fl = prf(kept, lenient)
        print("%-26s %5d %6.2f %6.2f %5.2f %6.2f %6.2f %5.2f"
              % (name, len(kept), ps, rs, fs, pl, rl, fl))

    th = relevance.THRESHOLD
    kept = {a["id"] for a in ads
            if relevance.is_relevant(a["title"], a["text"], th)}
    print("\ndisagreements at the shipped threshold of %d" % th)
    by_id = {a["id"]: a for a in ads}

    missed = sorted(lenient - kept)
    extra = sorted(kept - lenient)
    print("\n  dropped but labelled a real role (%d)" % len(missed))
    for i in missed:
        a = by_id[i]
        print("    %-46s %s" % (a["title"][:46],
                                relevance.why(a["title"], a["text"])[:70]))
    print("\n  kept but labelled a different job (%d)" % len(extra))
    for i in extra:
        a = by_id[i]
        print("    %-46s %s" % (a["title"][:46],
                                relevance.why(a["title"], a["text"])[:70]))


if __name__ == "__main__":
    main()
