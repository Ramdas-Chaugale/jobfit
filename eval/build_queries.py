"""Build the evaluation set once, so every retriever is scored on the same thing.

    python -m eval.build_queries        # writes eval/queries.json

Three families, because each answers a different question and each has a
different weakness. Keeping them apart is what stops one easy family from
hiding a failure in a hard one.

known_item
    Generated. Take a real requirement line out of an ad, strip it to content
    words, and ask for it back. One correct answer: the ad it came from.
    Words that appear in the title are removed first, otherwise the query
    carries the answer and every retriever looks good.

keyword_topical
    Generated. "pytorch", "spark airflow" and so on, with the correct answers
    being every relevant ad containing that term as a whole word. Honest
    warning: the correct answers are defined by a keyword, so a keyword
    retriever is being marked against its own definition of right. It is here
    as a floor, not as a fair fight.

semantic_topical
    Hand written, hand marked, and deliberately sharing almost no words with
    the ads. "keeping models running in production once they are built" has to
    reach an MLOps ad that never uses that phrase. This is the family that says
    whether embeddings are worth their weight.
"""

from __future__ import annotations

import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from jobfit.bm25 import tokens  # noqa: E402
from jobfit.relevance import is_boilerplate  # noqa: E402

CORPUS = os.path.join(ROOT, "data", "eval_corpus.jsonl")
LABELS = os.path.join(ROOT, "eval", "labels.json")
OUT = os.path.join(ROOT, "eval", "queries.json")

STOP = set(["a", "an", "and", "or", "the", "to", "of", "in", "for", "with", "on", "at", "by", "as", "is", "are", "be", "been", "we", "you", "your", "our", "their", "they", "it", "its", "this", "that", "these", "those", "will", "would", "can", "could", "may", "have", "has", "had", "do", "does", "from", "who", "whom", "which", "what", "when", "where", "why", "how", "not", "no", "if", "then", "than", "so", "such", "very", "more", "most", "other", "others", "into", "about", "over", "under", "out", "up", "down", "also", "both", "each", "any", "all", "some", "one", "two", "three", "new", "per", "via", "etc", "using", "use", "used", "within", "across", "while", "during", "after", "before", "between", "among", "own", "same", "too", "only", "just", "e.g", "i.e", "us", "them", "he", "she", "his", "her", "there", "here", "being", "am", "was", "were", "doing", "job", "role", "team", "work", "working", "experience", "years", "year", "ability", "strong", "good", "great", "excellent", "plus", "nice", "must", "should", "want", "wants", "looking", "seeking", "join", "joining", "candidate", "applicant", "position", "opportunity", "company", "office", "remote", "hybrid", "m", "f", "d", "w", "x", "genders", "mwd", "fmd"])

# Terms specific enough to be worth a topical query of their own.
TOPICS = [
    "pytorch", "tensorflow", "kubernetes", "docker", "aws", "azure",
    "spark", "airflow", "sql", "python", "llm", "rag", "langchain",
    "mlops", "computer vision", "nlp", "reinforcement learning",
    "terraform", "kafka", "databricks", "fastapi",
]

# query -> slug prefixes of the ads that should come back.
SEMANTIC = [
    ("something I could do while still a student", [
        "machine-learning-engineer-intern-paris-paris",
        "autonomy-engineer-intern-reinforcement-learning",
        "autonomy-engineer-intern-simulation-robotics",
        "autonomy-engineer-intern-autonomy-infrastructure",
        "working-student-ai-software-engineer-langchain",
        "working-student-interceptor-data-analytics",
        "clinical-data-analyst-intern-paris",
        "graduate-data-scientist-cyber-analytics",
    ]),
    ("keeping models running in production once they are built", [
        "senior-mlops-engineer-buyer",
        "ai-platform-engineer-eindhoven",
        "ai-ops-engineer-business-operations",
        "engineering-manager-data-platform",
    ]),
    ("academic research rather than product work", [
        "ai-research-scientist-scientific-evaluation",
        "centrale-lyon-post-doctorant",
    ]),
    ("building a chatbot that answers from company documents", [
        "ki-entwickler-llm-rag-agenten",
        "working-student-ai-software-engineer-langchain",
        "senior-generative-ai-engineer",
        "ai-workflows-engineer",
    ]),
    ("teaching a machine to act by trial and error", [
        "senior-reinforcement-learning-engineer",
        "autonomy-engineer-intern-reinforcement-learning",
    ]),
    ("a job based in Britain", [
        "machine-learning-engineer-research-participants-united-kingdom",
        "data-scientist-research-participants-united-kingdom",
        "data-engineer-research-participants-united-kingdom",
        "data-analyst-research-participants-united-kingdom",
        "graduate-data-scientist-cyber-analytics",
        "senior-manager-ai-architect-london",
        "ai-ops-engineer-business-operations-london",
        "director-engineering-ai-search-london",
    ]),
    ("a leadership job rather than hands on coding", [
        "engineering-manager-data-platform",
        "director-engineering-ai-search-london",
        "senior-manager-ai-architect-london",
        "staff-product-manager-agentic-ai-platform",
    ]),
    ("moving and cleaning large amounts of data rather than training models", [
        "data-engineer-berlin",
        "data-engineer-nijverdal",
        "data-engineer-research-participants-united-kingdom",
        "engineering-manager-data-platform",
    ]),
]


def load():
    with open(CORPUS, encoding="utf-8") as f:
        ads = [json.loads(line) for line in f if line.strip()]
    with open(LABELS, encoding="utf-8") as f:
        labels = json.load(f)["labels"]
    relevant = [a for a in ads if labels[a["id"]] in ("yes", "borderline")]
    return ads, relevant


def resolve(prefix: str, ads: list[dict]) -> str:
    hits = [a["id"] for a in ads if a["id"].startswith("arbeitnow:" + prefix)]
    if len(hits) != 1:
        raise SystemExit("%r matched %d ads, need exactly 1"
                         % (prefix, len(hits)))
    return hits[0]


def has_word(word: str, text: str) -> bool:
    return re.search(r"(?<![a-z0-9])%s(?![a-z0-9])" % re.escape(word),
                     text.lower()) is not None


def requirement_line(ad: dict) -> str | None:
    """The longest line that actually states a requirement.

    Taking the longest line outright does not work. In a lot of these ads the
    longest line is the equal opportunity notice, so the first version of this
    produced queries like "qualified applicants receive consideration
    employment without regard race colour religion". Every ad carries that
    paragraph, so the query has no single right answer and the known item
    score was measuring nothing.

    is_boilerplate is the same test retrieval uses to downweight a chunk, so a
    passage rejected here is a passage demoted there.
    """
    lines = [re.sub(r"^\s*(?:[-*•–]|\d+[.)])\s*", "", ln).strip()
             for ln in ad["text"].split("\n")]
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", ad["text"])]
    for pool in (lines, sents):
        usable = [x for x in pool
                  if 60 <= len(x) <= 300 and not is_boilerplate(x)]
        if usable:
            return max(usable, key=len)
    return None


def known_item_query(ad: dict) -> str | None:
    line = requirement_line(ad)
    if not line:
        return None
    # Drop anything already in the title or company name, or the query would
    # hand over the answer.
    give_away = set(tokens(ad["title"] + " " + ad["company"]))
    words, seen = [], set()
    for t in tokens(line):
        if t in STOP or t in give_away or t in seen or len(t) < 3:
            continue
        seen.add(t)
        words.append(t)
    return " ".join(words[:10]) if len(words) >= 4 else None


def main() -> None:
    ads, relevant = load()
    queries = []

    # Two ads can still produce the same query, because some of these ads are
    # the same template with the job title swapped. Splitting one query across
    # two entries would mark a retriever wrong for returning the other correct
    # ad, so identical queries share one entry and both ads count as right.
    by_query: dict[str, list[str]] = {}
    for ad in relevant:
        q = known_item_query(ad)
        if q:
            by_query.setdefault(q, []).append(ad["id"])
    for q, gold in by_query.items():
        queries.append({"family": "known_item", "query": q,
                        "gold": sorted(gold)})

    for topic in TOPICS:
        gold = [a["id"] for a in relevant
                if has_word(topic, a["title"] + " " + a["text"])]
        if 1 <= len(gold) <= 15:
            queries.append({"family": "keyword_topical", "query": topic,
                            "gold": sorted(gold)})

    for q, prefixes in SEMANTIC:
        queries.append({"family": "semantic_topical", "query": q,
                        "gold": sorted(resolve(p, ads) for p in prefixes)})

    doc = {
        "corpus": "data/eval_corpus.jsonl",
        "labels": "eval/labels.json",
        "built_by": "python -m eval.build_queries",
        "families": {
            "known_item": "one correct ad, query is a stripped requirement line",
            "keyword_topical": ("correct answers defined by a keyword, so it "
                                "favours keyword retrieval by construction"),
            "semantic_topical": "hand marked, little word overlap with the ads",
        },
        "counts": {},
        "queries": queries,
    }
    for q in queries:
        doc["counts"][q["family"]] = doc["counts"].get(q["family"], 0) + 1

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print("wrote %s" % os.path.relpath(OUT, ROOT))
    for k, v in doc["counts"].items():
        print("  %-18s %3d queries" % (k, v))
    gsz = [len(q["gold"]) for q in queries]
    print("  gold per query     %.1f average, %d largest"
          % (sum(gsz) / len(gsz), max(gsz)))
    print()
    print("examples")
    for fam in ("known_item", "keyword_topical", "semantic_topical"):
        ex = next(q for q in queries if q["family"] == fam)
        print("  [%s] %r -> %d gold"
              % (fam, ex["query"][:64], len(ex["gold"])))


if __name__ == "__main__":
    main()
