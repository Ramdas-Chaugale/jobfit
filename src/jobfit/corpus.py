"""Job ads in, clean documents out.

Three sources. Two need no key at all, so the project runs and the evaluation
reproduces on a fresh clone with nothing to sign up for.

    python -m jobfit.corpus                      # Arbeitnow, no key, ~300 ads
    python -m jobfit.corpus --source remoteok    # no key, ~100 ads
    python -m jobfit.corpus --source adzuna      # UK wide, free key needed

Only ads that actually mention AI, ML or data work are kept. A retrieval system
answering questions about machine learning roles has no use for a corpus that
is mostly sales and logistics.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data")
CORPUS = os.path.join(DATA, "corpus.jsonl")

ARBEITNOW = "https://www.arbeitnow.com/api/job-board-api"
REMOTEOK = "https://remoteok.com/api"
ADZUNA = "https://api.adzuna.com/v1/api/jobs/gb/search/1"

ADZUNA_QUERIES = [
    "machine learning engineer", "ai engineer", "data scientist",
    "mlops engineer", "data engineer", "llm engineer",
]

# An ad has to mention at least one of these to join the corpus.
RELEVANT = [
    "machine learning", "deep learning", "data scien", "data engineer",
    "ai engineer", "mlops", "artificial intelligence", "neural network",
    "pytorch", "tensorflow", "llm", "large language model", "nlp",
    "computer vision", "data analyst", "analytics engineer", " ai ",
]

MIN_CHARS = 400


@dataclass(frozen=True)
class Ad:
    """One job ad, normalised. The id is what a citation points at."""
    id: str
    title: str
    company: str
    location: str
    posted: str
    source: str
    url: str
    text: str
    salary_min: float | None = None
    salary_max: float | None = None

    def as_document(self) -> str:
        """The string that gets embedded.

        Title, company and location go in deliberately. A question like "who is
        hiring for PyTorch in Manchester" needs them retrievable, and keeping
        them in the embedded text beats bolting on a metadata filter later.
        """
        head = [self.title]
        if self.company:
            head.append("at " + self.company)
        if self.location:
            head.append("in " + self.location)
        return ". ".join(head) + ".\n\n" + self.text


def tidy(raw: str) -> str:
    """Unescape, strip tags, collapse whitespace.

    Order matters. Arbeitnow double-escapes, so the tags arrive as &lt;p&gt;
    and stripping before unescaping would leave the markup behind as text.
    """
    s = html.unescape(html.unescape(raw or ""))
    s = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", s,
               flags=re.S | re.I)
    s = re.sub(r"<[^>]+>", " ", s)
    return re.sub(r"[ \t ]+", " ", s).replace("\r", "").strip()


def is_relevant(title: str, text: str) -> bool:
    blob = (title + " " + text).lower()
    return any(w in blob for w in RELEVANT)


def _keep(ads: list[Ad]) -> list[Ad]:
    return [a for a in ads
            if len(a.text) >= MIN_CHARS and is_relevant(a.title, a.text)]


def _get_json(url: str, timeout: int = 60):
    req = urllib.request.Request(url, headers={"User-Agent": "jobfit"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def from_arbeitnow() -> list[Ad]:
    rows = _get_json(ARBEITNOW).get("data", [])
    print("  arbeitnow returned %d ads" % len(rows))
    out = []
    for j in rows:
        ts = j.get("created_at")
        posted = ""
        if ts:
            posted = datetime.fromtimestamp(
                int(ts), tz=timezone.utc).strftime("%Y-%m-%d")
        out.append(Ad(
            id="arbeitnow:" + (j.get("slug") or ""),
            title=tidy(j.get("title")),
            company=tidy(j.get("company_name")),
            location=tidy(j.get("location")) or ("Remote" if j.get("remote") else ""),
            posted=posted,
            source="arbeitnow",
            url=j.get("url") or "",
            text=tidy(j.get("description")),
        ))
    return out


def from_remoteok() -> list[Ad]:
    rows = _get_json(REMOTEOK)
    # The first row is a legal notice, not a job.
    rows = [r for r in rows if isinstance(r, dict) and r.get("position")]
    print("  remoteok returned %d ads" % len(rows))
    out = []
    for j in rows:
        out.append(Ad(
            id="remoteok:" + str(j.get("id") or j.get("slug") or ""),
            title=tidy(j.get("position")),
            company=tidy(j.get("company")),
            location=tidy(j.get("location")) or "Remote",
            posted=(j.get("date") or "")[:10],
            source="remoteok",
            url=j.get("url") or "",
            text=tidy(j.get("description")),
            salary_min=j.get("salary_min"),
            salary_max=j.get("salary_max"),
        ))
    return out


def from_adzuna(days: int, per_query: int = 50) -> list[Ad]:
    app_id = os.environ.get("ADZUNA_APP_ID")
    app_key = os.environ.get("ADZUNA_APP_KEY")
    creds = os.path.join(ROOT, "credentials.json")
    if (not app_id or not app_key) and os.path.exists(creds):
        with open(creds, encoding="utf-8") as f:
            c = json.load(f)
        app_id, app_key = c.get("app_id"), c.get("app_key")
    if not app_id or not app_key:
        raise SystemExit(
            "Adzuna needs a free key. Register at "
            "https://developer.adzuna.com/signup and put it in "
            "credentials.json, or just run without --source for the "
            "keyless default.")

    seen: dict[str, Ad] = {}
    for q in ADZUNA_QUERIES:
        params = {"app_id": app_id, "app_key": app_key,
                  "results_per_page": per_query, "what": q,
                  "max_days_old": days, "content-type": "application/json"}
        try:
            rows = _get_json(ADZUNA + "?" + urllib.parse.urlencode(params))
            rows = rows.get("results", [])
        except Exception as e:
            print("  %-28s failed (%s)" % (q, type(e).__name__))
            continue
        print("  %-28s %3d ads" % (q, len(rows)))
        for j in rows:
            jid = "adzuna:" + str(j.get("id") or "")
            if jid in seen:
                continue
            seen[jid] = Ad(
                id=jid,
                title=tidy(j.get("title")),
                company=((j.get("company") or {}).get("display_name") or ""),
                location=((j.get("location") or {}).get("display_name") or ""),
                posted=(j.get("created") or "")[:10],
                source="adzuna",
                url=j.get("redirect_url") or "",
                text=tidy(j.get("description")),
                salary_min=j.get("salary_min"),
                salary_max=j.get("salary_max"),
            )
        time.sleep(0.4)
    return list(seen.values())


def save(ads: list[Ad], path: str = CORPUS) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for a in ads:
            f.write(json.dumps(asdict(a), ensure_ascii=False) + "\n")


def load(path: str = CORPUS) -> list[Ad]:
    if not os.path.exists(path):
        raise SystemExit("no corpus yet, run: python -m jobfit.corpus")
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(Ad(**json.loads(line)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=("arbeitnow", "remoteok", "adzuna"),
                    default="arbeitnow")
    ap.add_argument("--days", type=int, default=14, help="adzuna only")
    args = ap.parse_args()

    print("fetching from %s" % args.source)
    if args.source == "arbeitnow":
        raw = from_arbeitnow()
    elif args.source == "remoteok":
        raw = from_remoteok()
    else:
        raw = from_adzuna(args.days)

    ads = _keep(raw)
    save(ads)

    print()
    print("%d ads fetched, %d kept as AI, ML or data roles"
          % (len(raw), len(ads)))
    if not ads:
        print("nothing kept, so nothing written")
        return
    lengths = sorted(len(a.text) for a in ads)
    print("  median length : %d characters" % lengths[len(lengths) // 2])
    print("  shortest      : %d" % lengths[0])
    print("  longest       : %d" % lengths[-1])
    print("  companies     : %d distinct" % len({a.company for a in ads}))
    print()
    print("wrote %s" % os.path.relpath(CORPUS, ROOT))


if __name__ == "__main__":
    main()
