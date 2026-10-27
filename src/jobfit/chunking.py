"""Split job ads into retrievable pieces.

A whole ad is the wrong unit. The median one here is 5,200 characters covering
the company blurb, the role, the requirements and the benefits. Embedding all
of that into one vector blurs the requirements, which is the part anyone is
actually asking about.

A single sentence is also wrong. "Experience with Kubernetes" retrieves well
and then tells the reader nothing about whose job it is or what else it needs.

So: split on blank lines and bullet boundaries, pack those pieces up to a
target size, and carry a little overlap so a requirement split across a
boundary still appears whole somewhere.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .corpus import Ad

TARGET = 700      # characters we aim for per chunk
MAX = 1100        # hard ceiling before forcing a split
OVERLAP = 120     # trailing characters repeated into the next chunk


@dataclass(frozen=True)
class Chunk:
    """One retrievable piece, with enough of its parent ad to cite it."""
    chunk_id: str
    ad_id: str
    title: str
    company: str
    location: str
    url: str
    position: int
    text: str

    def for_embedding(self) -> str:
        """Prepend the job title to every chunk.

        Without this a chunk that reads "5+ years with Spark and Airflow" is
        unattached to any role, and a query naming a job title cannot reach it.
        The title is short, so the cost in dilution is small next to what it
        buys in recall.
        """
        return "%s at %s. %s" % (self.title, self.company or "a company",
                                 self.text)


def _blocks(text: str) -> list[str]:
    """Break text at blank lines, then at bullet starts inside long runs."""
    rough = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    out: list[str] = []
    for b in rough:
        if len(b) <= MAX:
            out.append(b)
            continue
        # Long block. Bullets are the natural seam in a job ad.
        parts = re.split(r"\n(?=\s*(?:[-*•–]|\d+[.)])\s)", b)
        if len(parts) == 1:
            parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", b)
        buf = ""
        for p in parts:
            p = p.strip()
            if not p:
                continue
            if len(buf) + len(p) + 1 <= MAX:
                buf = (buf + "\n" + p).strip()
            else:
                if buf:
                    out.append(buf)
                buf = p[:MAX]
        if buf:
            out.append(buf)
    return out


def chunk_ad(ad: Ad) -> list[Chunk]:
    blocks = _blocks(ad.text)
    packed: list[str] = []
    buf = ""
    for b in blocks:
        if not buf:
            buf = b
        elif len(buf) + len(b) + 2 <= TARGET:
            buf = buf + "\n\n" + b
        else:
            packed.append(buf)
            # Carry a little context forward so a requirement that straddles
            # the seam is still complete in one of the two chunks.
            tail = buf[-OVERLAP:] if len(buf) > OVERLAP else buf
            tail = tail[tail.find(" ") + 1:] if " " in tail else ""
            buf = (tail + "\n\n" + b).strip() if tail else b
    if buf:
        packed.append(buf)

    return [
        Chunk(chunk_id="%s#%d" % (ad.id, i), ad_id=ad.id, title=ad.title,
              company=ad.company, location=ad.location, url=ad.url,
              position=i, text=t)
        for i, t in enumerate(packed)
    ]


def chunk_all(ads: list[Ad]) -> list[Chunk]:
    out: list[Chunk] = []
    for ad in ads:
        out.extend(chunk_ad(ad))
    return out
