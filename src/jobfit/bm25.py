"""BM25, written out rather than imported.

The eval needs a lexical baseline to compare the embeddings against. Without
one there is no way to tell whether a vector index earns its keep, and on a
corpus this small it might not.

Standard Okapi BM25 with the usual k1 and b. No dependency: it is forty lines,
and a dependency that CI has to install to run the tests is a worse trade.
"""

from __future__ import annotations

import math
import re
from collections import Counter

K1 = 1.5    # term frequency saturation
B = 0.75    # length normalisation

_WORD = re.compile(r"[a-z0-9+#.]+")


def tokens(text: str) -> list[str]:
    """Lowercase word split that keeps c++, c#, .net and node.js intact."""
    return [t.strip(".") for t in _WORD.findall(text.lower()) if t.strip(".")]


class BM25:
    def __init__(self, docs: list[str]):
        self.docs = [tokens(d) for d in docs]
        self.n = len(self.docs)
        self.lengths = [len(d) for d in self.docs]
        self.avg_len = (sum(self.lengths) / self.n) if self.n else 0.0
        self.freqs = [Counter(d) for d in self.docs]

        seen: Counter = Counter()
        for d in self.docs:
            seen.update(set(d))
        # The 1 + inside the log is why there is no flooring here. The older
        # form, log((n - df + 0.5) / (df + 0.5)), goes negative once a term is
        # in more than half the documents, which ranks a document down for
        # containing a word the query asked for. This form cannot.
        self.idf = {
            t: math.log(1 + (self.n - df + 0.5) / (df + 0.5))
            for t, df in seen.items()
        }

    def scores(self, query: str) -> list[float]:
        q = tokens(query)
        out = [0.0] * self.n
        for t in q:
            idf = self.idf.get(t)
            if not idf:
                continue
            for i, f in enumerate(self.freqs):
                tf = f.get(t, 0)
                if not tf:
                    continue
                norm = K1 * (1 - B + B * self.lengths[i] / self.avg_len)
                out[i] += idf * tf * (K1 + 1) / (tf + norm)
        return out

    def top(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        sc = self.scores(query)
        order = sorted(range(self.n), key=lambda i: -sc[i])[:k]
        return [(i, sc[i]) for i in order if sc[i] > 0]
