"""Draw the charts in the README from the numbers in eval/results.md.

The charts are generated from the results file rather than drawn by hand, so a
chart cannot quietly disagree with the table above it. Re-run this after any
change to the evaluation and the pictures move with the numbers.

    python -m eval.make_charts

Writes a light and a dark version of each chart into docs/. The README picks
between them with <picture>, so they read on either GitHub theme.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless, and it has to be set before pyplot loads

import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "eval" / "results.md"
DOCS = ROOT / "docs"

FAMILIES = ["known", "keyword", "semantic"]

THEMES = {
    "light": {"fg": "#1b1b1b", "muted": "#6b6b6b", "grid": "#dddddd", "bg": "#ffffff"},
    "dark":  {"fg": "#e6e6e6", "muted": "#9a9a9a", "grid": "#3a3a3a", "bg": "#0d1117"},
}

# Grey for the baseline, one accent for the variant that wins. Colour is used
# to say something, not to decorate.
BASE = "#9aa0a6"     # the baseline, deliberately dull
MID = "#5b8def"
BEST = "#1b6b4a"     # the variant that wins
WARN = "#c0504d"

# One colour per variant, so six bars in a group can actually be told apart.
# bm25 sits outside the hybrid ramp because it is a different approach, not a
# step along the same one.
VARIANT_COLOURS = {
    "dense (baseline)": BASE,
    "bm25 only": "#d98032",
    "hybrid rrf": "#bfd3f7",
    "hybrid + boilerplate": "#8fb2f2",
    "hybrid + relevance filter": "#4a7fe0",
    "hybrid + filter + boilerplate": "#3f9e74",
    "hybrid + filter + boilerplate + rerank": BEST,
}


def tables(text: str) -> list[list[list[str]]]:
    """Every markdown table in the file, as rows of cells."""
    out, current = [], []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if all(set(c) <= set("-: ") for c in cells):
                continue          # the ---|--- separator row
            current.append(cells)
        elif current:
            out.append(current)
            current = []
    if current:
        out.append(current)
    return out


def num(cell: str) -> float:
    m = re.search(r"-?\d+\.?\d*", cell)
    return float(m.group()) if m else 0.0


def style(ax, theme: dict, title: str, ylabel: str = "") -> None:
    c = THEMES[theme]
    ax.set_title(title, color=c["fg"], fontsize=13, fontweight="bold", loc="left", pad=14)
    if ylabel:
        ax.set_ylabel(ylabel, color=c["muted"], fontsize=10)
    ax.tick_params(colors=c["muted"], labelsize=10)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(c["grid"])
    ax.yaxis.grid(True, color=c["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_facecolor("none")


def save(fig, name: str, theme: str) -> None:
    DOCS.mkdir(exist_ok=True)
    path = DOCS / f"{name}-{theme}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight",
                facecolor=THEMES[theme]["bg"])
    plt.close(fig)
    print(f"  {path.relative_to(ROOT)}")


# --- chart 1: the corpus filter ---------------------------------------------

def chart_filter(rows: list[list[str]], theme: str) -> None:
    c = THEMES[theme]
    labels = ["substring match\n(first version)", "title decides,\nbody breaks ties"]
    kept = [num(r[1]) for r in rows[1:3]]
    prec = [num(r[2]) for r in rows[1:3]]
    rec = [num(r[3]) for r in rows[1:3]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    fig.patch.set_facecolor(c["bg"])

    bars = ax1.bar(labels, kept, color=[WARN, BEST], width=0.55)
    ax1.bar_label(bars, fmt="%.0f ads", color=c["fg"], fontsize=11, padding=4)
    ax1.set_ylim(0, max(kept) * 1.22)
    style(ax1, theme, "Ads kept by the filter", "ads in corpus")

    x = range(len(labels))
    w = 0.35
    b1 = ax2.bar([i - w / 2 for i in x], prec, w, label="precision", color=MID)
    b2 = ax2.bar([i + w / 2 for i in x], rec, w, label="recall", color=BASE)
    ax2.bar_label(b1, fmt="%.2f", color=c["fg"], fontsize=10, padding=3)
    ax2.bar_label(b2, fmt="%.2f", color=c["muted"], fontsize=10, padding=3)
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(labels)
    ax2.set_ylim(0, 1.45)
    leg = ax2.legend(frameon=False, fontsize=10, loc="upper left", ncol=2)
    for t in leg.get_texts():
        t.set_color(c["muted"])
    style(ax2, theme, "Precision rose, recall held", "score")

    fig.suptitle("Hand labelling all 135 ads found the filter was keeping everything",
                 color=c["muted"], fontsize=10.5, y=1.02, x=0.005, ha="left")
    save(fig, "filter", theme)


# --- chart 2 and 3: retrieval by query family -------------------------------

def chart_retrieval(rows: list[list[str]], theme: str, metric: str, name: str,
                    title: str, subtitle: str) -> None:
    c = THEMES[theme]
    header = rows[0]
    body = rows[1:]
    variants = [r[0] for r in body]

    cols = {f: header.index(f"{f} {metric}") for f in FAMILIES}
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    fig.patch.set_facecolor(c["bg"])

    n = len(variants)
    w = 0.8 / n
    colours = [VARIANT_COLOURS.get(v, MID) for v in variants]

    for i, (variant, colour) in enumerate(zip(variants, colours)):
        vals = [num(body[i][cols[f]]) for f in FAMILIES]
        pos = [j - 0.4 + w / 2 + i * w for j in range(len(FAMILIES))]
        ax.bar(pos, vals, w, label=variant, color=colour,
               edgecolor=c["bg"], linewidth=0.6)

    ax.set_xticks(range(len(FAMILIES)))
    ax.set_xticklabels([f"{f} queries" for f in FAMILIES], fontsize=11)
    ax.set_ylim(0, 1.08)
    # Down the right hand side so it reads in the same order as the bars.
    leg = ax.legend(frameon=False, fontsize=9.5, ncol=1,
                    loc="center left", bbox_to_anchor=(1.015, 0.5))
    for t in leg.get_texts():
        t.set_color(c["muted"])
    style(ax, theme, title, metric)

    fig.text(0.005, 1.0, subtitle, color=c["muted"], fontsize=10.5, ha="left")
    save(fig, name, theme)


def main() -> None:
    text = RESULTS.read_text(encoding="utf-8")
    found = tables(text)
    filter_table = next(t for t in found if "ads kept" in " ".join(t[0]))
    retrieval_table = next(t for t in found if "known recall@10" in " ".join(t[0]))

    print("writing charts from eval/results.md")
    for theme in THEMES:
        chart_filter(filter_table, theme)
        chart_retrieval(
            retrieval_table, theme, "recall@10", "recall",
            "Recall@10 by query family",
            "The semantic family is the real test. Keyword queries are a floor, "
            "not a verdict.",
        )
        chart_retrieval(
            retrieval_table, theme, "mrr", "mrr",
            "Mean reciprocal rank by query family",
            "Where the right answer lands, not just whether it appears.",
        )


if __name__ == "__main__":
    main()
