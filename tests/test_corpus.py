"""Tests for cleaning and filtering job ads.

tidy() is the one with a trap in it. Arbeitnow double-escapes its HTML, so the
tags arrive as &lt;p&gt; rather than <p>. Unescaping once, or stripping tags
before unescaping, leaves the markup behind as visible text and it gets
embedded along with the job description. The first version of this did exactly
that, which is why the order is pinned here.
"""

from __future__ import annotations

from jobfit.corpus import MIN_CHARS, Ad, _keep, is_relevant, tidy


def test_plain_tags_are_stripped():
    assert tidy("<p>Hello <b>world</b></p>") == "Hello world"


def test_double_escaped_html_is_handled():
    """The case that actually broke. Unescape twice, then strip."""
    raw = "&lt;p&gt;&lt;strong&gt;About us&lt;/strong&gt;&lt;/p&gt;"
    out = tidy(raw)
    assert out == "About us"
    assert "<" not in out and "&" not in out


def test_entities_become_characters():
    assert tidy("R&amp;D &amp; ops") == "R&D & ops"
    assert tidy("5&nbsp;years") == "5 years"


def test_script_and_style_contents_are_removed():
    """Tracking scripts are not part of the job description."""
    raw = "<p>Real text</p><script>var x = 1;</script><style>.a{color:red}</style>"
    out = tidy(raw)
    assert "Real text" in out
    assert "var x" not in out and "color:red" not in out


def test_whitespace_is_collapsed_but_lines_survive():
    out = tidy("a    b\n\nc")
    assert "a b" in out
    assert "\n" in out, "paragraph breaks are what the chunker splits on"


def test_tidy_handles_empty_input():
    assert tidy("") == ""
    assert tidy(None) == ""


def test_relevance_catches_the_roles_we_want():
    assert is_relevant("Machine Learning Engineer", "")
    assert is_relevant("Engineer", "You will use PyTorch daily")
    assert is_relevant("Analyst", "data science team")


def test_relevance_rejects_unrelated_roles():
    assert not is_relevant("Refrigeration Engineer",
                           "Service commercial fridges across the region.")
    assert not is_relevant("Retail Assistant", "Greet customers, handle stock.")


def ad(text, title="Data Scientist"):
    return Ad(id="x", title=title, company="c", location="l", posted="2026-01-01",
              source="test", url="u", text=text)


def test_a_one_line_ad_is_dropped():
    """Some feeds truncate to a sentence. That adds a retrievable item with
    nothing retrievable in it."""
    assert _keep([ad("We are hiring a data scientist.")]) == []


def test_a_long_relevant_ad_is_kept():
    kept = _keep([ad("We need a data scientist. " * 40)])
    assert len(kept) == 1


def test_a_long_but_unrelated_ad_is_dropped():
    assert _keep([ad("Drive a van. " * 60, title="Delivery Driver")]) == []


def test_min_chars_is_the_boundary():
    assert _keep([ad("machine learning " + "x" * (MIN_CHARS - 20))]) == []
    assert len(_keep([ad("machine learning " + "x" * (MIN_CHARS + 20))])) == 1
