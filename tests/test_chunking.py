"""Tests for splitting ads into retrievable pieces.

Chunk boundaries decide what can be retrieved. A requirement cut in half is a
requirement no query reaches cleanly, so the sizes and the overlap are pinned
rather than left to drift.
"""

from __future__ import annotations

from jobfit.chunking import MAX, TARGET, chunk_ad, chunk_all
from jobfit.corpus import Ad


def ad(text, title="ML Engineer", company="Acme"):
    return Ad(id="ad1", title=title, company=company, location="Belfast",
              posted="2026-01-01", source="test", url="http://x", text=text)


def test_a_short_ad_stays_one_chunk():
    chunks = chunk_ad(ad("A short role description."))
    assert len(chunks) == 1
    assert chunks[0].position == 0


def test_a_long_ad_is_split():
    para = "This paragraph describes part of the role in some detail. " * 6
    chunks = chunk_ad(ad("\n\n".join([para] * 6)))
    assert len(chunks) > 1


def test_no_chunk_runs_far_past_the_ceiling():
    """Packing is approximate, but nothing should balloon."""
    para = "Some requirement text that goes on for a while. " * 8
    chunks = chunk_ad(ad("\n\n".join([para] * 10)))
    assert all(len(c.text) <= MAX + TARGET for c in chunks)


def test_a_wall_of_text_with_no_blank_lines_still_splits():
    """Not every ad is tidily formatted. One giant paragraph has to split on
    sentences rather than coming back as a single huge chunk."""
    wall = "This is a sentence about the role. " * 120
    chunks = chunk_ad(ad(wall))
    assert len(chunks) > 1
    assert all(len(c.text) <= MAX + TARGET for c in chunks)


def test_bullets_are_a_split_point():
    bullets = "\n".join("- Requirement number %d, described at some length" % i
                        for i in range(60))
    chunks = chunk_ad(ad("Requirements:\n" + bullets))
    assert len(chunks) > 1


def test_every_chunk_carries_the_job_title_into_the_embedding():
    """Without this a chunk reading '5 years with Spark' belongs to no role."""
    chunks = chunk_ad(ad("Requirements. " * 80, title="Senior Data Engineer"))
    assert len(chunks) > 1
    assert all("Senior Data Engineer" in c.for_embedding() for c in chunks)


def test_chunks_keep_enough_to_cite_the_ad():
    chunks = chunk_ad(ad("Some text about the role. " * 50))
    for c in chunks:
        assert c.ad_id == "ad1"
        assert c.url == "http://x"
        assert c.company == "Acme"


def test_chunk_ids_are_unique_and_ordered():
    chunks = chunk_ad(ad("Details about this position. " * 90))
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert [c.position for c in chunks] == list(range(len(chunks)))


def test_consecutive_chunks_share_some_text():
    """The overlap is what stops a requirement split across a seam from being
    unreachable in both halves."""
    para = "Requirement text describing part of the job in detail. " * 6
    chunks = chunk_ad(ad("\n\n".join([para] * 8)))
    assert len(chunks) > 2

    shared = 0
    for a, b in zip(chunks, chunks[1:]):
        tail_words = a.text.split()[-8:]
        if tail_words and " ".join(tail_words[:4]) in b.text:
            shared += 1
    assert shared > 0, "no overlap carried forward between any pair"


def test_an_empty_ad_does_not_crash():
    assert chunk_ad(ad("")) == []


def test_chunk_all_covers_every_ad():
    ads = [ad("Role description text. " * 40) for _ in range(3)]
    for i, a in enumerate(ads):
        ads[i] = Ad(**{**a.__dict__, "id": "ad%d" % i})
    chunks = chunk_all(ads)
    assert {c.ad_id for c in chunks} == {"ad0", "ad1", "ad2"}
