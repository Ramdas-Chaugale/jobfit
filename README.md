# jobfit

[![CI](https://github.com/Ramdas-Chaugale/jobfit/actions/workflows/ci.yml/badge.svg)](https://github.com/Ramdas-Chaugale/jobfit/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-63%20passing-1b6b4a)](tests/)
[![No API key](https://img.shields.io/badge/API%20key-not%20required-1b6b4a)](#where-the-data-comes-from)

Retrieval over live job ads, built to be measured rather than demonstrated.

Ask it something in plain English, get back the roles that actually match, with
the sentence from the ad that justifies each one.

```bash
pip install -r requirements.txt
python -m jobfit.corpus              # 300 live ads, no API key needed
python -m jobfit.index build         # embed and index
python -m jobfit.index search "who wants PyTorch and Kubernetes"
```

Runs on a clean clone with nothing to sign up for.

---

## Why another RAG project

Most of them stop at "it answers questions", which is not a claim anyone can
check. The interesting question is whether the retrieval is any good, and that
needs numbers.

So this one is built around its evaluation. The pipeline is ordinary on
purpose; the measurement is the point. Two of the three things it found were
bugs in this code, not shortcomings of the model.

---

## What the measurement found

Full numbers in [eval/results.md](eval/results.md). CI regenerates that file
and fails if it does not match, so nothing below is a number typed in by hand.

### 1. The filter was keeping almost everything

The corpus is meant to hold only AI, ML and data roles. All 135 ads in the
frozen snapshot passed the first version of that filter. Hand labelling all
135 says 28 are plainly those roles, so it was running at **21% precision**:
close to no filter at all.

Two reasons, both ordinary.

- `llm` was matched as a substring, so it fired inside **fulfillment**.
- "AI" appears in **121 of the 135 ads**. Every company selling software says
  it somewhere, so a mention of AI says almost nothing about the job.

The replacement rests on one observation: the title says what the job is, the
body says what the company does. So the title decides, and the body is only
consulted for a title like "Senior Software Engineer" that does not say.

| filter | ads kept | precision | recall |
|---|---:|---:|---:|
| substring match | 135 | 0.21 | 1.00 |
| title decides, body breaks ties | 37 | 0.76 | 1.00 |

Recall stays at 1.00, so the precision is not bought by dropping real roles.
The threshold is swept in `python -m eval.run_filter`, which also prints every
ad it disagrees with the labels about, because a number on its own does not say
whether the rule is wrong or the label is.

### 2. A third of the chunks say nothing about the job

**315 of 926 chunks** name no requirement and no specific technology. They are
About Us paragraphs and equal opportunity notices. For the query *"PyTorch
computer vision deep learning"* the old top result was a chunk reading *"AI is
transforming our world"*, because generic marketing prose about AI sits closer
to a generic AI query than a list of tools does.

### 3. On two families out of three, plain keyword search beats the embeddings

61 queries, three families, every retriever returning ten ads.

| variant | known recall@10 | known mrr | keyword recall@10 | keyword mrr | semantic recall@10 | semantic mrr |
|---|---:|---:|---:|---:|---:|---:|
| dense (baseline) | 0.686 | 0.496 | 0.552 | 0.676 | 0.297 | 0.456 |
| bm25 only | 1.000 | 0.932 | 0.946 | 0.935 | 0.203 | 0.135 |
| hybrid rrf | 0.771 | 0.663 | 0.885 | 0.894 | 0.312 | 0.259 |
| hybrid + boilerplate | 0.800 | 0.701 | 0.890 | 0.917 | 0.391 | 0.287 |
| hybrid + relevance filter | 0.886 | 0.749 | 0.953 | 0.972 | 0.469 | 0.481 |
| hybrid + filter + boilerplate | 0.886 | 0.783 | **0.953** | 0.944 | **0.500** | **0.560** |

BM25 wins the keyword family by construction, since the correct answers for
that family are defined by a keyword. It also wins the known-item family
outright, which is not by construction and is worth saying plainly: for finding
one specific ad on a corpus this small, a forty line BM25 beats a transformer.

The semantic family is the one that justifies the embeddings. Its queries are
hand marked and share almost no words with the ads, so *"keeping models running
in production once they are built"* has to reach an MLOps ad that never uses
that phrase. BM25 manages 0.203 recall@10 there. The baseline manages 0.297,
and the two fixes together take it to **0.500**, with MRR from 0.456 to 0.560.

Filtering removes three ads that are correct answers, so the filtered rows sit
under a recall ceiling of 0.90, 0.96 and 0.94. Their shortfall against the
unfiltered rows on known-item recall is a missing ad, not a worse ranking, and
results.md says so next to the table.

---

## How the evaluation is built

Three query families, kept apart so an easy one cannot hide a failure in a hard
one. Built by `python -m eval.build_queries` and committed.

| family | n | correct answers come from | its weakness |
|---|---:|---|---|
| known_item | 35 | the ad a requirement line was taken from | generated, so only as good as the line picked |
| keyword_topical | 18 | a whole-word keyword match | favours keyword retrieval by construction |
| semantic_topical | 8 | hand marked, little word overlap | small, and the marking is one person's |

Two things that had to be fixed before these numbers meant anything:

- Known-item queries have the ad's title words stripped out. Leave them in and
  the query carries its own answer, and every retriever looks good.
- The first version took the longest line of each ad, which in a lot of these
  ads is the equal opportunity notice. That produced queries like *"qualified
  applicants receive consideration employment without regard race"*, which
  every ad matches, so the question has no answer. `is_boilerplate` now rejects
  those lines, and it is the same function retrieval uses to downweight a
  chunk.

`tests/test_eval_set.py` guards the set itself: every correct answer is an ad
that exists, no correct answer is an ad labelled a different job, no known-item
query leaks its title, and the semantic queries stay low-overlap. A query set
that quietly drifts from the corpus does not crash, it just stops being true.

---

## Where the data comes from

| Source | Key needed | Ads | What it gives |
|---|---|---|---|
| **Arbeitnow** (default) | no | ~300 fetched | Europe and remote, full descriptions |
| RemoteOK | no | ~100 | remote tech |
| Adzuna | free key | UK wide | salary bands, UK locations |

These feeds are live, so counts move between runs. Everything measured above
uses the committed snapshot in `data/eval_corpus.jsonl`, which is 135 ads
across 77 companies, median length 5,020 characters. Chunk and query vectors
are committed too, so the whole evaluation replays with numpy and nothing else.

Adzuna truncates some descriptions to a single sentence. Those are dropped
rather than embedded, because a one-line ad adds a retrievable item with
nothing retrievable in it.

---

## How it is put together

**Chunking.** A whole ad is the wrong unit: the median one covers the company
blurb, the role, the requirements and the benefits, and embedding all of it
into one vector blurs the requirements. A single sentence is also wrong, since
"Experience with Kubernetes" retrieves well and then says nothing about whose
job it is.

Ads are split on blank lines and bullet boundaries, packed to about 700
characters, with 120 characters of overlap so a requirement that straddles a
seam still appears whole somewhere. That gives 926 chunks, median 833
characters, none over 1,221.

Every chunk carries its job title into the embedding. Without that, a chunk
reading "5+ years with Spark and Airflow" belongs to no role and a query naming
a job title cannot reach it.

**Index.** Vectors are L2 normalised and the index uses inner product, so the
score is plain cosine similarity. Exact search, no approximation: at a thousand
chunks an approximate index adds failure modes and saves nothing.

**Fusion.** The hybrid rows fuse the dense and BM25 rankings with reciprocal
rank fusion rather than adding their scores, because a cosine similarity and a
BM25 score are not on the same scale and normalising them would itself need
tuning.

**Roll-up.** Several chunks of one ad often rank together. A reader wants a list
of jobs, not the same job four times, so results are grouped by ad, each taking
its best chunk's score and keeping the rest for citation.

---

## Running it

```bash
python -m eval.run_filter               # filter precision and recall, with disagreements
python -m eval.build_queries            # rebuild the query set
python -m eval.run --inspect            # the table, plus the baseline's worst failures
python -m eval.run --inspect --check    # fail if results.md would change
pytest                                  # 63 tests, no model download
```

---

## Status

Working: ingest from three sources, the relevance filter, chunking, embedding,
FAISS index, BM25, hybrid fusion, ad-level roll-up, and the evaluation above.
63 tests in CI, plus a second CI job that replays the whole evaluation and
compares it to the committed results.

Not built yet: answer generation with citations, faithfulness scoring of a
generated answer against those citations, a cross-encoder reranker, an HTTP
API, and deployment.

Next, in order:

1. A reranker over the hybrid top 25, measured on the semantic family where
   there is the most room left
2. Answer generation with citations, and a faithfulness measure for it
3. Serve it, then watch latency and retrieval quality over time

The README carries the numbers as they arrive, including the ones that do not
flatter it.
