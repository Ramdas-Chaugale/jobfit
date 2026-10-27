# jobfit

[![CI](https://github.com/Ramdas-Chaugale/jobfit/actions/workflows/ci.yml/badge.svg)](https://github.com/Ramdas-Chaugale/jobfit/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-23%20passing-1b6b4a)](tests/)
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
purpose; the measurement is the point.

---

## Where the data comes from

| Source | Key needed | Ads | What it gives |
|---|---|---|---|
| **Arbeitnow** (default) | no | ~300, 115 kept | Europe and remote, full descriptions |
| RemoteOK | no | ~100 | remote tech |
| Adzuna | free key | UK wide | salary bands, UK locations |

Ads are kept only if they mention AI, ML or data work and run to at least 400
characters. From the default source that was **115 ads across 70 companies**,
median length 5,219 characters, on the run these notes describe.

These feeds are live, so the exact counts move between runs. A later run gave
115 ads across 77 companies. Any number quoted below comes from one specific
build and is labelled as such rather than presented as fixed.

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
seam still appears whole somewhere. That gives **943 chunks, median 833
characters, none over 1,221**.

Every chunk carries its job title into the embedding. Without that, a chunk
reading "5+ years with Spark and Airflow" belongs to no role and a query naming
a job title cannot reach it.

**Index.** Vectors are L2 normalised and the index uses inner product, so the
score is plain cosine similarity. Exact search, no approximation: at a thousand
chunks an approximate index adds failure modes and saves nothing.

**Roll-up.** Several chunks of one ad often rank together. A reader wants a
list of jobs, not the same job four times, so results are grouped by ad, each
taking its best chunk's score and keeping the rest for citation.

---

## What the first measurement showed

Running the obvious queries against the first build exposed a real problem.

For **"PyTorch computer vision deep learning"**, the top result is a chunk
reading *"AI is transforming our world. It can perform cognitive functions
that previously only humans could do"*. That is the company's About Us
paragraph. It outranks actual requirements sections because generic AI
marketing language sits closer to a generic AI query than a list of specific
technologies does.

Scores are also low where they should be high. **"LLM RAG retrieval augmented
generation"** tops out at 0.358, and the second and third results fall to
0.219 and 0.218.

This is the known weakness of naive chunking and it is the baseline the
evaluation exists to improve on. It is recorded here rather than quietly fixed,
because the before and after is the part worth showing.

### Next, in order

1. An evaluation set with known-correct answers, so recall@k and MRR mean something
2. Measure the baseline above properly instead of eyeballing three queries
3. Try the fixes: drop boilerplate chunks, weight requirement sections, hybrid keyword and vector search, a reranker
4. Report which ones actually helped and by how much
5. Answer generation with citations, measured for faithfulness
6. Serve it, deploy it, watch latency and retrieval quality over time

---

## Status

Working: ingest from three sources, chunking, embedding, FAISS index, search,
ad-level roll-up. 23 tests over the cleaning, filtering and chunking logic, run
in CI.

CI deliberately does not install torch, sentence-transformers or faiss. The
tests cover the text handling, which is the part that breaks quietly; pulling
two gigabytes of model weights to test a splitter would be slow for no extra
coverage.

Not built yet: evaluation harness, answer generation, API, deployment.

The README will carry the numbers as they arrive, including the ones that do
not flatter it.
