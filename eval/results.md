# Results

Produced by `python -m eval.run` over the frozen snapshot in `data/eval_corpus.jsonl`, which is 135 ads. Rebuilding the live corpus changes the ads, so the snapshot is committed and the numbers below refer only to it.

## The corpus filter

Measured in `python -m eval.run_filter` against 135 hand labels in `eval/labels.json`.

| filter | ads kept | precision | recall |
|---|---:|---:|---:|
| substring match, the first version | 135 | 0.21 | 1.00 |
| title decides, body breaks ties | 37 | 0.76 | 1.00 |

Precision and recall are against the 28 ads labelled plainly an AI, ML or data role. The first filter kept everything, because a single mention of AI passed an ad and 121 of the 135 mention it. It also matched `llm` inside the word fulfillment.

## Retrieval

61 queries in three families. Every retriever returns ten ads, scored the same way.

| variant | known recall@10 | known mrr | keyword recall@10 | keyword mrr | semantic recall@10 | semantic mrr |
|---|---:|---:|---:|---:|---:|---:|
| dense (baseline) | 0.686 | 0.496 | 0.552 | 0.676 | 0.297 | 0.456 |
| bm25 only | 1.000 | 0.932 | 0.946 | 0.935 | 0.203 | 0.135 |
| hybrid rrf | 0.771 | 0.663 | 0.885 | 0.894 | 0.312 | 0.259 |
| hybrid + boilerplate | 0.800 | 0.701 | 0.890 | 0.917 | 0.391 | 0.287 |
| hybrid + relevance filter | 0.886 | 0.749 | 0.953 | 0.972 | 0.469 | 0.481 |
| hybrid + filter + boilerplate | 0.886 | 0.783 | 0.953 | 0.944 | 0.500 | 0.560 |

Recall ceiling for the two filtered variants, since filtering removes three ads that are correct answers: known 0.90, keyword 0.96, semantic 0.94. Their recall cannot pass that and a shortfall against the unfiltered variants is partly a missing ad, not a worse ranking.

## Reading it

- **bm25 beats the embeddings on the keyword family.** It would be strange if it did not: the correct answers for that family are defined by a keyword. That family is a floor, not a verdict.
- **the semantic family is the real test**, and the baseline scores 0.297 recall@10 there against 0.500 for hybrid + filter + boilerplate.
- **34% of chunks carry no requirement signal**, which is the About Us problem the first README described. Downweighting them is the cheapest of the fixes tried here.

## Still missing

Answer generation, faithfulness of a generated answer against its citations, a reranker, and anything served over HTTP. The numbers here cover retrieval only.

## What the baseline gets wrong


**academic research rather than product work**  (0% of the top 5 correct)

- `no ` Product Manager - Research Participants - United Kingdom
- `no ` Data Scientist - Research Participants - United Kingdom
- `no ` Data Analyst - Research Participants - United Kingdom
- `no ` Senior User Researcher

**building a chatbot that answers from company documents**  (0% of the top 5 correct)

- `no ` Pflichtpraktikum Informatik (Webprogrammierung + SEO)
- `no ` Senior Product Owner
- `no ` AI Tooling Engineer
- `no ` Director, Engineering, AI Search

**something I could do while still a student**  (20% of the top 5 correct)

- `ok ` Working Student AI Software Engineer, LangChain (all genders)
- `no ` Senior MLOps Engineer, Buyer
- `no ` Full Stack Developer (Onboarding & Enablement)
- `no ` Senior Paid Social Manager

**a job based in Britain**  (20% of the top 5 correct)

- `ok ` Data Engineer - Research Participants - United Kingdom
- `no ` Sales / Account Executive - Research Participants - United Kingdom
- `no ` Engineering Manager - Research Participants - United Kingdom
- `no ` Account Manager - EMEA
