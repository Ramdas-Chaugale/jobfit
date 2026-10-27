"""Decide whether a job ad is actually an AI, ML or data role.

The first version of this asked whether certain words appeared anywhere in the
ad. Measured against hand labels it kept 135 ads of which 28 were the roles
wanted, so 21% precision: close to no filter at all. Two reasons.

The substring problem. "llm" matched inside "fulfillment", and "data scien"
inside a sentence about the company's data science team rather than the job.

The bigger one: "AI" appears in 121 of those 135 ads. Every company selling
software in 2026 says it somewhere, so a mention of AI carries almost no
information about what the job is.

The fix rests on one observation: the title says what the job is, the body says
what the company does. So the title decides, and the body is only consulted
when the title does not say, as with "Senior Software Engineer".

THRESHOLD is deliberately a parameter. eval/run_filter.py sweeps it and prints
precision and recall at each value instead of one number being asserted here.
"""

from __future__ import annotations

import re

# Titles that name the job outright.
TITLE_ROLE = [
    "machine learning", "ml engineer", "mlops", "ai engineer", "ai/ml",
    "data scientist", "data engineer", "data analyst", "analytics engineer",
    "deep learning", "computer vision", "reinforcement learning",
    "nlp", "research scientist", "ai platform", "ai software",
    "generative ai", "data platform", "data analytics", "algorithm engineer",
    "ai product", "ai tooling", "ai workflows", "ai ops", "ai solution",
    "ki-entwickler", "vision et apprentissage",
]

# Titles that name a different job. An account manager at an AI company is
# still an account manager, however much the ad talks about the product.
TITLE_NOT_ROLE = [
    "account manager", "account executive", "sales", "business development",
    "marketing", "seo", "affiliate", "paid social", "social creative",
    "art director", "scriptwriter", "video producer", "designer", "ux", "ui",
    "talent acquisition", "recruit", "customer onboarding", "contracts",
    "insurance", "actuary", "clinician", "facilitator", "co-founder",
    "investor relations", "property", "frontend", "ios", "firmware",
    "hardware", "electrical", "manufacturing", "kundenservice",
    "grafik-design", "ubersetzung", "übersetzung", "growth intern",
    "solutions engineer", "enablement", "programme director", "qa manager",
    "business analyst", "product owner", "produktmanager", "user researcher",
]

# Specific enough that a marketing page would not use them in passing.
SPECIFIC = [
    "pytorch", "tensorflow", "scikit-learn", "sklearn", "keras", "jax",
    "hugging face", "huggingface", "transformers", "mlflow", "kubeflow",
    "sagemaker", "vertex ai", "databricks", "airflow", "dbt", "spark",
    "feature store", "model registry", "model serving", "model deployment",
    "fine-tuning", "fine tuning", "embeddings", "vector database",
    "rag", "retrieval augmented", "langchain", "llamaindex",
    "computer vision", "nlp", "natural language processing",
    "reinforcement learning", "deep learning", "neural network",
    "mlops", "machine learning", "data pipeline", "etl", "pandas", "numpy",
]

# Worth nothing on their own. This is the whole point.
GENERIC = ["ai", "artificial intelligence", "ki", "genai"]

# Swept in eval/run_filter.py. Two, not three, because the sweep shows three
# keeps nothing off-target but loses six adjacent roles a job seeker would
# still want to see, and showing one extra ad costs less than hiding a real
# one. Both values keep every strictly-labelled role.
THRESHOLD = 2


def _has(word: str, text: str) -> bool:
    """Whole-word match, so fulfillment stops counting as an LLM mention."""
    return re.search(r"(?<![a-z0-9])%s(?![a-z0-9])" % re.escape(word),
                     text) is not None


def score(title: str, text: str) -> int:
    t, b = title.lower(), text.lower()

    if any(p in t for p in TITLE_ROLE):
        return 4
    if any(p in t for p in TITLE_NOT_ROLE):
        return 0

    # Title does not say. Count distinct specific mentions in the body, worth
    # one each and capped, so a long ad cannot pass on repetition alone.
    found = {s for s in SPECIFIC if _has(s, b)}
    return min(3, len(found))


def is_relevant(title: str, text: str, threshold: int = THRESHOLD) -> bool:
    return score(title, text) >= threshold


def why(title: str, text: str) -> str:
    """Readable reason, for inspecting disagreements with the labels."""
    t, b = title.lower(), text.lower()
    for p in TITLE_ROLE:
        if p in t:
            return "title says %r" % p
    for p in TITLE_NOT_ROLE:
        if p in t:
            return "title says %r, a different job" % p
    found = sorted(s for s in SPECIFIC if _has(s, b))
    if not found:
        g = [x for x in GENERIC if _has(x, b)]
        return "nothing specific in the body%s" % (
            " (only %s)" % ", ".join(g) if g else "")
    return "body has %s" % ", ".join(found[:5])


# A passage that says what the job needs will almost always contain one of
# these. A company blurb or an equal opportunity notice will not.
REQUIREMENT_CUES = [
    "experience", "you have", "you'll", "you will", "degree", "years",
    "skills", "proficien", "familiar", "knowledge of", "ability to",
    "we expect", "requirement", "qualification", "must have", "nice to have",
    "background in", "comfortable with", "track record",
]


def is_boilerplate(text: str) -> bool:
    """True when a passage names no requirement and no specific technology.

    Two uses, one definition. Retrieval downweights these chunks, because an
    About Us paragraph was outranking real requirements sections: generic
    marketing prose about AI sits closer to a generic AI query than a list of
    tools does. Query building rejects them as well, since the longest line in
    a job ad is often the equal opportunity notice, and "find the ad with this
    standard legal disclaimer" has no answer when every ad carries it.
    """
    low = text.lower()
    has_cue = any(c in low for c in REQUIREMENT_CUES)
    has_tech = any(_has(s, low) for s in SPECIFIC)
    return not (has_cue or has_tech)
