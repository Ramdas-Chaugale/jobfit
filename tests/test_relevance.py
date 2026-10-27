"""Tests for deciding whether an ad is an AI, ML or data role.

The two bugs that made the first filter useless are both pinned here, because
both were invisible until the ads were hand labelled.
"""

from __future__ import annotations

from jobfit import relevance
from jobfit.corpus import is_relevant as old_filter

LONG = " ".join(["The company builds software for its customers."] * 20)


def test_llm_no_longer_matches_inside_fulfillment():
    """The original bug. 'llm' as a substring hits fulfillment, so a logistics
    ad counted as a large language model role."""
    text = "We run a fulfillment centre for our retail partners. " + LONG
    assert old_filter("Operations Manager", text), "the old filter did match"
    assert not relevance.is_relevant("Operations Manager", text)


def test_a_passing_mention_of_ai_is_not_evidence_of_an_ai_job():
    """121 of 135 ads in the snapshot mention AI. It carries no information."""
    text = "We are an AI first company transforming our industry. " + LONG
    assert old_filter("Account Manager", text), "the old filter did match"
    assert not relevance.is_relevant("Account Manager", text)


def test_the_title_decides_when_it_names_the_role():
    assert relevance.is_relevant("Machine Learning Engineer", LONG)
    assert relevance.is_relevant("Senior MLOps Engineer, Buyer", LONG)
    assert relevance.is_relevant("Data Engineer", LONG)


def test_a_different_job_at_an_ai_company_is_still_a_different_job():
    text = ("Join our machine learning platform team's go to market effort. "
            "We use PyTorch, Kubernetes and Spark across the product. " + LONG)
    assert not relevance.is_relevant("Enterprise Account Executive", text)
    assert not relevance.is_relevant("Senior Field Marketing Manager", text)


def test_a_generic_engineering_title_falls_through_to_the_body():
    """'Senior Software Engineer' says nothing either way, so the body has to
    answer, and it needs more than one specific term to pass."""
    thin = "You will write backend services in Go. " + LONG
    thick = ("You will own model serving and the feature store, with PyTorch "
             "and MLflow in production. " + LONG)
    assert not relevance.is_relevant("Senior Software Engineer", thin)
    assert relevance.is_relevant("Senior Software Engineer", thick)


def test_one_specific_term_is_not_enough_on_its_own():
    text = "Our stack mentions PyTorch somewhere. " + LONG
    assert relevance.score("Senior Software Engineer", text) == 1
    assert not relevance.is_relevant("Senior Software Engineer", text)


def test_repeating_one_term_does_not_pass_the_threshold():
    """Otherwise a long ad passes by saying the same word often enough."""
    text = ("pytorch " * 50) + LONG
    assert relevance.score("Senior Software Engineer", text) == 1


def test_score_is_capped_so_the_threshold_means_something():
    text = ("pytorch tensorflow mlflow kubeflow airflow spark dbt sagemaker "
            "embeddings langchain " + LONG)
    assert relevance.score("Senior Software Engineer", text) == 3
    assert relevance.score("Machine Learning Engineer", text) == 4


def test_why_explains_the_decision():
    assert "title says" in relevance.why("Machine Learning Engineer", LONG)
    assert "different job" in relevance.why("Account Manager", LONG)
    assert "nothing specific" in relevance.why("Senior Software Engineer", LONG)
