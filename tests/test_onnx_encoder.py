"""The ONNX encoder has to agree with the torch one, or the committed vectors lie.

data/eval_vectors.npy holds 926 chunk vectors embedded with sentence-transformers.
Serving embeds the query with onnxruntime instead. If the two paths disagree,
every cosine similarity in the system is comparing vectors from two slightly
different spaces, and nothing would look broken: results would just be quietly
worse.

So this pins it against the 61 query vectors that were produced by the torch
path and committed alongside the chunks.
"""

from __future__ import annotations

import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUERY_VECTORS = os.path.join(ROOT, "data", "eval_query_vectors.npz")

# The lint and test job installs neither of these on purpose: it covers the
# text handling, which is what breaks quietly. Skip rather than drag 100 MB of
# runtime into a job that tests a text splitter. The evaluate job has numpy,
# and a local run before a release has both.
np = pytest.importorskip("numpy", reason="numpy not installed")
pytest.importorskip("onnxruntime", reason="onnxruntime not installed")


@pytest.fixture(scope="module")
def pairs():
    if not os.path.exists(QUERY_VECTORS):
        pytest.skip("committed query vectors not present")
    with np.load(QUERY_VECTORS) as z:
        queries = list(z.files)
        return queries, np.stack([z[q] for q in queries])


def test_matches_the_torch_vectors(pairs):
    from jobfit.onnx_encoder import encode

    queries, reference = pairs
    got = encode(queries)
    sims = (got * reference).sum(axis=1)

    worst = int(np.argmin(sims))
    assert sims.min() > 0.999, (
        f"ONNX drifted from the torch vectors. Worst: {queries[worst]!r} "
        f"at {sims[worst]:.5f}. The committed chunk vectors cannot be reused "
        f"with this encoder."
    )


def test_output_is_normalised(pairs):
    from jobfit.onnx_encoder import encode

    queries, _ = pairs
    got = encode(queries[:8])
    norms = np.linalg.norm(got, axis=1)
    # cosine similarity is computed as a plain dot product everywhere else,
    # which is only correct if these are unit vectors
    assert np.allclose(norms, 1.0, atol=1e-5), f"norms were {norms}"


def test_shape_and_dtype(pairs):
    from jobfit.onnx_encoder import encode

    got = encode(["keeping models running in production", "pytorch"])
    assert got.shape == (2, 384)
    assert got.dtype == np.float32
