"""Query embedding without torch.

sentence-transformers is the right tool for building the index, and that
happens once on a machine with room for it. Serving is a different problem: a
request embeds one short query, and dragging torch along for that costs about
800 MB unpacked, which is more than a serverless function is allowed to be.

So this runs the same all-MiniLM-L6-v2 weights through onnxruntime instead.
The fp32 model is 90 MB, tokenizer 0.5 MB, runtime about 15 MB. Together
with numpy that is roughly 130 MB, inside the 250 MB a serverless function gets.

The int8 quantised model is 23 MB and was the obvious pick until it was
measured: it reproduces the committed query vectors at 0.989 mean cosine, not
0.999. Saving 67 MB is not worth serving queries from a slightly different
vector space to the one the chunks were embedded in, so fp32 it is.

It has to produce the same vectors as the torch path, because the chunk vectors
committed to data/ were built with torch and a query embedded differently would
be compared against them. test_onnx_encoder.py checks exactly that against the
61 committed query vectors, and fails below 0.999 cosine similarity.
"""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np

REPO = "sentence-transformers/all-MiniLM-L6-v2"
ONNX_FILE = os.environ.get("JOBFIT_ONNX_FILE", "onnx/model.onnx")
MAX_TOKENS = 256

# Set JOBFIT_MODEL_DIR to a directory holding model.onnx and tokenizer.json to
# avoid any network call at runtime. A deploy vendors the files and sets this.
LOCAL_DIR = os.environ.get("JOBFIT_MODEL_DIR")


def _fetch(filename: str) -> str:
    if LOCAL_DIR:
        local = os.path.join(LOCAL_DIR, os.path.basename(filename))
        if os.path.exists(local):
            return local
    from huggingface_hub import hf_hub_download
    return hf_hub_download(REPO, filename)


@lru_cache(maxsize=1)
def _session():
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1          # one short query, threads only add overhead
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(_fetch(ONNX_FILE), opts,
                                providers=["CPUExecutionProvider"])


@lru_cache(maxsize=1)
def _tokenizer():
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(_fetch("tokenizer.json"))
    tok.enable_truncation(max_length=MAX_TOKENS)
    tok.enable_padding()
    return tok


def encode(texts: list[str]) -> np.ndarray:
    """Mean pool over tokens, masked, then L2 normalise.

    That pooling is not a choice. It is what all-MiniLM-L6-v2 does, and getting
    it wrong produces vectors that look fine and rank badly.
    """
    encoded = _tokenizer().encode_batch(list(texts))
    ids = np.array([e.ids for e in encoded], dtype=np.int64)
    mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)

    feed = {"input_ids": ids, "attention_mask": mask}
    names = {i.name for i in _session().get_inputs()}
    if "token_type_ids" in names:
        feed["token_type_ids"] = np.zeros_like(ids)

    hidden = _session().run(None, feed)[0]            # (batch, tokens, 384)

    m = mask[..., None].astype(hidden.dtype)
    summed = (hidden * m).sum(axis=1)
    counts = np.clip(m.sum(axis=1), 1e-9, None)
    pooled = summed / counts

    norms = np.linalg.norm(pooled, axis=1, keepdims=True)
    return (pooled / np.clip(norms, 1e-12, None)).astype("float32")


def encode_one(text: str) -> np.ndarray:
    return encode([text])[0]
