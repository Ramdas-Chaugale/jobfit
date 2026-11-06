"""Web interface over the retrieval.

    pip install -r requirements.txt
    python -m app.server          # then open http://127.0.0.1:8000

The corpus is built once at startup and held in memory. Two corpora actually,
filtered and unfiltered, because comparing the six variants needs both.
"""

from __future__ import annotations

import os

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.search import DEFAULT_VARIANT, VARIANTS, SearchService

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(HERE, "static")

app = FastAPI(title="jobfit", docs_url="/api/docs")
service: SearchService | None = None


@app.on_event("startup")
def _warm() -> None:
    global service
    print("building the corpus, about ten seconds")
    service = SearchService()
    print(f"ready: {len(service.ads)} ads, {len(service.plain.chunks)} chunks")


def _svc() -> SearchService:
    if service is None:
        raise HTTPException(503, "still starting up")
    return service


@app.get("/api/variants")
def variants() -> dict:
    return {"variants": list(VARIANTS), "default": DEFAULT_VARIANT}


@app.get("/api/examples")
def examples() -> dict:
    """The evaluation's own queries, so a visitor can try one that is known to
    be hard rather than having to invent one."""
    return {"examples": _svc().examples()}


@app.get("/api/search")
def search(q: str = Query(min_length=2),
           variant: str = DEFAULT_VARIANT,
           k: int = Query(10, ge=1, le=25)) -> dict:
    svc = _svc()
    try:
        hits = svc.search(q, variant, k)
    except KeyError:
        raise HTTPException(400, f"unknown variant: {variant}")
    return {"query": q, "variant": variant, "instant": svc.is_instant(q),
            "results": [h.as_dict() for h in hits]}


@app.get("/api/compare")
def compare(q: str = Query(min_length=2),
            k: int = Query(5, ge=1, le=10)) -> dict:
    """All six retrievers on one query, side by side."""
    svc = _svc()
    return {"query": q, "instant": svc.is_instant(q),
            "variants": svc.compare(q, k)}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(os.path.join(STATIC, "index.html"))


app.mount("/static", StaticFiles(directory=STATIC), name="static")


if __name__ == "__main__":
    uvicorn.run("app.server:app", host="127.0.0.1", port=8000, reload=False)
