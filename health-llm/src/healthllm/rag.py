"""Retrieval over the research corpus (papers/chunks.jsonl), exposed to the model as a tool.

How it works:
  * Each chunk is turned into a vector (an "embedding") by a small model, bge-small (33M params).
    Texts that mean similar things get vectors that point in similar directions.
  * At question time the query is embedded the same way, and we return the chunks whose vectors
    have the highest cosine similarity with it. With 179 chunks a brute-force dot product over a
    NumPy matrix is instant, so no vector database is needed.

Build the index once: `uv run python -m healthllm.rag "does sleep regularity matter?"`
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

import numpy as np

from .papers import CHUNKS_PATH, PAPERS_DIR

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
INDEX_PATH = PAPERS_DIR / "index.npy"
# bge models retrieve better when the *query* (not the passages) carries this instruction.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
EXCERPT_WORDS = 110  # trim returned chunks so tool results stay small for a 2B model's context


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBED_MODEL, device="cpu")


@lru_cache(maxsize=1)
def _chunks() -> list[dict[str, Any]]:
    return [json.loads(line) for line in CHUNKS_PATH.read_text().splitlines()]


@lru_cache(maxsize=1)
def _index() -> np.ndarray:
    chunks = _chunks()
    if INDEX_PATH.exists():
        idx = np.load(INDEX_PATH)
        if len(idx) == len(chunks):
            return idx
    texts = [f"{c['title']}. {c['section']}. {c['text']}" for c in chunks]
    idx = _model().encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
    np.save(INDEX_PATH, idx.astype(np.float32))
    return idx


def search_papers(query: str, marker: str | None = None, k: int = 3) -> dict[str, Any]:
    """Top-k excerpts for a query, optionally limited to one longevity marker's studies."""
    query = (query or "").strip()
    if not query:
        raise ValueError("query must not be empty.")
    chunks, idx = _chunks(), _index()
    q = _model().encode([QUERY_PREFIX + query], normalize_embeddings=True)[0]
    scores = idx @ q  # cosine similarity, since every vector is unit length
    order = np.argsort(-scores)
    results, seen = [], set()
    for i in order:
        c = chunks[i]
        if marker and c["marker"] != marker:
            continue
        if (c["study"], c["section"]) in seen:  # one excerpt per section keeps results varied
            continue
        seen.add((c["study"], c["section"]))
        words = c["text"].split()
        results.append({
            "cite": c["cite"], "section": c["section"],
            "excerpt": " ".join(words[:EXCERPT_WORDS]) + (" …" if len(words) > EXCERPT_WORDS else ""),
            "url": c["url"], "score": round(float(scores[i]), 3),
        })
        if len(results) == k:
            break
    return {"query": query, "results": results,
            "note": "Excerpts from the studies behind the longevity targets. Cite them by name."}


MARKERS = sorted({c["marker"] for c in _chunks()}) if CHUNKS_PATH.exists() else []

SEARCH_SPEC = {
    "name": "search_papers",
    "description": "Search the full text of the research papers behind the longevity targets (steps, exercise, "
                   "strength, sleep, resting heart rate, VO2 max) and return the most relevant excerpts with "
                   "their citation. Use it when the person asks what the research says, why a target matters, "
                   "or how strong the evidence is.",
    "input_schema": {"type": "object", "additionalProperties": False, "required": ["query"], "properties": {
        "query": {"type": "string"},
        "marker": {"type": "string", "enum": ["steps", "exercise", "strength", "sleep", "resting_hr", "vo2max"]}}},
}


if __name__ == "__main__":
    import sys

    _index()
    print(f"index: {INDEX_PATH.relative_to(PAPERS_DIR.parent)} ({len(_chunks())} chunks)")
    for r in search_papers(" ".join(sys.argv[1:]) or "is sleep regularity linked to mortality?")["results"]:
        print(f"- [{r['score']}] {r['cite']} / {r['section']}: {r['excerpt'][:160]}")
