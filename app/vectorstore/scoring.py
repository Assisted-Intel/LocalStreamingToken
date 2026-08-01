#!/usr/bin/env python3
"""
Local Streaming Token — storage-independent scoring helpers.

Moved out of ``rag.py`` so both it and the backends can use them without an import
cycle (``rag`` imports ``vectorstore``, so a backend must not import ``rag``).

Pure Python on purpose: cosine similarity and BM25 need no DuckDB extension and no
third-party numerics, so keyword and hybrid retrieval keep working offline, with no
embedding model, and on any build.
"""

import math
import re


def cosine(a, b) -> float:
    """Standard cosine similarity between two equal-length float lists."""
    if not a or not b or len(a) != len(b):
        return -1.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return -1.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list:
    return _TOKEN_RE.findall((text or "").lower())


def bm25_search(rows: list, query: str, top_k: int, k1: float = 1.5, b: float = 0.75) -> list:
    """Rank ``rows`` (dicts with 'content' and optional 'context') against ``query``
    using Okapi BM25 over the in-scope candidate set. Returns rows (copied, with a
    'score') sorted by descending BM25 score; only rows with score > 0 are returned.

    Note this scores a *materialised* candidate set, so its cost grows with the size of
    the scope, not with the number of matches — which is why the LanceDB backend uses a
    real inverted index (tantivy) instead of calling this."""
    q_terms = [t for t in set(tokenize(query)) if t]
    if not rows or not q_terms:
        return []
    docs = [tokenize((r.get("context") or "") + " " + (r.get("content") or "")) for r in rows]
    n = len(docs)
    avgdl = (sum(len(d) for d in docs) / n) or 1.0
    df = {}
    for d in docs:
        for term in set(d):
            df[term] = df.get(term, 0) + 1
    scored = []
    for r, d in zip(rows, docs):
        if not d:
            continue
        dl = len(d)
        tf = {}
        for term in d:
            tf[term] = tf.get(term, 0) + 1
        score = 0.0
        for term in q_terms:
            f = tf.get(term)
            if not f:
                continue
            ni = df.get(term, 0)
            idf = math.log(1 + (n - ni + 0.5) / (ni + 0.5))
            score += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / avgdl))
        if score > 0:
            rr = dict(r)
            rr["score"] = score
            scored.append(rr)
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:int(top_k)]


def result_key(r: dict):
    """Stable identity for a retrieved chunk, used to dedupe across queries/modes."""
    return r.get("id") or (r.get("source_id"), r.get("item_id"), r.get("content"))


def rrf_merge(result_lists: list, top_k: int, k: int = 60) -> list:
    """Reciprocal-rank-fusion merge of several ranked result lists (each already sorted
    best-first). The fused score replaces the per-list score; ties broken by insertion
    order. Combines multiple query variants and the vector/keyword sets in hybrid mode.

    Because it consumes *ranks*, not raw scores, it merges cleanly across backends whose
    scores are not comparable — BM25 magnitudes and tantivy's BM25 differ, as do cosine
    similarity and Lance's distance-derived scores."""
    fused = {}
    keep = {}
    for lst in result_lists:
        for rank, r in enumerate(lst):
            kk = result_key(r)
            fused[kk] = fused.get(kk, 0.0) + 1.0 / (k + rank + 1)
            if kk not in keep:
                keep[kk] = r
    merged = []
    for kk, s in fused.items():
        rr = dict(keep[kk])
        rr["score"] = s
        merged.append(rr)
    merged.sort(key=lambda x: x["score"], reverse=True)
    return merged[:int(top_k)]
