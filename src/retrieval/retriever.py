"""Retriever policies over the vector store + the relevance gate.

Relevance score: embeddings are L2-normalised and FAISS returns SQUARED L2 distance d, so
cosine similarity = 1 - d/2. We report that (clamped to [0, 1]) because it is interpretable. The gate threshold is an EMPIRICAL property of this store + metric + model;
calibrate it with the Retrieval Lab (Task 06) rather than copying a number.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.documents import Document


def scored_search(store, query: str, k: int, filter: dict | None = None) -> list[tuple[Document, float]]:
    """Similarity search returning cosine similarity in [0, 1] (see module docstring)."""
    raw = store.similarity_search_with_score(query, k=k, filter=filter)
    return [(d, max(0.0, min(1.0, 1.0 - float(dist) / 2.0))) for d, dist in raw]


@dataclass
class RetrievalResult:
    query: str
    strategy: str
    hits: list[tuple[Document, float]] = field(default_factory=list)
    best_score: float = 0.0
    gate_passed: bool = False
    reason: str = ""


def retrieve(store, query: str, s, strategy: str | None = None, k: int | None = None,
             filter: dict | None = None, threshold: float | None = None) -> RetrievalResult:
    strategy = strategy or s.search_type
    k = k or s.retrieval_k
    thr = s.score_threshold if threshold is None else threshold
    cands = scored_search(store, query, max(s.fetch_k, k), filter)
    scores = {d.metadata["chunk_id"]: float(sc) for d, sc in cands}
    best = max(scores.values(), default=0.0)
    if strategy == "similarity":
        hits = [(d, float(sc)) for d, sc in cands[:k]]
    elif strategy == "mmr":
        docs = store.max_marginal_relevance_search(query, k=k, fetch_k=max(s.fetch_k, k), lambda_mult=s.mmr_lambda, filter=filter)
        hits = [(d, scores.get(d.metadata["chunk_id"], 0.0)) for d in docs]
        hits.sort(key=lambda x: -x[1])
    elif strategy == "threshold":
        hits = [(d, float(sc)) for d, sc in cands if sc >= thr][:k]
    else:
        raise ValueError(f"unknown strategy {strategy}")
    if not cands:
        passed, reason = False, "empty result set"
    elif best < thr:
        passed, reason = False, f"best relevance {best:.3f} < threshold {thr:.2f}"
    else:
        passed, reason = True, f"best relevance {best:.3f} >= threshold {thr:.2f}"
    return RetrievalResult(query, strategy, hits, best, passed, reason)


def as_langchain_retriever(store, s, filter: dict | None = None):
    """Idiomatic LangChain retriever (used in the LCEL chain's comparison mode / experiments)."""
    kw = {"k": s.retrieval_k}
    if filter:
        kw["filter"] = filter
    if s.search_type == "mmr":
        return store.as_retriever(search_type="mmr", search_kwargs={**kw, "fetch_k": s.fetch_k, "lambda_mult": s.mmr_lambda})
    if s.search_type == "threshold":
        return store.as_retriever(search_type="similarity_score_threshold", search_kwargs={**kw, "score_threshold": s.score_threshold})
    return store.as_retriever(search_type="similarity", search_kwargs=kw)


def hit_at_k(store, cases: list[dict], s, strategy: str, k: int) -> dict:
    """Retrieval-only evaluation: does any of the top-k chunks come from the ground-truth source AND
    contain a key fact? No LLM involved."""
    rows, hits = [], 0
    for c in cases:
        r = retrieve(store, c["question"], s, strategy=strategy, k=k)
        facts = [f.lower() for f in c.get("key_facts", [])]
        ok = any(d.metadata["source"] in c["sources"] and (not facts or any(f in d.page_content.lower() for f in facts))
                 for d, _ in r.hits)
        hits += ok
        rows.append({"id": c["id"], "question": c["question"], "hit": ok, "best_score": round(r.best_score, 3),
                     "top_sources": ", ".join(d.metadata["source"] for d, _ in r.hits[:3])})
    return {"strategy": strategy, "k": k, "hit_at_k": round(hits / len(cases), 3) if cases else 0, "rows": rows}
