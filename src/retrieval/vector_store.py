"""Create, persist, load and incrementally update the FAISS index.

Persistence: data/vector_store/<collection>.faiss/.pkl + manifest.json recording the embedding model,
dimension, chunking config and per-source file hashes + chunk ids.
Incremental update policy: delete-by-source-and-reinsert. When a file's hash changes, its old chunks
are DELETED (never left beside the new ones) and the new chunks inserted; removed files are deleted.
"""
from __future__ import annotations

import json
import logging
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore", message=".*langchain-community.*")
from langchain_community.vectorstores import FAISS  # noqa: E402

from src.ingestion.chunking import chunk_stats, split_documents  # noqa: E402
from src.ingestion.loaders import load_knowledge_base  # noqa: E402

log = logging.getLogger("supportpearlz.index")


class IndexError_(RuntimeError):
    pass


def _manifest_path(s) -> Path:
    return Path(s.vector_store_dir) / f"{s.collection_name}.manifest.json"


def read_manifest(s) -> dict | None:
    p = _manifest_path(s)
    return json.loads(p.read_text()) if p.exists() else None


def _write_manifest(s, store, chunks_by_source: dict, extra: dict | None = None):
    m = {"collection": s.collection_name, "embedding_model": s.embedding_model,
         "dimension": store.index.d, "chunk_size": s.chunk_size, "chunk_overlap": s.chunk_overlap,
         "chunk_strategy": s.chunk_strategy, "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
         "n_chunks": store.index.ntotal, "sources": chunks_by_source, **(extra or {})}
    _manifest_path(s).write_text(json.dumps(m, indent=2))
    return m


def build_index(s, embeddings, log_cb=None) -> tuple[FAISS, dict]:
    t0 = time.perf_counter()
    docs, report = load_knowledge_base(s.kb_dir)
    if not docs:
        raise IndexError_("No documents could be loaded from the knowledge base.")
    chunks = split_documents(docs, s.chunk_size, s.chunk_overlap, s.chunk_strategy)
    stats = chunk_stats(chunks)
    if stats["approx_max_tokens"] > 8000:
        log.warning("Some chunks may exceed the embedding model's input limit and be truncated.")
    store = FAISS.from_documents(chunks, embeddings, ids=[c.metadata["chunk_id"] for c in chunks],
                                 normalize_L2=True, distance_strategy="EUCLIDEAN_DISTANCE")
    Path(s.vector_store_dir).mkdir(parents=True, exist_ok=True)
    store.save_local(str(s.vector_store_dir), index_name=s.collection_name)
    by_src: dict = {}
    for c in chunks:
        e = by_src.setdefault(c.metadata["source"], {"file_hash": c.metadata["file_hash"], "chunk_ids": []})
        e["chunk_ids"].append(c.metadata["chunk_id"])
    m = _write_manifest(s, store, by_src, {"ingestion_report": report, "chunk_stats": stats,
                                           "build_seconds": round(time.perf_counter() - t0, 2)})
    log.info("Built index: %d chunks from %d sources in %.1fs", store.index.ntotal, len(by_src), time.perf_counter() - t0)
    return store, m


def load_index(s, embeddings) -> tuple[FAISS, dict]:
    m = read_manifest(s)
    faiss_file = Path(s.vector_store_dir) / f"{s.collection_name}.faiss"
    if not m or not faiss_file.exists():
        raise IndexError_("No persisted index found. Build it first (python -m src.ingestion.build_index).")
    if m["embedding_model"] != s.embedding_model:
        raise IndexError_(f"Index was built with '{m['embedding_model']}' but queries would use '{s.embedding_model}'. "
                          "Mixing embedding models silently breaks retrieval — rebuild the index.")
    t0 = time.perf_counter()
    # allow_dangerous_deserialization: the pickle is our own docstore written by build_index, never user-supplied.
    store = FAISS.load_local(str(s.vector_store_dir), embeddings, index_name=s.collection_name,
                             allow_dangerous_deserialization=True, normalize_L2=True)
    log.info("Loaded index from disk (%d chunks, model=%s) in %.2fs — no re-embedding", store.index.ntotal,
             m["embedding_model"], time.perf_counter() - t0)
    return store, m


def load_or_build(s, embeddings, rebuild: bool = False):
    if not rebuild:
        try:
            return load_index(s, embeddings) + ("loaded",)
        except IndexError_ as e:
            log.info("%s Building a new index.", e)
    return build_index(s, embeddings) + ("built",)


def sync_index(s, embeddings, store: FAISS) -> dict:
    """Incremental update: re-ingest changed/new files, delete chunks of removed/changed files."""
    m = read_manifest(s) or {"sources": {}}
    docs, report = load_knowledge_base(s.kb_dir)
    current: dict[str, list] = {}
    for d in docs:
        current.setdefault(d.metadata["source"], []).append(d)
    changed, added, removed = [], [], []
    for src, ds in current.items():
        h = ds[0].metadata["file_hash"]
        if src not in m["sources"]:
            added.append(src)
        elif m["sources"][src]["file_hash"] != h:
            changed.append(src)
    removed = [src for src in m["sources"] if src not in current]
    for src in changed + removed:
        ids = m["sources"][src]["chunk_ids"]
        if ids:
            store.delete(ids)  # superseded chunks removed, not accumulated
        m["sources"].pop(src, None)
    new_chunks = split_documents([d for src in changed + added for d in current[src]], s.chunk_size, s.chunk_overlap, s.chunk_strategy)
    if new_chunks:
        store.add_documents(new_chunks, ids=[c.metadata["chunk_id"] for c in new_chunks])
        for c in new_chunks:
            e = m["sources"].setdefault(c.metadata["source"], {"file_hash": c.metadata["file_hash"], "chunk_ids": []})
            e["chunk_ids"].append(c.metadata["chunk_id"])
    store.save_local(str(s.vector_store_dir), index_name=s.collection_name)
    _write_manifest(s, store, m["sources"], {"ingestion_report": report})
    out = {"added": added, "changed": changed, "removed": removed, "new_chunks": len(new_chunks), "total_chunks": store.index.ntotal}
    log.info("Index sync: %s", out)
    return out


def similarity_debug(store: FAISS, query: str, k: int = 5, filter: dict | None = None) -> list[dict]:
    """The debugging tool: score, source metadata and a preview for each hit."""
    from src.retrieval.retriever import scored_search
    rows = []
    for doc, score in scored_search(store, query, k, filter):
        rows.append({"score": round(float(score), 4), "chunk_id": doc.metadata.get("chunk_id"),
                     "source": doc.metadata.get("source"), "location": doc.metadata.get("location"),
                     "doc_type": doc.metadata.get("doc_type"), "preview": doc.page_content[:200].replace("\n", " ")})
    return rows
