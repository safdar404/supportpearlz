"""Embedding model + a persistent SQLite cache so a crash at 80% never forces a full re-embed,
and so chunking experiments don't pay twice for the same text."""
from __future__ import annotations

import hashlib
import logging
import random
import sqlite3
import struct
import time
from pathlib import Path

from langchain_core.embeddings import Embeddings

log = logging.getLogger("supportpearlz.embeddings")


class CachedEmbeddings(Embeddings):
    def __init__(self, inner: Embeddings, model_name: str, cache_dir: Path, batch_size: int = 64, max_retries: int = 4):
        self.inner, self.model_name, self.batch_size, self.max_retries = inner, model_name, batch_size, max_retries
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(Path(cache_dir) / "embeddings.sqlite"), check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS emb (k TEXT PRIMARY KEY, v BLOB)")
        self.stats = {"cached": 0, "embedded": 0, "seconds": 0.0}

    def _key(self, text: str) -> str:
        return hashlib.sha256(f"{self.model_name}\x00{text}".encode()).hexdigest()

    @staticmethod
    def _pack(v: list[float]) -> bytes:
        return struct.pack(f"{len(v)}f", *v)

    @staticmethod
    def _unpack(b: bytes) -> list[float]:
        return list(struct.unpack(f"{len(b) // 4}f", b))

    def _with_retry(self, fn, *a):
        for attempt in range(self.max_retries + 1):
            try:
                return fn(*a)
            except Exception as e:  # transient API failures: exponential backoff with jitter
                if attempt == self.max_retries:
                    raise
                delay = min(20, 2 ** attempt) + random.uniform(0, 1)
                log.warning("Embedding call failed (%s); retry %d in %.1fs", type(e).__name__, attempt + 1, delay)
                time.sleep(delay)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        t0 = time.perf_counter()
        keys = [self._key(t) for t in texts]
        found = {}
        for i in range(0, len(keys), 500):
            q = keys[i:i + 500]
            rows = self.db.execute(f"SELECT k, v FROM emb WHERE k IN ({','.join('?' * len(q))})", q).fetchall()
            found.update({k: self._unpack(v) for k, v in rows})
        todo = [i for i, k in enumerate(keys) if k not in found]
        for s in range(0, len(todo), self.batch_size):
            batch = todo[s:s + self.batch_size]
            vecs = self._with_retry(self.inner.embed_documents, [texts[i] for i in batch])
            self.db.executemany("INSERT OR REPLACE INTO emb VALUES (?, ?)",
                                [(keys[i], self._pack(v)) for i, v in zip(batch, vecs)])
            self.db.commit()  # committed per batch => resumable
            for i, v in zip(batch, vecs):
                found[keys[i]] = v
        self.stats["cached"] += len(texts) - len(todo)
        self.stats["embedded"] += len(todo)
        self.stats["seconds"] += time.perf_counter() - t0
        if todo:
            log.info("Embedded %d new texts (%d from cache) in %.1fs", len(todo), len(texts) - len(todo), time.perf_counter() - t0)
        return [found[k] for k in keys]

    def embed_query(self, text: str) -> list[float]:
        return self._with_retry(self.inner.embed_query, text)


def make_embeddings(settings, inner: Embeddings | None = None) -> CachedEmbeddings:
    if inner is None:
        from langchain_openai import OpenAIEmbeddings
        inner = OpenAIEmbeddings(model=settings.embedding_model, api_key=settings.openai_api_key,
                                 timeout=settings.request_timeout_s, max_retries=0)
    return CachedEmbeddings(inner, settings.embedding_model, settings.cache_dir, max_retries=settings.max_retries)


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0
