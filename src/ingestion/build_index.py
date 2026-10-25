"""Ingestion entry point (offline path). Run once per knowledge-base update.

  python -m src.ingestion.build_index            load existing index or build it
  python -m src.ingestion.build_index --rebuild  force a full rebuild
  python -m src.ingestion.build_index --sync     incremental: delete-by-source and reinsert changed files
  python -m src.ingestion.build_index --report   print the ingestion report only (no embedding)
"""
from __future__ import annotations

import argparse
import json

from src.config import ConfigError, Settings
from src.ingestion.chunking import chunk_stats, split_documents
from src.ingestion.loaders import load_knowledge_base
from src.retrieval.embeddings import make_embeddings
from src.retrieval.vector_store import load_index, load_or_build, sync_index
from src.utils.logging_setup import setup_logging


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--sync", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args(argv)
    try:
        s = Settings.from_env(require_key=not a.report)
    except ConfigError as e:
        raise SystemExit(f"[config] {e}")
    setup_logging(s.log_level, s.log_dir)
    if a.report:
        docs, report = load_knowledge_base(s.kb_dir)
        for r in report:
            print(f"{r['status']:8s} {r['file']:40s} {r.get('units', ''):12s} chars={r.get('chars', '-'):>6} {r.get('reason', '')}")
        print(json.dumps(chunk_stats(split_documents(docs, s.chunk_size, s.chunk_overlap, s.chunk_strategy)), indent=2))
        return
    emb = make_embeddings(s)
    if a.sync:
        store, _ = load_index(s, emb)
        print(json.dumps(sync_index(s, emb, store), indent=2))
        return
    store, manifest, how = load_or_build(s, emb, rebuild=a.rebuild)
    print(f"Index {how}: {store.index.ntotal} chunks · model={manifest['embedding_model']} · dim={manifest['dimension']}")
    print(f"Embedding cache: {emb.stats}")


if __name__ == "__main__":
    main()
