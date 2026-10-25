"""Interactive CLI chat.   python -m src.app_cli      (commands: /reset  /sources  /quit)"""
from __future__ import annotations

import uuid

from src.chains.memory import SessionMemory
from src.chains.rag_chain import SupportAgent
from src.config import ConfigError, Settings
from src.retrieval.embeddings import make_embeddings
from src.retrieval.vector_store import IndexError_, load_index
from src.utils.logging_setup import setup_logging


def main():
    try:
        s = Settings.from_env()
    except ConfigError as e:
        raise SystemExit(f"[config] {e}")
    setup_logging("WARNING", s.log_dir)
    try:
        store, m = load_index(s, make_embeddings(s))
    except IndexError_ as e:
        raise SystemExit(f"[index] {e}")
    agent, mem, sid = SupportAgent(s, store), SessionMemory(window=s.history_turns), uuid.uuid4().hex[:8]
    print(f"[Session {sid} | knowledge base: {s.collection_name} | {m['n_chunks']} chunks | {len(m['sources'])} documents]")
    last = None
    while True:
        try:
            q = input("\nYou > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q in ("/quit", "/exit"):
            break
        if q == "/reset":
            mem.reset()
            print("(conversation reset)")
            continue
        if q == "/sources" and last:
            for r in last.retrieved:
                print(f"  {r['label']} {r['score']:.3f} {r['source']} {r['location']}")
            continue
        last = agent.ask(q, mem, sid)
        print(f"Bot > {last.answer}")
        if last.standalone_question != q:
            print(f"      (searched for: {last.standalone_question})")
        if last.sources:
            print("Sources: " + "\n         ".join(f"[{i}] {c.source} — {c.location}" for i, c in enumerate(last.sources, 1)))
        else:
            print("Sources: none — no supporting document was found")
        print(f"Confidence: {last.confidence.value}" + (f"   flags: {', '.join(last.flags)}" if last.flags else ""))


if __name__ == "__main__":
    main()
