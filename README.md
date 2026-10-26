# 🤖 SupportPearlz — RAG Customer Support Agent

**AAI-407 / PRAC-03.** A LangChain retrieval-augmented support assistant for the fictional Pearlz Home Systems. It ingests 12 company documents (PDF, DOCX, Markdown, CSV) once into a persistent FAISS index, then answers multi-turn customer questions by rewriting follow-ups into standalone queries, retrieving relevant passages, and generating a schema-validated answer with validated citations, or an honest refusal.

## ✨ What's new (Prism AI UI)

- 🎭 **Demo mode** — try the whole pipeline with no API key: offline hashed embeddings + an extractive stand-in that quotes the docs with real citations (`src/demo.py`). Clearly labelled; add a key for real AI answers.
- 🌈 **Prism AI theme** — deep-space base, animated rainbow accents, glass cards, AI-agent logo (`assets/agent_logo.svg`), compact mobile layout.
- ✨ Suggested-question pills, 👍/👎 feedback on every answer, 📊 live session stats, 📥 Markdown export of the conversation.
- 📖 Document viewer + 🔍 find-in-documents (offline) in the Knowledge tab.
- 🔒 Turn log now shows only the current session (other visitors' questions were visible before); uploads capped at 5 MB.
- 🐛 Fixed: sidebar brand text invisible on the dark theme; Turn-log table failing Arrow serialisation; similarity debugger re-embedding on every rerun (now a form).

## Prerequisites
- Python 3.10+ (tested on 3.12)
- An OpenAI project API key with a spend limit (embeddings + chat)

## Setup
```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                       # or enter the key in the Streamlit sidebar
python -m src.config                                       # validated settings, key masked
python -m src.ingestion.build_index --report               # ingestion report, no API calls
python -m src.ingestion.build_index                        # build once; later runs load from disk
```

## Run
| What | Command |
|---|---|
| 🎨 Streamlit app (enter API key in the sidebar) | `streamlit run app.py` |
| CLI chat (`/reset`, `/sources`, `/quit`) | `python -m src.app_cli` |
| Force a full rebuild | `python -m src.ingestion.build_index --rebuild` |
| Incremental update after editing a document | `python -m src.ingestion.build_index --sync` |
| Full evaluation (22 cases) | `python -m evaluation.run_eval` |
| Retrieval-only hit@k by strategy | `python -m evaluation.run_eval --retrieval-only` |
| Offline tests (no key, no cost) | `pytest -q` |
| Regenerate the PDF/DOCX documents | `python scripts/build_kb_documents.py` |

The app's **🔬 Retrieval lab** runs the Task 03 chunking experiment, Task 04 embedding diagnostic, Task 06 strategy comparison and threshold calibration, and the Task 08 prompt before/after, all from buttons.

## Environment variables
| Variable | Meaning | Default |
|---|---|---|
| `OPENAI_API_KEY` | API key (or enter in app) | — |
| `LLM_MODEL` / `EMBEDDING_MODEL` | Chat / embedding models | gpt-4o-mini / text-embedding-3-small |
| `TEMPERATURE` | Sampling temperature | 0.0 |
| `KB_DIR` / `VECTOR_STORE_DIR` / `COLLECTION_NAME` | Paths and index name | data/knowledge_base / data/vector_store / pearlz_kb |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` / `CHUNK_STRATEGY` | Chunking | 800 / 120 / structure |
| `RETRIEVAL_K` / `FETCH_K` / `SEARCH_TYPE` / `MMR_LAMBDA` | Retrieval | 4 / 20 / mmr / 0.6 |
| `SCORE_THRESHOLD` | Relevance gate (cosine; calibrate!) | 0.30 |
| `HISTORY_TURNS` / `MAX_CONTEXT_CHARS` | Memory window / context budget | 4 / 6000 |
| `PROMPT_VERSION` | v1 (weak) / v2 / v3 (final) | v3 |
| `REQUEST_TIMEOUT_S` / `MAX_RETRIES` / `LOG_LEVEL` | Reliability / logging | 45 / 3 / INFO |

## Architecture (as built)
```
INGESTION (offline)  data/knowledge_base ─► loaders.py (PyPDF / Docx2txt / Text / row-wise CSV, metadata contract,
                     per-file errors, dedupe) ─► chunking.py (markdown-header + recursive, nearest-heading labels,
                     CSV rows unsplit) ─► CachedEmbeddings (SQLite, resumable) ─► FAISS + manifest (persisted)

QUERY (per turn)     question + history ─► condense (LLM) ─► retrieve (similarity | MMR | threshold, metadata filter)
                     ─► relevance gate ──fail──► deterministic refusal (no LLM call)
                                       └pass──► context builder [S1..Sk] + budget ─► prompt v3 ─► LLM structured output
                     (1 retry, typed fallback) ─► citation validator (drop fabricated labels, de-dupe, suppress uncited
                     answers, canary leak check) ─► GroundedResponse ─► memory + JSONL turn log
```
Composed with LCEL: `RunnablePassthrough.assign(...) | RunnableBranch(...) | RunnableLambda(...)` in `src/chains/rag_chain.py`.

## Knowledge base
12 authored documents in `data/knowledge_base/` (19 PDF pages; ~6,500 words; **extend to the required 9–12k words**). Gaps, seams and the injection plant are documented in `data/PLANTED_GAPS_AND_SEAMS.md`, which sits outside the KB so it is never indexed.

## Known limitations
- Pure dense retrieval can miss exact identifiers (SKUs, "E-05"); hybrid BM25 (Challenge C1) would help.
- The relevance gate cannot separate every in-domain-but-unanswered question (e.g. "NSF certification" is close to filter text); the v3 refusal rule catches those, probabilistically.
- History windowing forgets constraints older than N turns.
- `langchain-community` (FAISS, loaders) is marked as being sunset; pin versions and track its replacement packages.
