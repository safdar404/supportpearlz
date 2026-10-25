# SupportPearlz — Technical report (skeleton)

> Design rationale below is fixed by the code. Sections marked **[MEASURE]** need numbers from your own runs:
> the rubric awards the measurement and the reasoning, not the claim.

## 1. Architecture
Two executable paths: ingestion writes to disk (`build_index.py`); querying only reads (`load_index`). A manifest records the embedding model, dimension and chunk config; loading with a different query embedding model raises an error instead of silently returning near-random results.

## 2. Chunking (Task 03) — **[MEASURE]**
Run 🔬 Lab → *Run chunking experiment* (7 configs: 300/800/1500 × 0/15% overlap + structure-aware). Paste the table. Expected trade-off: small chunks raise precision but strip context (e.g. split a policy clause from its condition); large chunks average several clauses into one vector. Structure-aware splitting keeps Markdown sections and CSV rows intact so a price is never separated from its SKU.

## 3. Embeddings and vector store (Tasks 04–05)
Model `text-embedding-3-small` (record the dimension from the manifest). Cosine is used because the vectors are normalised, so direction carries meaning and magnitude doesn't. Mixing models breaks retrieval because the two vector spaces have unrelated geometry. **[MEASURE]** paste the diagnostic output. Update policy: delete-by-source-and-reinsert (`sync_index`), so superseded chunks never coexist with new ones.

## 4. Retrieval (Task 06) — **[MEASURE]**
Paste the strategy comparison (similarity k=3, k=8, MMR k=4, threshold). Paste the calibration result and state the chosen threshold and why. The score is cosine similarity derived from FAISS squared-L2 on normalised vectors (`1 − d/2`); the number is only meaningful for this model + metric + store.

## 5. Prompt evolution (Task 08) — **[MEASURE]**
v1 → v2 → v3 are in `src/chains/prompts.py`. Run the before/after in the Lab and quote one concrete failure of v1 (typically a guessed answer to a planted gap, or agreeing to the 5-year warranty premise) eliminated by v3. Injection: the handbook plant and T-17 are refused; the canary check blocks system-prompt leakage in code.

## 6. Conversation (Task 09)
Condensation before retrieval; window of N turns (see `memory.py` for the trade-off). Both queries are logged in `logs/turns.jsonl`. **[MEASURE]** paste a three-turn log plus the topic-switch case T-11.

## 7. Citations (Task 10)
The model returns the labels it used; code intersects them with the labels supplied, drops fabricated ones (flag `invalid_citation`), de-duplicates by source + location, and suppresses any "answered" response with no valid citation. **[MEASURE]** manually verify 10 answers at the cited location.

## 8. Evaluation (Task 11) — **[MEASURE]**
Paste `evaluation/results/eval_*.md`, aggregate metrics, and a failure analysis attributing every non-pass to retrieval, chunking, gate, rewrite, prompt or model.

## 9. Limitations and next two weeks
Hybrid search, cross-encoder reranking, a judge-model groundedness check validated against manual labels.
