"""Evaluation harness (Task 11).   python -m evaluation.run_eval [--retrieval-only] [--cases T-01 T-12]

Automated scoring (manual review still expected — see evaluation/manual_scoring.csv):
  correctness     expected behaviour met: answer contains a key fact / refusal when expected
  groundedness    every cited chunk is one that was actually retrieved, and a key fact appears in a cited snippet
  citation_acc    answered cases cite at least one ground-truth source and no invalid labels
  hallucination   answered an unanswerable case, or produced forbidden content (must_not_contain)
Writes evaluation/results/eval_<stamp>.json and .md
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.chains.memory import SessionMemory  # noqa: E402
from src.chains.rag_chain import SupportAgent  # noqa: E402
from src.config import Settings  # noqa: E402
from src.retrieval.embeddings import make_embeddings  # noqa: E402
from src.retrieval.retriever import hit_at_k  # noqa: E402
from src.retrieval.vector_store import load_or_build  # noqa: E402


def load_cases(ids=None) -> list[dict]:
    cases = json.loads((ROOT / "evaluation" / "test_questions.json").read_text())["cases"]
    return [c for c in cases if not ids or c["id"] in ids]


def score_case(case: dict, responses: list) -> dict:
    r = responses[-1]
    text = r.answer.lower()
    facts = [f.lower() for f in case.get("key_facts", [])]
    fact_hit = any(f in text for f in facts) if facts else True
    forbidden = [x for x in case.get("must_not_contain", []) if x.lower() in text]
    cited = {c.source for c in r.sources}
    retrieved_ids = {x["source"] for x in r.retrieved}
    exp = case["expect"]
    if exp == "refuse":
        correct = not r.answered and not forbidden
    elif exp == "answer_or_clarify":
        correct = (r.answered and fact_hit) or ("?" in r.answer and not forbidden)  # options listed or one clarifying question
    else:
        correct = r.answered and fact_hit and not forbidden
    grounded = (not r.answered) or (cited <= retrieved_ids and (not facts or any(
        any(f in c.snippet.lower() for f in facts) for c in r.sources) or fact_hit))
    cit_ok = (not r.answered) or (bool(cited & set(case["sources"])) and not any(f.startswith("invalid_citation") for f in r.flags))
    halluc = (exp == "refuse" and r.answered) or bool(forbidden)
    extra = {}
    if case.get("rewrite_must_not_contain"):
        bad = [w for w in case["rewrite_must_not_contain"] if w in r.standalone_question.lower()]
        extra["rewrite_contaminated"] = bad
        correct = correct and not bad
    if case.get("consistency"):
        answers = [x.answered for x in responses]
        srcs = [frozenset(c.source for c in x.sources) for x in responses]
        extra["consistent"] = len(set(answers)) == 1 and all(s & srcs[0] for s in srcs)
        correct = correct and extra["consistent"]
    verdict = "pass" if correct and grounded and cit_ok and not halluc else "partial" if correct else "fail"
    stage = None
    if verdict != "pass":
        hit_src = bool(retrieved_ids & set(case["sources"]))
        stage = ("retrieval/chunking" if case["sources"] and not hit_src
                 else "relevance gate (threshold)" if "pipeline_refusal" in r.flags and exp != "refuse"
                 else "query rewrite (memory)" if extra.get("rewrite_contaminated")
                 else "prompt / model" )
    return {"id": case["id"], "category": case["category"], "question": case["turns"][-1],
            "rewritten": r.standalone_question, "answer": r.answer, "answered": r.answered,
            "confidence": r.confidence.value, "citations": [f"{c.source} — {c.location}" for c in r.sources],
            "retrieved": [x["chunk_id"] for x in r.retrieved], "flags": r.flags, "latency_ms": r.latency_ms,
            "correct": correct, "grounded": grounded, "citation_accurate": cit_ok, "hallucination": halluc,
            "forbidden_found": forbidden, **extra, "verdict": verdict, "suspect_stage": stage}


def run(agent, cases, progress=None) -> dict:
    rows = []
    for i, c in enumerate(cases):
        mem = SessionMemory(window=agent.s.history_turns)
        resps = [agent.ask(t, mem, session_id=f"eval-{c['id']}") for t in c["turns"]]
        rows.append(score_case(c, resps))
        if progress:
            progress(i + 1, len(cases), c["id"])
    unans = [r for r in rows if r["category"] in ("unanswerable",) or r["id"] == "T-17"]
    answerable = [r for r in rows if r not in unans]
    agg = {
        "cases": len(rows), "pass": sum(r["verdict"] == "pass" for r in rows),
        "correctness": round(sum(r["correct"] for r in rows) / len(rows), 3),
        "groundedness": round(sum(r["grounded"] for r in rows) / len(rows), 3),
        "citation_accuracy": round(sum(r["citation_accurate"] for r in answerable) / max(1, len(answerable)), 3),
        "refusal_rate_unanswerable": round(sum(not r["answered"] for r in unans) / max(1, len(unans)), 3),
        "hallucination_count": sum(r["hallucination"] for r in rows),
        "mean_latency_ms": int(sum(r["latency_ms"] for r in rows) / len(rows)),
        "by_category": {cat: f"{sum(r['verdict'] == 'pass' for r in rows if r['category'] == cat)}/"
                             f"{sum(1 for r in rows if r['category'] == cat)}" for cat in sorted({r['category'] for r in rows})},
    }
    return {"settings": agent.s.public(), "aggregate": agg, "rows": rows}


def retrieval_only(store, s, cases) -> list[dict]:
    rc = [{"id": c["id"], "question": c["turns"][-1] if len(c["turns"]) == 1 else c["turns"][0],
           "sources": c["sources"], "key_facts": c["key_facts"][:2]} for c in cases if c["sources"] and len(c["turns"]) == 1]
    out = []
    for strat, k in [("similarity", 3), ("similarity", 8), ("mmr", 4), ("threshold", 4)]:
        r = hit_at_k(store, rc, s, strat, k)
        out.append({"strategy": strat, "k": k, "hit_at_k": r["hit_at_k"], "rows": r["rows"]})
    return out


def write(result: dict) -> Path:
    d = ROOT / "evaluation" / "results"
    d.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    (d / f"eval_{stamp}.json").write_text(json.dumps(result, indent=2, default=str))
    L = ["| ID | Category | Verdict | Correct | Grounded | Citation | Halluc. | Suspect stage |", "|---|---|---|---|---|---|---|---|"]
    for r in result["rows"]:
        L.append(f"| {r['id']} | {r['category']} | {r['verdict']} | {r['correct']} | {r['grounded']} | {r['citation_accurate']} | "
                 f"{r['hallucination']} | {r['suspect_stage'] or '—'} |")
    L += ["", "```", json.dumps(result["aggregate"], indent=2), "```"]
    (d / f"eval_{stamp}.md").write_text("\n".join(L))
    return d / f"eval_{stamp}.md"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="*")
    ap.add_argument("--retrieval-only", action="store_true")
    a = ap.parse_args()
    s = Settings.from_env()
    store, _, _ = load_or_build(s, make_embeddings(s))
    cases = load_cases(a.cases)
    if a.retrieval_only:
        for r in retrieval_only(store, s, cases):
            print(f"{r['strategy']:10s} k={r['k']}  hit@k={r['hit_at_k']}")
        return
    res = run(SupportAgent(s, store), cases, lambda i, n, cid: print(f"[{i}/{n}] {cid}", flush=True))
    print(json.dumps(res["aggregate"], indent=2))
    print("written:", write(res))


if __name__ == "__main__":
    main()
