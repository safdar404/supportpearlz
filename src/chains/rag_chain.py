"""The composed RAG chain (LCEL):

  {question, history, filter}
    -> assign(standalone = condense)         history-aware query rewriter
    -> assign(retrieval  = retrieve)         retriever + relevance gate
    -> branch: gate failed  -> refusal path  (deterministic, no LLM call)
               gate passed  -> context builder [S1..Sk] -> prompt -> LLM (structured output, 1 retry, fallback)
    -> finalise: citation resolver/validator, confidence, safety checks
"""
from __future__ import annotations

import logging
import re
import time
import uuid

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough

from src.chains.memory import SessionMemory
from src.chains.prompts import CANARY, CONDENSE_PROMPT, OUT_OF_SCOPE, REFUSAL, answer_prompt
from src.chains.schemas import Citation, Confidence, GroundedResponse, LLMAnswer
from src.retrieval.retriever import retrieve
from src.utils.logging_setup import log_turn

log = logging.getLogger("supportpearlz.chain")
MAX_QUESTION_CHARS = 1000
INJECTION = re.compile(r"ignore (all |any )?(previous|prior|above) instructions|developer mode|system prompt|"
                       r"you are now|jailbreak|disregard (your|the) rules", re.I)


def make_llm(s):
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=s.llm_model, temperature=s.temperature, api_key=s.openai_api_key,
                      timeout=s.request_timeout_s, max_retries=s.max_retries)


def friendly_error(e: Exception) -> tuple[str, str]:
    name = type(e).__name__
    table = {
        "AuthenticationError": ("auth", "The AI service rejected the API key. Please check the key in the settings."),
        "RateLimitError": ("rate_limit", "The AI service is busy right now. Please try again in a minute."),
        "APITimeoutError": ("timeout", "The AI service took too long to respond. Please try again."),
        "APIConnectionError": ("network", "I couldn't reach the AI service — please check your internet connection."),
        "NotFoundError": ("model", "The configured model was not found. Check LLM_MODEL / EMBEDDING_MODEL."),
    }
    return table.get(name, ("internal", "Something went wrong while answering. Please try again or contact support."))


class SupportAgent:
    def __init__(self, s, store, llm=None):
        self.s, self.store = s, store
        self.llm = llm or make_llm(s)
        self.condense_chain = CONDENSE_PROMPT | self.llm | StrOutputParser()
        self.answer_llm = self.llm.with_structured_output(LLMAnswer, method="json_schema", include_raw=True, strict=True)
        self.chain = self._build()

    # ---------------------------------------------------------------- composition
    def _build(self):
        generate = (RunnablePassthrough.assign(context=RunnableLambda(self._format_context))
                    | RunnableLambda(self._generate))
        return (RunnablePassthrough.assign(standalone=RunnableLambda(self._condense))
                | RunnablePassthrough.assign(retrieval=RunnableLambda(self._retrieve))
                | RunnableBranch((lambda x: not x["retrieval"].gate_passed, RunnableLambda(self._refusal)), generate)
                | RunnableLambda(self._finalise))

    # ---------------------------------------------------------------- steps
    def _condense(self, x: dict) -> str:
        if not x["history"]:
            return x["question"]
        try:
            q = self.condense_chain.invoke({"history": x["history"], "question": x["question"]}).strip().strip('"')
            return q[:MAX_QUESTION_CHARS] or x["question"]
        except Exception as e:  # condensation failure must not kill the turn
            log.warning("Condensation failed (%s); using the raw question", type(e).__name__)
            x.setdefault("flags", []).append("condense_failed")
            return x["question"]

    def _retrieve(self, x: dict):
        return retrieve(self.store, x["standalone"], self.s, filter=x.get("filter") or None)

    def _format_context(self, x: dict) -> dict:
        blocks, labels, used, truncated = [], {}, 0, 0
        for i, (doc, score) in enumerate(x["retrieval"].hits, start=1):
            m = doc.metadata
            header = (f'[S{i}] source={m["source"]} location={m.get("location")} doc_type={m.get("doc_type")} '
                      f'version={m.get("version")} last_updated={m.get("last_updated")} relevance={score:.2f}')
            block = f"<context_block label=\"S{i}\">\n{header}\n{doc.page_content}\n</context_block>"
            if used + len(block) > self.s.max_context_chars:
                truncated += 1
                continue
            blocks.append(block)
            labels[f"S{i}"] = (doc, score)
            used += len(block)
        if truncated:
            log.info("Context budget: dropped %d block(s) beyond %d chars", truncated, self.s.max_context_chars)
        return {"text": "\n\n".join(blocks), "labels": labels, "truncated": truncated, "chars": used}

    def _generate(self, x: dict) -> dict:
        prompt = answer_prompt(self.s.prompt_version)
        msgs = prompt.format_messages(context=x["context"]["text"], question=x["standalone"], history=x["history"])
        tokens = {}
        for attempt in range(2):  # structured output: one retry with the error fed back, then typed fallback
            out = self.answer_llm.invoke(msgs)
            raw = out.get("raw")
            if raw is not None and getattr(raw, "usage_metadata", None):
                tokens = {"input": raw.usage_metadata.get("input_tokens"), "output": raw.usage_metadata.get("output_tokens")}
            if out.get("parsed") is not None and not out.get("parsing_error"):
                return {**x, "llm_answer": out["parsed"], "tokens": tokens}
            err = str(out.get("parsing_error"))[:500]
            log.warning("Structured output invalid (attempt %d): %s", attempt + 1, err)
            msgs = msgs + [("human", f"Your previous reply did not match the required schema ({err}). Reply again with valid fields.")]
        fallback = LLMAnswer(answer=REFUSAL, used_sources=[], confidence=Confidence.none, answered=False,
                             uncovered_parts=[x["question"]], out_of_scope=False, injection_attempt=False)
        return {**x, "llm_answer": fallback, "tokens": tokens, "flags": x.get("flags", []) + ["structured_output_fallback"]}

    def _refusal(self, x: dict) -> dict:
        ans = LLMAnswer(answer=REFUSAL, used_sources=[], confidence=Confidence.none, answered=False,
                        uncovered_parts=[x["question"]], out_of_scope=False, injection_attempt=False)
        return {**x, "llm_answer": ans, "context": {"labels": {}}, "tokens": {},
                "flags": x.get("flags", []) + ["pipeline_refusal"]}

    def _finalise(self, x: dict) -> GroundedResponse:
        a: LLMAnswer = x["llm_answer"]
        labels = x["context"].get("labels", {})
        flags = list(x.get("flags", []))
        answer, answered, conf = a.answer.strip(), a.answered, a.confidence
        valid = [l for l in dict.fromkeys(a.used_sources) if l in labels]
        invalid = [l for l in a.used_sources if l not in labels]
        if invalid:
            flags.append("invalid_citation:" + ",".join(invalid))  # fabricated label detected and dropped
        if CANARY in answer:
            flags.append("system_prompt_leak_blocked")
            answer, answered, valid = "I can't share my internal instructions, but I'm happy to help with any Pearlz question.", False, []
        if a.out_of_scope:
            answer, answered, valid = OUT_OF_SCOPE, False, []
            flags.append("out_of_scope")
        if a.injection_attempt:
            flags.append("injection_attempt")
        if answered and not valid:
            flags.append("ungrounded_answer_suppressed")  # an answer with no valid source is not shown as fact
            answer, answered = REFUSAL, False
        citations, seen = [], set()
        for l in valid:
            doc, _ = labels[l]
            key = (doc.metadata["source"], doc.metadata.get("location"))
            if key in seen:
                continue  # de-duplicate: same document location cited once
            seen.add(key)
            citations.append(Citation(label=l, source=doc.metadata["source"], location=str(doc.metadata.get("location")),
                                      doc_type=doc.metadata.get("doc_type", ""), version=str(doc.metadata.get("version", "")),
                                      snippet=re.sub(r"\s+", " ", doc.page_content)[:240]))
        best, thr = x["retrieval"].best_score, self.s.score_threshold
        if not answered:
            conf = Confidence.none
        elif a.confidence == Confidence.partial or a.uncovered_parts:
            conf = Confidence.partial
        elif a.confidence == Confidence.high and best >= thr + 0.15:
            conf = Confidence.high
        else:
            conf = Confidence.partial
        return GroundedResponse(
            answer=answer, sources=citations, confidence=conf, answered=answered,
            uncovered_parts=a.uncovered_parts if answered else [], flags=flags,
            original_question=x["question"], standalone_question=x["standalone"], gate=x["retrieval"].reason,
            retrieved=[{"label": f"S{i}", "chunk_id": d.metadata["chunk_id"], "score": round(sc, 4), "source": d.metadata["source"],
                        "location": d.metadata.get("location"), "preview": d.page_content[:220]}
                       for i, (d, sc) in enumerate(x["retrieval"].hits, start=1)],
            tokens=x.get("tokens", {}), prompt_version=self.s.prompt_version)

    # ---------------------------------------------------------------- public API
    def ask(self, question: str, memory: SessionMemory, session_id: str = "cli", filter: dict | None = None) -> GroundedResponse:
        t0 = time.perf_counter()
        q = (question or "").strip()
        pre_flags = []
        if not q:
            return GroundedResponse(answer="Please type a question.", flags=["empty_input"])
        if len(q) > MAX_QUESTION_CHARS:
            q = q[:MAX_QUESTION_CHARS]
            pre_flags.append("input_truncated")
        if INJECTION.search(q):
            pre_flags.append("injection_pattern_in_question")
        try:
            resp: GroundedResponse = self.chain.invoke({"question": q, "history": memory.messages(), "filter": filter, "flags": pre_flags})
        except Exception as e:
            kind, msg = friendly_error(e)
            log.error("Turn failed: %s: %s", type(e).__name__, str(e)[:300])
            resp = GroundedResponse(answer=msg, flags=[f"error:{kind}"] + pre_flags, original_question=q, standalone_question=q)
        resp.latency_ms = int((time.perf_counter() - t0) * 1000)
        if not any(f.startswith("error:") for f in resp.flags):
            memory.add(q, resp.answer)
        log_turn(self.s.log_dir, {
            "session_id": session_id, "turn_id": uuid.uuid4().hex[:8], "original_query": q,
            "rewritten_query": resp.standalone_question, "retrieved": [(r["chunk_id"], r["score"]) for r in resp.retrieved],
            "gate": resp.gate, "answered": resp.answered, "refusal": not resp.answered, "confidence": resp.confidence.value,
            "citations": [c.source + " " + c.location for c in resp.sources], "flags": resp.flags,
            "latency_ms": resp.latency_ms, "tokens": resp.tokens, "prompt_version": resp.prompt_version,
            "filter": filter})
        return resp
