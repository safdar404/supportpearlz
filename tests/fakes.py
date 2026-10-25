"""Offline test doubles: a deterministic bag-of-words embedding and a scripted chat model.
They let every pipeline path run for free; they are NOT used for evaluation evidence."""
from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda

from src.chains.schemas import LLMAnswer

DIM = 256
SYN = {"beeps": "alert", "beep": "alert", "beeping": "alert", "alarm": "alert", "three": "e03", "e-03": "e03"}


class HashEmbeddings(Embeddings):
    def _vec(self, t: str):
        v = [0.0] * DIM
        for w in re.findall(r"[a-z0-9\-]+", t.lower()):
            w = SYN.get(w, w)
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % DIM] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


class FakeChat(BaseChatModel):
    """Condense: echoes the question (prefixing 'returns' context for ellipsis). Answer: cites S1."""
    @property
    def _llm_type(self) -> str:
        return "fake-chat"

    def _generate(self, messages, stop=None, run_manager=None, **kw) -> ChatResult:
        last = messages[-1].content
        m = re.search(r"Latest message: (.*)\n", last)
        q = m.group(1) if m else last
        if m and len(messages) > 2 and len(q.split()) < 6:
            prev = [x.content for x in messages if x.type == "human"][0]
            q = f"{q} (about: {prev})"
        return ChatResult(generations=[ChatGeneration(message=AIMessage(q))])

    def with_structured_output(self, schema, **kw):
        def run(msgs):
            sys = msgs[0].content
            q = msgs[-1].content.lower() if hasattr(msgs[-1], "content") else str(msgs[-1]).lower()
            labels = re.findall(r'label="(S\d+)"', sys)
            if "weather" in q:
                a = LLMAnswer(answer="n/a", used_sources=[], confidence="none", answered=False, uncovered_parts=[],
                              out_of_scope=True, injection_attempt=False)
            elif "ignore all previous" in q:
                a = LLMAnswer(answer="I can't do that.", used_sources=[], confidence="none", answered=False,
                              uncovered_parts=[], out_of_scope=False, injection_attempt=True)
            elif "fake citation" in q:
                a = LLMAnswer(answer="Answer.", used_sources=["S9"], confidence="high", answered=True, uncovered_parts=[],
                              out_of_scope=False, injection_attempt=False)
            else:
                first = re.search(r'<context_block label="S1">\n.*\n(.*)', sys)
                a = LLMAnswer(answer=(first.group(1)[:200] if first else "none"), used_sources=labels[:2] + labels[:1],
                              confidence="high", answered=bool(labels), uncovered_parts=[], out_of_scope=False,
                              injection_attempt=False)
            raw = AIMessage(json.dumps(a.model_dump(mode="json")), usage_metadata={"input_tokens": 500, "output_tokens": 60, "total_tokens": 560})
            return {"raw": raw, "parsed": a, "parsing_error": None}
        return RunnableLambda(run)
