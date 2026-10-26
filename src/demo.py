"""🎭 Demo mode: run the full RAG pipeline with NO API key and NO network.

DemoEmbeddings  - deterministic hashed bag-of-words (with light stemming + stop-words), 512-d.
DemoChat        - extractive "LLM": answers with the context sentences that best overlap the question
                  and cites the blocks they came from. It never invents text, so grounding/citation
                  logic, the relevance gate and the UI all behave exactly as with the real model.

Demo answers are a preview of the pipeline, not model-quality answers, and are labelled as such.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import replace

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda

from src.chains.prompts import REFUSAL
from src.chains.schemas import LLMAnswer

DIM = 512
DEMO_MODEL = "demo-hash-v1"
STOP = set("""a an the is are was were be been am do does did i me my you your we our it its this that these those of to in on
for at by with from and or but if so as can could would should will shall may might how what when where which who whom why
please tell about any there their them they he she his her have has had not no yes get got also just than then too very""".split())
SYN = {"beeping": "beep", "beeps": "beep", "alarm": "beep", "e-03": "e03", "e03": "e03", "refunds": "refund",
       "refunded": "refund", "delivered": "deliver", "delivery": "deliver", "shipping": "deliver", "ship": "deliver",
       "guarantee": "warranty", "warranties": "warranty", "cost": "price", "costs": "price", "prices": "price",
       "pricing": "price", "filters": "filter", "cartridge": "filter", "install": "installation", "installing": "installation"}
CHAT_HINTS = ("weather", "joke", "recipe", "cricket", "movie", "politics", "bitcoin")


def tokens(text: str) -> list[str]:
    out = []
    for w in re.findall(r"[a-z0-9][a-z0-9\-]*", text.lower()):
        if w in STOP or len(w) < 2:
            continue
        w = SYN.get(w, w)
        if len(w) > 4 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        out.append(w)
    return out


class DemoEmbeddings(Embeddings):
    def _vec(self, t: str) -> list[float]:
        v = [0.0] * DIM
        toks = tokens(t)
        for w in toks + [a + "_" + b for a, b in zip(toks, toks[1:])]:
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % DIM] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


_BLOCK = re.compile(r'<context_block label="(S\d+)">\n[^\n]*\n(.*?)</context_block>', re.S)


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    out = [re.sub(r"^[#>*\-\s|]+", "", p.replace("**", "")).strip() for p in parts]
    return [s for s in out if 12 < len(s) < 400 and not s.endswith("?")]


class DemoChat(BaseChatModel):
    """Extractive stand-in for the chat model (see module docstring)."""

    @property
    def _llm_type(self) -> str:
        return "demo-extractive"

    def _generate(self, messages, stop=None, run_manager=None, **kw) -> ChatResult:
        # Condense step: return the latest message, plus the previous question's keywords for short follow-ups.
        last = messages[-1].content
        m = re.search(r"Latest message: (.*)", last)
        q = (m.group(1) if m else last).strip()
        humans = [x.content for x in messages[:-1] if x.type == "human"]
        if humans and len(tokens(q)) < 3 and re.search(r"\b(it|that|this|those|they|them)\b", q, re.I):
            q = f"{q} ({' '.join(tokens(humans[-1])[:8])})"
        return ChatResult(generations=[ChatGeneration(message=AIMessage(q))])

    def with_structured_output(self, schema, **kw):
        def run(msgs):
            system = msgs[0].content
            q = msgs[-1].content if hasattr(msgs[-1], "content") else str(msgs[-1])
            qt = set(tokens(q))
            if any(h in q.lower() for h in CHAT_HINTS):
                a = LLMAnswer(answer="n/a", used_sources=[], confidence="none", answered=False, uncovered_parts=[],
                              out_of_scope=True, injection_attempt=False)
            elif re.search(r"ignore (all |any )?(previous|prior) instructions|system prompt|jailbreak", q, re.I):
                a = LLMAnswer(answer="I can't change my instructions, but I'm happy to help with any Pearlz question.",
                              used_sources=[], confidence="none", answered=False, uncovered_parts=[],
                              out_of_scope=False, injection_attempt=True)
            else:
                scored = []
                for label, body in _BLOCK.findall(system):
                    for s in _sentences(body):
                        st = set(tokens(s))
                        ov = len(qt & st)
                        if ov:
                            scored.append((ov / (1 + 0.04 * len(st)), label, s))
                scored.sort(key=lambda x: -x[0])
                best = scored[:2]
                if not best or best[0][0] < 0.8:
                    a = LLMAnswer(answer=REFUSAL, used_sources=[], confidence="none", answered=False,
                                  uncovered_parts=[q], out_of_scope=False, injection_attempt=False)
                else:
                    text = " ".join(s if s.endswith((".", "!", "?")) else s + "." for _, _, s in best)
                    labels = list(dict.fromkeys(l for _, l, _ in best))
                    conf = "high" if best[0][0] >= 2 else "partial"
                    a = LLMAnswer(answer=text, used_sources=labels, confidence=conf, answered=True,
                                  uncovered_parts=[], out_of_scope=False, injection_attempt=False)
            raw = AIMessage(json.dumps(a.model_dump(mode="json")), usage_metadata={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
            return {"raw": raw, "parsed": a, "parsing_error": None}
        return RunnableLambda(run)


def demo_settings(s):
    """Separate index + cache so demo vectors never mix with real OpenAI vectors."""
    return replace(s, embedding_model=DEMO_MODEL, collection_name="pearlz_demo", vector_store_dir=s.vector_store_dir / "demo",
                   score_threshold=min(s.score_threshold, 0.12), prompt_version="v3")
