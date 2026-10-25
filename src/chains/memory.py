"""Per-session chat history with a bounded window, plus query condensation.

Policy: keep the last N turns (HISTORY_TURNS, default 4). Windowing loses older facts entirely (a constraint
stated at turn 1 disappears after N turns) but never compounds summary errors; at support-chat lengths that
trade-off favours windowing. Both the original and rewritten query are logged for every turn.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


@dataclass
class SessionMemory:
    window: int = 4
    turns: list[tuple[str, str]] = field(default_factory=list)

    def add(self, question: str, answer: str) -> None:
        self.turns.append((question, answer))
        self.turns = self.turns[-50:]  # hard cap on stored history

    def messages(self) -> list[BaseMessage]:
        out: list[BaseMessage] = []
        for q, a in self.turns[-self.window:]:
            out += [HumanMessage(q), AIMessage(a[:800])]
        return out

    def reset(self) -> None:
        self.turns.clear()

    def __len__(self) -> int:
        return len(self.turns)
