"""Single validated settings object. No model name, chunk size, k or threshold is hard-coded elsewhere."""
from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
PROMPT_VERSIONS = ("v1", "v2", "v3")
SEARCH_TYPES = ("similarity", "mmr", "threshold")


class ConfigError(RuntimeError):
    pass


def mask(key: str | None) -> str:
    return "<missing>" if not key else (key[:3] + "…" + key[-4:] if len(key) > 10 else "***")


_SECRET = re.compile(r"sk-[A-Za-z0-9_\-]{8,}")


def scrub(text: str) -> str:
    return _SECRET.sub("sk-***redacted***", text)


@dataclass
class Settings:
    openai_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    temperature: float = 0.0
    kb_dir: Path = ROOT / "data" / "knowledge_base"
    vector_store_dir: Path = ROOT / "data" / "vector_store"
    cache_dir: Path = ROOT / "data" / "embedding_cache"
    log_dir: Path = ROOT / "logs"
    collection_name: str = "pearlz_kb"
    chunk_size: int = 800
    chunk_overlap: int = 120
    chunk_strategy: str = "structure"
    retrieval_k: int = 4
    fetch_k: int = 20
    search_type: str = "mmr"
    mmr_lambda: float = 0.6
    score_threshold: float = 0.30
    history_turns: int = 4
    max_context_chars: int = 6000
    prompt_version: str = "v3"
    request_timeout_s: float = 45.0
    max_retries: int = 3
    log_level: str = "INFO"

    @classmethod
    def from_env(cls, require_key: bool = True, **overrides) -> "Settings":
        load_dotenv(ROOT / ".env", override=False)
        s = cls()
        env = {
            "openai_api_key": "OPENAI_API_KEY", "llm_model": "LLM_MODEL", "embedding_model": "EMBEDDING_MODEL",
            "collection_name": "COLLECTION_NAME", "chunk_strategy": "CHUNK_STRATEGY", "search_type": "SEARCH_TYPE",
            "prompt_version": "PROMPT_VERSION", "log_level": "LOG_LEVEL",
        }
        for attr, var in env.items():
            if os.getenv(var):
                setattr(s, attr, os.environ[var].strip())
        for attr, var, typ in [("temperature", "TEMPERATURE", float), ("chunk_size", "CHUNK_SIZE", int),
                               ("chunk_overlap", "CHUNK_OVERLAP", int), ("retrieval_k", "RETRIEVAL_K", int),
                               ("fetch_k", "FETCH_K", int), ("mmr_lambda", "MMR_LAMBDA", float),
                               ("score_threshold", "SCORE_THRESHOLD", float), ("history_turns", "HISTORY_TURNS", int),
                               ("max_context_chars", "MAX_CONTEXT_CHARS", int), ("request_timeout_s", "REQUEST_TIMEOUT_S", float),
                               ("max_retries", "MAX_RETRIES", int)]:
            v = os.getenv(var)
            if v not in (None, ""):
                try:
                    setattr(s, attr, typ(v))
                except ValueError as e:
                    raise ConfigError(f"{var} must be a {typ.__name__}, got {v!r}") from e
        for attr, var in [("kb_dir", "KB_DIR"), ("vector_store_dir", "VECTOR_STORE_DIR")]:
            if os.getenv(var):
                setattr(s, attr, (ROOT / os.environ[var]).resolve())
        for k, v in overrides.items():
            if v is None:
                continue
            if not hasattr(s, k):
                raise ConfigError(f"Unknown setting: {k}")
            setattr(s, k, v)
        s.validate(require_key)
        return s

    def validate(self, require_key: bool = True) -> None:
        if require_key and not self.openai_api_key:
            raise ConfigError("Missing required variable OPENAI_API_KEY (set it in .env or enter it in the app sidebar).")
        if not 0 <= self.temperature <= 1:
            raise ConfigError("TEMPERATURE must be between 0 and 1 for grounded answers")
        if self.chunk_overlap >= self.chunk_size:
            raise ConfigError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        if not 1 <= self.retrieval_k <= self.fetch_k:
            raise ConfigError("Require 1 <= RETRIEVAL_K <= FETCH_K")
        if self.search_type not in SEARCH_TYPES:
            raise ConfigError(f"SEARCH_TYPE must be one of {SEARCH_TYPES}")
        if self.prompt_version not in PROMPT_VERSIONS:
            raise ConfigError(f"PROMPT_VERSION must be one of {PROMPT_VERSIONS}")
        if self.chunk_strategy not in ("structure", "recursive"):
            raise ConfigError("CHUNK_STRATEGY must be 'structure' or 'recursive'")
        if not Path(self.kb_dir).exists():
            raise ConfigError(f"KB_DIR does not exist: {self.kb_dir}")

    def public(self) -> dict:
        d = {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(self).items()}
        d["openai_api_key"] = mask(self.openai_api_key)
        return d


if __name__ == "__main__":
    try:
        print(json.dumps(Settings.from_env().public(), indent=2))
    except ConfigError as e:
        raise SystemExit(f"[config] {e}")
