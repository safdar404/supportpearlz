"""Chunking: recursive baseline + structure-aware splitting. Metadata is copied to every chunk.

structure: Markdown split on headings first; DOCX/PDF text split recursively and each chunk labelled with the
           nearest preceding numbered heading; CSV rows are never split.
recursive: plain RecursiveCharacterTextSplitter for everything except CSV rows.
"""
from __future__ import annotations

import re
import statistics

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

_HEADING = re.compile(r"^(#{1,4}\s+.+|\d+(\.\d+)*\.?\s+[A-Z][^\n]{2,80})$", re.M)


def _nearest_heading(text: str, pos: int) -> str | None:
    last = None
    for m in _HEADING.finditer(text):
        if m.start() > pos:
            break
        last = m.group(0).lstrip("#").strip()
    return last


def split_documents(docs: list[Document], chunk_size: int, chunk_overlap: int, strategy: str = "structure") -> list[Document]:
    rec = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True,
                                         separators=["\n\n", "\n", ". ", " ", ""])
    md = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")], strip_headers=False)
    out: list[Document] = []
    for d in docs:
        fmt = d.metadata.get("format")
        if fmt == "csv":
            out.append(Document(page_content=d.page_content, metadata=dict(d.metadata)))
            continue
        if strategy == "structure" and fmt == "md":
            for sec in md.split_text(d.page_content):
                heading = sec.metadata.get("h3") or sec.metadata.get("h2") or sec.metadata.get("h1")
                for piece in rec.split_text(sec.page_content):
                    meta = {**d.metadata, "section": heading or "Introduction"}
                    meta["location"] = f'§ "{meta["section"]}"'
                    out.append(Document(page_content=piece, metadata=meta))
            continue
        for piece in rec.split_documents([Document(page_content=d.page_content, metadata=dict(d.metadata))]):
            meta = dict(piece.metadata)
            if strategy == "structure":
                h = _nearest_heading(d.page_content, meta.get("start_index", 0) + 1)
                if h:
                    meta["section"] = h
            loc = []
            if "page" in meta:
                loc.append(f"p.{meta['page']}")
            if meta.get("section"):
                loc.append(f'"{meta["section"]}"')
            meta["location"] = ", ".join(loc) or meta.get("location", "document")
            out.append(Document(page_content=piece.page_content, metadata=meta))
    counters: dict[str, int] = {}
    for c in out:  # stable chunk ids: <source>::<n>
        n = counters.get(c.metadata["source"], 0)
        counters[c.metadata["source"]] = n + 1
        c.metadata["chunk_id"] = f"{c.metadata['source']}::{n:03d}"
        c.metadata.pop("start_index", None)
    return out


def chunk_stats(chunks: list[Document]) -> dict:
    lens = [len(c.page_content) for c in chunks] or [0]
    missing = sum(1 for c in chunks if not c.metadata.get("source") or not c.metadata.get("doc_type"))
    return {"chunks": len(chunks), "mean_len": round(statistics.mean(lens)), "max_len": max(lens),
            "min_len": min(lens), "chunks_missing_metadata": missing,
            "approx_max_tokens": max(lens) // 4}
