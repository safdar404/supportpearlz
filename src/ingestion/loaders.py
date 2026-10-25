"""Loader layer: walk the KB folder, dispatch by extension, normalise text, attach the metadata contract.

Metadata on every Document: source, doc_type, product_line, version, last_updated, and a location
pointer (page for PDFs; row/SKU for CSV; section is added at chunking time for text formats).
Errors are handled per file: a bad file is logged and skipped, never fatal to the run.
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import re
import warnings
from collections import Counter
from pathlib import Path

from langchain_core.documents import Document

warnings.filterwarnings("ignore", message=".*langchain-community.*")
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader  # noqa: E402

log = logging.getLogger("supportpearlz.ingestion")
SUPPORTED = {".pdf", ".docx", ".md", ".txt", ".csv"}
DOC_TYPE_GUESS = {".pdf": "manual", ".docx": "policy", ".md": "guide", ".txt": "guide", ".csv": "pricing"}


def _normalise(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _strip_repeated_lines(pages: list[Document]) -> list[Document]:
    """Remove header/footer lines that repeat on most PDF pages (page numbers normalised away)."""
    if len(pages) < 3:
        return pages
    norm = lambda l: re.sub(r"\d+", "#", l.strip())
    counts = Counter(n for p in pages for n in {norm(l) for l in p.page_content.splitlines() if l.strip()})
    repeated = {l for l, c in counts.items() if c >= 0.6 * len(pages) and len(l) < 120}
    for p in pages:
        p.page_content = "\n".join(l for l in p.page_content.splitlines() if norm(l) not in repeated)
    return pages


def _load_csv(path: Path) -> list[Document]:
    """Row-wise, self-describing sentences, so a price can never be split from its SKU."""
    docs = []
    with path.open(encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f), start=2):  # row 1 is the header
            row = {k: (v or "").strip() for k, v in row.items()}
            parts = [f"SKU {row.get('sku')}: {row.get('product_name')} ({row.get('product_line')}, {row.get('category')})."]
            if row.get("description"):
                parts.append(f"Description: {row['description']}.")
            if row.get("list_price_pkr"):
                parts.append(f"List price: PKR {int(row['list_price_pkr']):,}.")
            if row.get("promo_price_pkr"):
                parts.append(f"Current promotional price: PKR {int(row['promo_price_pkr']):,}.")
            parts.append(f"Availability: {row.get('availability') or 'unknown'}.")
            if row.get("replacement_interval"):
                parts.append(f"Replace every {row['replacement_interval']}.")
            docs.append(Document(page_content=" ".join(parts),
                                 metadata={"row": i, "sku": row.get("sku"), "location": f"row {i} (SKU {row.get('sku')})",
                                           "product_line_row": row.get("product_line")}))
    return docs


def load_file(path: Path) -> list[Document]:
    ext = path.suffix.lower()
    if ext == ".pdf":
        pages = PyPDFLoader(str(path)).load()
        pages = _strip_repeated_lines(pages)
        for p in pages:
            p.metadata["page"] = int(p.metadata.get("page", 0)) + 1  # 1-based for humans
            p.metadata["location"] = f"p.{p.metadata['page']}"
        return pages
    if ext == ".docx":
        return Docx2txtLoader(str(path)).load()
    if ext in (".md", ".txt"):
        return TextLoader(str(path), encoding="utf-8").load()
    if ext == ".csv":
        return _load_csv(path)
    raise ValueError(f"unsupported extension {ext}")


def load_knowledge_base(kb_dir: Path) -> tuple[list[Document], list[dict]]:
    kb_dir = Path(kb_dir)
    manifest = {}
    mf = kb_dir / "_metadata.json"
    if mf.exists():
        manifest = json.loads(mf.read_text(encoding="utf-8"))
    docs, report, seen = [], [], set()
    for path in sorted(kb_dir.rglob("*")):
        if not path.is_file() or path.name.startswith(("_", ".")):
            continue
        rel = path.relative_to(kb_dir).as_posix()
        if path.suffix.lower() not in SUPPORTED:
            report.append({"file": rel, "type": path.suffix, "status": "skipped", "reason": "unsupported format"})
            continue
        try:
            raw = load_file(path)
        except Exception as e:  # corrupt / password-protected / unreadable: log and continue
            log.warning("Skipping unreadable file %s: %s", rel, e)
            report.append({"file": rel, "type": path.suffix, "status": "skipped", "reason": f"{type(e).__name__}: {str(e)[:120]}"})
            continue
        meta = manifest.get(rel, {})
        file_hash = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        kept, chars, dups = 0, 0, 0
        for d in raw:
            d.page_content = _normalise(d.page_content)
            if not d.page_content:
                continue
            h = hashlib.sha256(d.page_content.encode()).hexdigest()
            if h in seen:
                dups += 1
                continue
            seen.add(h)
            base = {"source": rel, "doc_type": meta.get("doc_type", DOC_TYPE_GUESS[path.suffix.lower()]),
                    "product_line": meta.get("product_line", "all"), "version": meta.get("version", "unversioned"),
                    "last_updated": meta.get("last_updated", "unknown"), "file_hash": file_hash, "format": path.suffix.lower()[1:]}
            base.update({k: v for k, v in d.metadata.items() if k not in ("source", "file_path", "total_pages")
                         and k in ("page", "row", "sku", "location", "product_line_row")})
            if d.metadata.get("product_line_row"):
                base["product_line"] = d.metadata["product_line_row"]
            d.metadata = base
            docs.append(d)
            kept += 1
            chars += len(d.page_content)
        if kept == 0:
            report.append({"file": rel, "type": path.suffix, "status": "skipped", "reason": "empty after extraction"})
            continue
        unit = "pages" if path.suffix.lower() == ".pdf" else "rows" if path.suffix.lower() == ".csv" else "parts"
        report.append({"file": rel, "type": path.suffix[1:], "status": "loaded", "units": f"{kept} {unit}", "chars": chars,
                       "duplicates_skipped": dups, "metadata_keys": ", ".join(sorted(docs[-1].metadata)), "reason": ""})
    loaded = sum(1 for r in report if r["status"] == "loaded")
    log.info("Ingestion: %d files found, %d loaded, %d skipped", len(report), loaded, len(report) - loaded)
    return docs, report
