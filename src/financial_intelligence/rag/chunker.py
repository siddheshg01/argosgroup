"""Boundary-aware text chunking that carries source/page/section metadata."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import re
from typing import Any, Iterable
from .document_loader import DocumentPage


@dataclass
class DocumentChunk:
    id: str
    text: str
    metadata: dict[str, Any]


def infer_section(text: str) -> str | None:
    known_headings = {"purpose", "scope", "definitions", "roles and responsibilities", "responsibilities",
                      "policy requirements", "requirements", "procedures", "controls", "documentation",
                      "exceptions", "monitoring and review", "enforcement", "approval", "effective date",
                      "accountability", "standards alignment note", "accounting and analytics"}
    for line in text.splitlines():
        candidate = line.strip().strip("# ")
        if candidate and (line.lstrip().startswith("#") or re.match(r"^\d+(?:\.\d+)*\.?\s+\S", candidate)
                          or candidate.casefold() in known_headings
                          or (len(candidate) < 100 and candidate.isupper())):
            return candidate
    return None


def _split_units(text: str) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    units: list[str] = []
    for para in paragraphs:
        if len(para) <= 1200:
            units.append(para)
        else:
            units.extend(s.strip() for s in re.split(r"(?<=[.!?])\s+", para) if s.strip())
    return units


def chunk_documents(documents: Iterable[DocumentPage], chunk_size: int = 1200,
                    overlap: int = 150) -> list[DocumentChunk]:
    """Create overlapping chunks, preferring paragraph and sentence boundaries."""
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Require chunk_size > overlap >= 0")
    output = []
    for document in documents:
        units = _split_units(document.text)
        current_section = infer_section(document.text) or document.metadata.get("section", "unknown")
        buffers: list[str] = []
        size = 0
        chunks: list[str] = []
        for unit in units:
            # Split unusually long single sentences so all pieces fit the bound.
            pieces = [unit[i:i + chunk_size] for i in range(0, len(unit), chunk_size)] if len(unit) > chunk_size else [unit]
            for piece in pieces:
                extra = len(piece) + (1 if buffers else 0)
                if buffers and size + extra > chunk_size:
                    chunks.append("\n".join(buffers))
                    tail = chunks[-1][-overlap:] if overlap else ""
                    buffers = [tail] if tail else []
                    size = len(tail)
                buffers.append(piece); size += len(piece) + (1 if size else 0)
        if buffers:
            chunks.append("\n".join(buffers))
        for index, text in enumerate(chunks):
            meta = dict(document.metadata)
            current_section = infer_section(text) or current_section
            meta.update({"chunk_index": index, "section": current_section})
            digest = hashlib.sha256((str(meta.get("document_id", "doc")) + f":{meta.get('page', '')}:{index}:{text}").encode()).hexdigest()[:24]
            output.append(DocumentChunk(digest, text, meta))
    return output
