"""Load plain text, PDF, and Word documents with page-level metadata."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import re
from typing import Any

SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx"}


@dataclass
class DocumentPage:
    text: str
    metadata: dict[str, Any]


def clean_text(text: str) -> str:
    """Normalize whitespace while retaining paragraph boundaries."""
    text = text.replace("\x00", "").replace("\x7f", " • ").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\t\f\v ]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def load_document(path: str | Path) -> list[DocumentPage]:
    """Load a supported file into page/section units with source metadata."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Knowledge document not found: {file_path}")
    suffix = file_path.suffix.lower()
    base = {"source": file_path.name, "document_id": file_path.stem}
    if suffix == ".txt":
        pages = [(1, file_path.read_text(encoding="utf-8-sig", errors="replace"))]
    elif suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError("PDF support requires pypdf; install requirements.txt") from exc
        reader = PdfReader(str(file_path))
        pages = [(index, page.extract_text() or "") for index, page in enumerate(reader.pages, 1)]
    elif suffix == ".docx":
        try:
            from docx import Document
        except ImportError as exc:
            raise RuntimeError("DOCX support requires python-docx; install requirements.txt") from exc
        doc = Document(str(file_path))
        # Word does not expose stable page boundaries; use headings as section units.
        paragraphs = []
        for paragraph in doc.paragraphs:
            text = paragraph.text.strip()
            if not text:
                continue
            style_name = getattr(paragraph.style, "name", "") or ""
            paragraphs.append(f"## {text}" if style_name.lower().startswith("heading") else text)
        pages = [(None, "\n".join(paragraphs))]
        for table_index, table in enumerate(doc.tables, 1):
            rows = [" | ".join(cell.text.strip() for cell in row.cells) for row in table.rows]
            pages.append((None, "\n".join(rows)))
            base.setdefault("table_count", table_index)
    else:
        raise ValueError(f"Unsupported document type: {suffix or '(no extension)'}")

    result: list[DocumentPage] = []
    for page_num, raw in pages:
        text = clean_text(raw)
        if not text:
            continue
        metadata = {**base}
        if page_num is not None:
            metadata["page"] = page_num
        result.append(DocumentPage(text, metadata))
    return result


def load_knowledge_base(directory: str | Path) -> list[DocumentPage]:
    """Load every supported document from a folder in deterministic order."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    pages = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
                        and p.stem.lower() != "readme"):
        relative = path.relative_to(root)
        loaded = load_document(path)
        for page in loaded:
            page.metadata["source"] = relative.as_posix()
            page.metadata["document_id"] = relative.with_suffix("").as_posix()
        pages.extend(loaded)
    return pages
