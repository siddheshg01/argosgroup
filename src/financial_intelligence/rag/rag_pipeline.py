"""End-to-end policy document ingestion, retrieval, grounded analysis, and reporting."""
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any
from ..llm_analyst import load_local_env
from ..policy_analyst import PolicyAnalyst
from .chunker import chunk_documents
from .document_loader import load_knowledge_base
from .embeddings import GeminiEmbedder
from .retriever import PolicyRetriever
from .vector_store import ChromaVectorStore


def _write_report(report: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "rag_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# Financial Policy Intelligence", "", f"**Query:** {report['query']}", "",
             f"**Status:** {report.get('status')}", f"**Confidence:** {report.get('confidence')}", "",
             "## Answer", "", report["answer"], "", "## Retrieved sources", ""]
    if report["source_citations"]:
        for citation in report["source_citations"]:
            location = f", page {citation['page']}" if citation.get("page") else ""
            lines.append(f"- `{citation['chunk_id']}` — {citation['source']}{location}, section: {citation.get('section', 'unknown')}; relevance {citation['relevance_score']:.3f}")
    else:
        lines.append("No relevant policy documents were retrieved.")
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {limitation}" for limitation in report["limitations"])
    (output_dir / "rag_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


class RAGPipeline:
    """Index the current policy folder and answer one question from retrieved passages."""
    def __init__(self, knowledge_dir: str | Path = "data/knowledge_base",
                 vector_dir: str | Path = "data/processed/policy_chroma",
                 output_dir: str | Path = "output", relevance_threshold: float | None = None):
        self.knowledge_dir, self.vector_dir, self.output_dir = map(Path, (knowledge_dir, vector_dir, output_dir))
        self.relevance_threshold = relevance_threshold

    def run(self, query: str | None = None, financial_analysis: dict[str, Any] | None = None) -> dict[str, Any]:
        load_local_env()
        query = (query or os.getenv("POLICY_QUESTION") or
                 "What policy or procedure is relevant to the current financial findings?").strip()
        if not query:
            query = "What policy or procedure is relevant to the current financial findings?"
        threshold = self.relevance_threshold
        if threshold is None:
            threshold = float(os.getenv("RAG_RELEVANCE_THRESHOLD", "0.25"))
        report: dict[str, Any] = {
            "query": query, "retrieved_documents": [], "relevance_scores": [], "source_citations": [],
            "context": "", "answer": "No relevant policy document was found in the knowledge base.",
            "confidence": "low", "limitations": [], "status": "no_relevant_documents",
            "documents_indexed": 0, "chunks_created": 0, "vector_store": False,
        "retrieval_status": "not_run", "financial_context_source": "Phase 4 report, kept separate from policy evidence",
        }
        store = None
        try:
            store = ChromaVectorStore(self.vector_dir)
            report["vector_store"] = True
            pages = load_knowledge_base(self.knowledge_dir)
            extraction_warnings = sorted({
                f"Text extraction from {page.metadata.get('source')} page {page.metadata.get('page')} contained unreadable characters; verify the cited passage in the original file."
                for page in pages if "\ufffd" in page.text
            })
            chunks = chunk_documents(pages)
            report["documents_indexed"] = len({p.metadata["document_id"] for p in pages})
            report["chunks_created"] = len(chunks)
            if not chunks:
                store.replace_all([], [])
                report["limitations"] = ["The knowledge base contains no supported, readable policy documents. Add PDF, TXT, or DOCX files to data/knowledge_base/."]
                report["retrieval_status"] = "no_documents"
            else:
                embedder = GeminiEmbedder()
                existing = store.get_existing_embeddings([c.id for c in chunks])
                missing = [chunk for chunk in chunks if chunk.id not in existing]
                new_vectors = embedder.embed_many([c.text for c in missing], "RETRIEVAL_DOCUMENT")
                existing.update({chunk.id: vector for chunk, vector in zip(missing, new_vectors)})
                vectors = [existing[chunk.id] for chunk in chunks]
                store.replace_all(chunks, vectors)
                retriever = PolicyRetriever(store, embedder, threshold)
                retrieved = retriever.retrieve(query, top_k=int(os.getenv("RAG_TOP_K", "5")))
                report["retrieval_status"] = "complete"
                report["retrieved_documents"] = retrieved
                report["relevance_scores"] = [{"chunk_id": item["id"], "score": item["score"]} for item in retrieved]
                report["context"] = "\n\n".join(
                    f"[{item['id']}] {item['metadata'].get('source')} | section={item['metadata'].get('section')} | page={item['metadata'].get('page', 'n/a')}\n{item['text']}"
                    for item in retrieved)
                report["source_citations"] = [{"chunk_id": item["id"], "source": item["metadata"].get("source", "unknown"),
                    "document_id": item["metadata"].get("document_id"), "section": item["metadata"].get("section"),
                    "page": item["metadata"].get("page"), "relevance_score": item["score"]} for item in retrieved]
                analysis = PolicyAnalyst().analyze(query, retrieved, financial_analysis)
                report.update({"answer": analysis["answer"], "confidence": analysis["confidence"],
                               "limitations": extraction_warnings + analysis["limitations"], "status": analysis["status"],
                               "claims": analysis["claims"]})
                if not retrieved:
                    report["limitations"] = analysis["limitations"]
        except Exception as exc:
            report["status"] = "error"
            report["limitations"].append(str(exc))
            if not report["retrieval_status"].startswith("no_"):
                report["retrieval_status"] = "error"
        _write_report(report, self.output_dir)
        return report
