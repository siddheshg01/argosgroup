import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.financial_intelligence.policy_analyst import (
    PolicyAnalyst, build_policy_prompt, validate_policy_response,
)
from src.financial_intelligence.rag.chunker import chunk_documents
from src.financial_intelligence.rag.document_loader import DocumentPage, load_document
from src.financial_intelligence.rag.embeddings import GeminiEmbedder
from src.financial_intelligence.rag.rag_pipeline import RAGPipeline
from src.financial_intelligence.rag.retriever import PolicyRetriever
from src.financial_intelligence.rag.vector_store import ChromaVectorStore


def test_text_and_docx_loading_preserve_metadata(tmp_path):
    txt = tmp_path / "returns.txt"
    txt.write_text("# RETURNS\nRefunds need approval.\n\nKeep proof.", encoding="utf-8")
    page = load_document(txt)[0]
    assert page.metadata["source"] == "returns.txt"
    assert page.metadata["page"] == 1
    assert "Refunds need approval" in page.text

    docx = pytest.importorskip("docx")
    document = docx.Document()
    document.add_heading("Payment approval", level=1)
    document.add_paragraph("Two approvals are required.")
    path = tmp_path / "approval.docx"
    document.save(path)
    pages = load_document(path)
    assert "Two approvals are required" in pages[0].text
    assert "page" not in pages[0].metadata


def test_chunking_keeps_page_section_and_unique_ids():
    text = "# REFUNDS\n" + ("Refund requests need documented approval. " * 100)
    source = DocumentPage(text, {"source": "policy.txt", "document_id": "policy", "page": 3})
    chunks = chunk_documents([source], chunk_size=180, overlap=20)
    assert len(chunks) > 1
    assert all(chunk.metadata["page"] == 3 for chunk in chunks)
    assert all(chunk.metadata["section"] == "REFUNDS" for chunk in chunks)
    assert len({chunk.id for chunk in chunks}) == len(chunks)
    assert max(len(chunk.text) for chunk in chunks) <= 180


def test_embedding_generation_uses_gemini_rest(monkeypatch):
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self): return json.dumps({"embedding": {"values": [0.1, 0.2, 0.3]}}).encode()
    def fake_urlopen(request, timeout):
        assert request.full_url.endswith("gemini-embedding-001:embedContent")
        assert request.headers["X-goog-api-key"] == "secret-test-key"
        payload = json.loads(request.data)
        assert payload["taskType"] == "RETRIEVAL_DOCUMENT"
        return Response()
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    vector = GeminiEmbedder(api_key="secret-test-key").embed("refund policy")
    assert vector == [0.1, 0.2, 0.3]


def test_chroma_vector_storage_and_semantic_search(tmp_path):
    chunks = chunk_documents([DocumentPage("approval evidence", {"source": "a.txt", "document_id": "a", "page": 1})])
    store = ChromaVectorStore(tmp_path / "vectors")
    store.replace_all(chunks, [[1.0, 0.0]])
    assert store.count == 1
    results = store.search([1.0, 0.0], top_k=3)
    assert results[0]["id"] == chunks[0].id
    assert results[0]["score"] == pytest.approx(1.0)


def test_retrieval_threshold_filters_low_relevance():
    class Embedder:
        def embed(self, text, task): return [0.0]
    class Store:
        def search(self, vector, top_k):
            return [{"id": "ok", "text": "policy", "metadata": {}, "score": 0.8},
                    {"id": "weak", "text": "noise", "metadata": {}, "score": 0.2}]
    got = PolicyRetriever(Store(), Embedder(), relevance_threshold=0.5).retrieve("refund", top_k=4)
    assert [item["id"] for item in got] == ["ok"]


def test_policy_prompt_separates_finance_from_retrieved_policy():
    prompt = build_policy_prompt("What applies?", [{"id": "abc", "text": "Approval is needed."}], {"revenue": 100})
    assert "not policy evidence" in prompt
    assert "Approval is needed" in prompt
    assert "Do not infer or invent policy rules" in prompt


def test_policy_citations_and_numeric_grounding_are_checked():
    retrieved = [{"id": "chunk1", "text": "Claims above 500 require manager approval.", "metadata": {}}]
    assert validate_policy_response({"claims": [{"text": "Above 500, manager approval is needed.", "source_chunk_ids": ["chunk1"]}],
                                     "confidence": "high", "limitations": []}, retrieved)["claims"]
    with pytest.raises(ValueError, match="Unsupported numeric"):
        validate_policy_response({"claims": [{"text": "Above 900, manager approval is needed.", "source_chunk_ids": ["chunk1"]}],
                                  "confidence": "high", "limitations": []}, retrieved)
    with pytest.raises(ValueError, match="source citation"):
        validate_policy_response({"claims": [{"text": "Approval is needed.", "source_chunk_ids": ["fake"]}],
                                  "confidence": "high", "limitations": []}, retrieved)


def test_no_result_returns_clear_answer_without_gemini(monkeypatch):
    monkeypatch.setattr("src.financial_intelligence.policy_analyst.call_policy_gemini",
                        lambda *args: pytest.fail("LLM must not run without retrieved policy evidence"))
    result = PolicyAnalyst().analyze("Question", [], {"financial_summary": "separate"})
    assert result["status"] == "no_relevant_documents"
    assert "No relevant policy" in result["answer"]


def test_mocked_gemini_policy_integration_and_validation(monkeypatch):
    retrieved = [{"id": "chunk-1", "text": "Refund requests require approval.", "metadata": {"source": "policy.txt"}, "score": 0.9}]
    monkeypatch.setattr("src.financial_intelligence.policy_analyst.call_policy_gemini", lambda prompt, **kwargs: {
        "claims": [{"text": "Refund requests require approval.", "source_chunk_ids": ["chunk-1"]}],
        "confidence": "high", "limitations": []})
    result = PolicyAnalyst().analyze("How are refunds approved?", retrieved, {"status": "generated"})
    assert result["status"] == "answered"
    assert result["claims"][0]["source_chunk_ids"] == ["chunk-1"]


def test_policy_analyst_retries_alternative_after_ungrounded_output(monkeypatch):
    retrieved = [{"id": "chunk-1", "text": "Refund requests require approval.", "metadata": {}, "score": 0.8}]
    monkeypatch.setenv("LLM_MODEL", "primary-model")
    monkeypatch.setenv("LLM_FALLBACK_MODELS", "fallback-model")
    attempts = []
    def fake_call(prompt, model=None, **kwargs):
        attempts.append(model)
        if model == "primary-model":
            return {"claims": [{"text": "Refunds above 500 require approval.", "source_chunk_ids": ["chunk-1"]}],
                    "confidence": "high", "limitations": []}
        return {"claims": [{"text": "Refund requests require approval.", "source_chunk_ids": ["chunk-1"]}],
                "confidence": "high", "limitations": []}
    monkeypatch.setattr("src.financial_intelligence.policy_analyst.call_policy_gemini", fake_call)
    result = PolicyAnalyst().analyze("How are refunds approved?", retrieved)
    assert result["status"] == "answered"
    assert attempts == ["primary-model", "fallback-model"]


def test_pipeline_empty_knowledge_base_writes_reports_without_api(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    result = RAGPipeline(tmp_path / "knowledge", tmp_path / "chroma", tmp_path / "output").run("Which policy applies?")
    assert result["status"] == "no_relevant_documents"
    assert result["retrieval_status"] == "no_documents"
    assert result["vector_store"] is True
    assert result["retrieved_documents"] == []
    assert (tmp_path / "output" / "rag_report.json").is_file()
    assert (tmp_path / "output" / "rag_summary.md").is_file()
