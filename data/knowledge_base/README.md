# Policy knowledge base

Place your approved financial policies, procedures, and business documents in this folder. Supported file types are PDF, TXT, and DOCX. The Phase 5 pipeline indexes only files actually placed here; an empty folder deliberately produces a no-policy-found result.

Keep confidential material out unless you are permitted to send document excerpts to the configured Gemini API for embeddings and policy analysis. The local Chroma index is stored under `data/processed/policy_chroma/` and is ignored by Git.

Set `POLICY_QUESTION` in `.env` to ask a question when running `python main.py`. You can also call `RAGPipeline().run(query="...")` from Python.
