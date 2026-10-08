"""Gemini text embeddings for semantic document retrieval."""
from __future__ import annotations
import json
import os
import urllib.error
import urllib.request
from typing import Sequence
from ..llm_analyst import load_local_env


class GeminiEmbedder:
    """Call the Gemini embedding endpoint; no API key is stored in this object."""
    def __init__(self, api_key: str | None = None, model: str = "gemini-embedding-001",
                 timeout: float = 30, output_dimension: int = 768):
        load_local_env()
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("LLM_API_KEY")
        self.model, self.timeout, self.output_dimension = model, timeout, output_dimension

    def embed(self, text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> list[float]:
        """Embed one text string for document storage or query matching."""
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is required for document embeddings")
        if task_type not in {"RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"}:
            raise ValueError("task_type must be RETRIEVAL_DOCUMENT or RETRIEVAL_QUERY")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:embedContent"
        body = {"model": f"models/{self.model}", "content": {"parts": [{"text": text}]},
                "taskType": task_type, "outputDimensionality": self.output_dimension}
        request = urllib.request.Request(url, data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json", "x-goog-api-key": self.api_key},
                                         method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
            vector = result["embedding"]["values"]
            if not isinstance(vector, list) or not vector:
                raise ValueError("Gemini returned an empty embedding")
            return [float(value) for value in vector]
        except urllib.error.HTTPError as exc:
            try:
                message = json.loads(exc.read().decode("utf-8", errors="replace")).get("error", {}).get("message", "")
            except (ValueError, AttributeError):
                message = ""
            raise RuntimeError(f"Gemini embeddings returned HTTP {exc.code}: {message}".rstrip()) from None
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Gemini embedding network error: {type(exc.reason).__name__}") from None
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError("Invalid Gemini embedding response") from exc

    def embed_many(self, texts: Sequence[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
        """Embed texts in stable order; sequential calls respect simple API limits."""
        return [self.embed(text, task_type) for text in texts]
