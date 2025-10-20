from __future__ import annotations
from typing import Any
from app.settings import Settings
from langchain_community.embeddings import FakeEmbeddings

try:
    from langchain_community.embeddings import OllamaEmbeddings  # optional
except Exception:  # pragma: no cover
    OllamaEmbeddings = None  # type: ignore

try:
    from langchain_openai import OpenAIEmbeddings  # optional
except Exception:  # pragma: no cover
    OpenAIEmbeddings = None  # type: ignore

class EmbeddingsProvider:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self._instance: Any = None

    @property
    def instance(self):
        if self._instance is not None:
            return self._instance
        provider = (self.settings.embeddings_provider or "fake").lower()
        model = self.settings.embeddings_model
        if provider == "ollama":
            if OllamaEmbeddings is None:
                raise RuntimeError("OllamaEmbeddings indisponível.")
            self._instance = OllamaEmbeddings(model=model or "nomic-embed-text")
        elif provider == "openai":
            if OpenAIEmbeddings is None:
                raise RuntimeError("OpenAIEmbeddings indisponível.")
            self._instance = OpenAIEmbeddings(model=model or "text-embedding-3-large")
        else:
            self._instance = FakeEmbeddings(size=1536)
        return self._instance
