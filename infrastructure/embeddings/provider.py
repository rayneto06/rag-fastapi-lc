from __future__ import annotations

from langchain_community.embeddings import FakeEmbeddings
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings

from app.settings import Settings


class EmbeddingsProvider:
    """Provider centralizado de embeddings, com cache e fallback."""

    _instance = None

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()

    @property
    def instance(self):
        if self._instance is None:
            self._instance = self._build()
        return self._instance

    def _build(self):
        provider = (self.settings.embeddings_provider or "fake").lower()
        model = self.settings.embeddings_model or "fake"

        if provider == "openai":
            return OpenAIEmbeddings(model=model)
        elif provider == "ollama":
            # Usa o novo pacote oficial (langchain_ollama)
            return OllamaEmbeddings(model=model or "nomic-embed-text")
        else:
            # Fallback determinístico para testes
            return FakeEmbeddings(size=1536)
