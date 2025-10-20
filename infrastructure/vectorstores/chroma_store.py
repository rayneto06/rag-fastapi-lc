from __future__ import annotations

from langchain_chroma import Chroma
from infrastructure.embeddings.provider import EmbeddingsProvider

class ChromaVectorStore:
    def __init__(self, persist_dir: str, collection_name: str = "documents") -> None:
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self._vs: Chroma | None = None

    def _ensure_vs(self) -> Chroma:
        if self._vs is None:
            embeddings = EmbeddingsProvider().instance
            self._vs = Chroma(
                collection_name=self.collection_name,
                persist_directory=self.persist_dir,
                embedding_function=embeddings,
            )
        return self._vs

    def add_documents(self, documents):
        vs = self._ensure_vs()
        vs.add_documents(documents)
        return len(documents)

    def as_retriever(self, search_type: str = "mmr", k: int = 5):
        return self._ensure_vs().as_retriever(search_type=search_type, search_kwargs={"k": k})
