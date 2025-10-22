from __future__ import annotations

from typing import Any, TypedDict

from app.settings import Settings
from infrastructure.llm.langchain_llm_provider import LangChainLLMProvider
from infrastructure.vectorstores.chroma_store import ChromaVectorStore


class RAGHit(TypedDict):
    content: str
    metadata: dict[str, Any]


class RAGResult(TypedDict):
    answer: str | None
    hits: list[RAGHit]


class QueryRAGUseCase:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.store = ChromaVectorStore(
            persist_dir=self.settings.chroma_dir,
            collection_name=self.settings.chroma_collection,
        )
        self.llm = LangChainLLMProvider(self.settings)

    def _retrieve(self, question: str) -> list[Any]:
        retriever = self.store.as_retriever(
            search_type=self.settings.retriever_search_type,
            k=self.settings.retriever_k,
        )
        docs = retriever.invoke(question)
        return docs

    def _to_hits(self, docs: list[Any]) -> list[RAGHit]:
        return [{"content": d.page_content, "metadata": d.metadata} for d in docs]

    def _to_context(self, docs: list[Any]) -> list[tuple[str, dict]]:
        return [(d.page_content, d.metadata) for d in docs]

    def execute(
        self,
        question: str,
        *,
        generate: bool = False,
        k: int | None = None,
        search_type: str | None = None,
    ) -> RAGResult:
        # Permite overrides por requisição
        if k is not None:
            self.settings.retriever_k = k
        if search_type:
            self.settings.retriever_search_type = search_type

        docs = self._retrieve(question)
        hits = self._to_hits(docs)

        answer: str | None = None
        if generate:
            answer = self.llm.generate(question, context_snippets=self._to_context(docs))

        return {"answer": answer, "hits": hits}
