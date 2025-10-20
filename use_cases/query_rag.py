from __future__ import annotations
from typing import TypedDict, List, Dict, Any
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from infrastructure.vectorstores.chroma_store import ChromaVectorStore
from app.settings import Settings

class RAGHit(TypedDict):
    content: str
    metadata: Dict[str, Any]

class RAGResult(TypedDict):
    answer: str | None
    hits: List[RAGHit]

class QueryRAGUseCase:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.store = ChromaVectorStore(
            persist_dir=self.settings.chroma_dir,
            collection_name=self.settings.chroma_collection,
        )

    def _build_generate_chain(self, retriever):
        def _format_answer(inputs: Dict[str, Any]) -> str:
            question = inputs.get("question", "")
            docs = inputs.get("context", [])
            joined = "\n\n".join([d.page_content[:500] for d in docs[:3]])
            return f"Q: {question}\n\nA (based on retrieved context):\n{joined}"
        return RunnableParallel(context=retriever, question=RunnablePassthrough()) | RunnableLambda(_format_answer)

    def execute(self, question: str, *, generate: bool = False, k: int | None = None, search_type: str | None = None) -> RAGResult:
        retriever = self.store.as_retriever(
            search_type=search_type or self.settings.retriever_search_type,
            k=k or self.settings.retriever_k,
        )
        docs = retriever.invoke(question)
        hits: List[RAGHit] = [{"content": d.page_content, "metadata": d.metadata} for d in docs]
        answer: str | None = None
        if generate:
            chain = self._build_generate_chain(retriever)
            answer = chain.invoke(question)
        return {"answer": answer, "hits": hits}
