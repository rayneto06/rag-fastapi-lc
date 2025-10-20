from __future__ import annotations
from typing import List, Tuple
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from infrastructure.loaders.pdf_loader import PDFLoaderAdapter
from infrastructure.vectorstores.chroma_store import ChromaVectorStore
from app.settings import Settings

class IngestDocumentsUseCase:
    """Carrega PDF, fatiando em chunks e persistindo no Chroma."""
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.loader = PDFLoaderAdapter()
        self.store = ChromaVectorStore(
            persist_dir=self.settings.chroma_dir,
            collection_name=self.settings.chroma_collection,
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            add_start_index=True,
        )

    def execute(self, filepath: str) -> Tuple[int, int]:
        docs = self.loader.load(filepath)  # list[Document]
        if not docs:
            return 0, 0
        chunks: List[Document] = self.splitter.split_documents(docs)
        added = self.store.add_documents(chunks)
        return len(docs), added
