from __future__ import annotations
from datetime import datetime, UTC
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, status, HTTPException
from pydantic import BaseModel
from app.settings import Settings
from use_cases.ingest_documents import IngestDocumentsUseCase
from infrastructure.vectorstores.chroma_store import ChromaVectorStore


router = APIRouter(tags=["documents"])

class DocumentIngestResponse(BaseModel):
    filename: str
    uploaded_at: datetime
    num_docs: int
    num_chunks: int
    collection: str

class DocumentStatsResponse(BaseModel):
    collection: str
    persist_directory: str
    total_vectors: int

@router.post("/documents", response_model=DocumentIngestResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(file: UploadFile = File(...)) -> DocumentIngestResponse:
    settings = Settings()
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Apenas PDFs são aceitos.")

    raw_dir = Path(settings.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / file.filename
    data = await file.read()
    dest.write_bytes(data)

    uc = IngestDocumentsUseCase(settings=settings)
    num_docs, num_chunks = uc.execute(str(dest))

    return DocumentIngestResponse(
        filename=file.filename,
        uploaded_at=datetime.now(UTC),
        num_docs=num_docs,
        num_chunks=num_chunks,
        collection=settings.chroma_collection,
    )

@router.get("/documents", response_model=DocumentStatsResponse)
async def get_documents_stats() -> DocumentStatsResponse:
    settings = Settings()
    store = ChromaVectorStore(
        persist_dir=settings.chroma_dir,
        collection_name=settings.chroma_collection,
    )
    data = store.stats()
    return DocumentStatsResponse(**data)
