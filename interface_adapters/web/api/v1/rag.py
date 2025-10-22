from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from use_cases.query_rag import QueryRAGUseCase

router = APIRouter(tags=["rag"])


class RAGQueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    generate: bool = False
    k: int | None = None
    search_type: str | None = None  # "mmr" | "similarity" | etc.


class RAGHit(BaseModel):
    content: str
    metadata: dict[str, Any]


class RAGQueryResponse(BaseModel):
    answer: str | None = None
    hits: list[RAGHit]


@router.post("/rag/query", response_model=RAGQueryResponse)
def rag_query(req: RAGQueryRequest) -> RAGQueryResponse:
    uc = QueryRAGUseCase()
    out = uc.execute(req.question, generate=req.generate, k=req.k, search_type=req.search_type)
    return RAGQueryResponse(**out)
