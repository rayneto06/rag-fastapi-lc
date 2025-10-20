from __future__ import annotations
from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
from use_cases.query_rag import QueryRAGUseCase

router = APIRouter(tags=["rag"])

class RAGQueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    generate: bool = False
    k: Optional[int] = None
    search_type: Optional[str] = None  # "mmr" | "similarity" | etc.

class RAGHit(BaseModel):
    content: str
    metadata: Dict[str, Any]

class RAGQueryResponse(BaseModel):
    answer: Optional[str] = None
    hits: List[RAGHit]

@router.post("/rag/query", response_model=RAGQueryResponse)
def rag_query(req: RAGQueryRequest) -> RAGQueryResponse:
    uc = QueryRAGUseCase()
    out = uc.execute(req.question, generate=req.generate, k=req.k, search_type=req.search_type)
    return RAGQueryResponse(**out)
