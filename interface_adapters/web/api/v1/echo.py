from fastapi import APIRouter
from pydantic import BaseModel
from use_cases.query_simple import QuerySimpleUseCase

router = APIRouter(tags=["echo"])

class EchoRequest(BaseModel):
    question: str

class EchoResponse(BaseModel):
    answer: str

@router.post("/echo", response_model=EchoResponse)
def echo(req: EchoRequest) -> EchoResponse:
    uc = QuerySimpleUseCase()
    out = uc.execute(req.question)
    return EchoResponse(**out)
