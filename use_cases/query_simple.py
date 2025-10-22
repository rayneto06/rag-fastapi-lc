from typing import TypedDict

from infrastructure.llm.langchain_provider import build_echo_chain


class EchoInput(TypedDict):
    question: str


class EchoOutput(TypedDict):
    answer: str


class QuerySimpleUseCase:
    def __init__(self) -> None:
        self._chain = build_echo_chain()

    def execute(self, question: str) -> EchoOutput:
        payload: EchoInput = {"question": question}
        result = self._chain.invoke(payload)
        return {"answer": result["answer"]}
