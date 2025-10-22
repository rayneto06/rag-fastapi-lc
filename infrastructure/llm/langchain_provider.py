from typing import Any

from langchain_core.runnables import RunnableLambda


def build_echo_chain():
    """Retorna um Runnable simples que ecoa a pergunta."""

    def _echo_fn(inputs: dict[str, Any]) -> dict[str, str]:
        q = inputs.get("question", "")
        return {"answer": f"echo: {q}"}

    return RunnableLambda(_echo_fn)
