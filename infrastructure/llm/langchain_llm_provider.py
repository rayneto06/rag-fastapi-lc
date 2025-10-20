from __future__ import annotations

from typing import List, Tuple
from app.settings import Settings
from domain.services.llm_provider import LLMProvider

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.language_models.fake_chat_models import FakeListChatModel

try:
    from langchain_openai import ChatOpenAI  # type: ignore
except Exception:  # pragma: no cover
    ChatOpenAI = None  # type: ignore

try:
    from langchain_ollama import ChatOllama  # type: ignore
except Exception:  # pragma: no cover
    ChatOllama = None  # type: ignore


class LangChainLLMProvider(LLMProvider):
    """
    Provider de LLM baseado em LangChain:
      - openai: ChatOpenAI (precisa OPENAI_API_KEY)
      - ollama: ChatOllama (precisa Ollama rodando localmente)
      - fake:   FakeListChatModel (determinístico; ideal p/ testes)
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self._llm = self._build_llm()

    def _build_llm(self):
        provider = (self.settings.llm_provider or "fake").lower()
        model = self.settings.llm_model or "fake"
        temperature = float(self.settings.llm_temperature or 0.0)

        if provider == "openai":
            if ChatOpenAI is None:
                raise RuntimeError("langchain-openai não instalado.")
            return ChatOpenAI(model=model, temperature=temperature)
        elif provider == "ollama":
            if ChatOllama is None:
                raise RuntimeError("langchain-ollama não instalado.")
            # Base URL padrão é http://localhost:11434; se precisar customizar, exportear OLLAMA_BASE_URL
            return ChatOllama(model=model, temperature=temperature)
        else:
            # Fake determinístico para testes unitários
            return FakeListChatModel(responses=["This is a fake LLM answer."])

    def generate(self, question: str, context_snippets: List[Tuple[str, dict]] | None = None) -> str:
        context_text = ""
        if context_snippets:
            joined = "\n\n".join([text[:500] for text, _meta in context_snippets[:3]])
            context_text = f"Use ONLY the following context:\n{joined}\n\n"

        messages = [
            SystemMessage(content="You are a concise assistant. If the context is insufficient, say you cannot answer."),
            HumanMessage(content=f"{context_text}Question: {question}"),
        ]
        out = self._llm.invoke(messages)
        # Chat models retornam BaseMessage; pegar .content
        return getattr(out, "content", str(out))
