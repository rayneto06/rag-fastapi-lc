from __future__ import annotations

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from app.settings import Settings
from domain.services.llm_provider import LLMProvider

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

    def generate(
        self, question: str, context_snippets: list[tuple[str, dict]] | None = None
    ) -> str:
        ctx_texts = []
        pages = []
        if context_snippets:
            for text, meta in context_snippets[:10]:
                ctx_texts.append(text)
                p = meta.get("page")
                if p is not None:
                    pages.append(p)

        context_text = "CONTEXT (use apenas o que segue):\n" + "\n\n---\n\n".join(
            t[:1200] for t in ctx_texts
        )
        available = (
            f"PÁGINAS DISPONÍVEIS NOS TRECHOS: {sorted(set(pages))}"
            if pages
            else "PÁGINAS DISPONÍVEIS: (não informadas)"
        )

        system = (
            "Você é um assistente conciso e fiel ao contexto.\n"
            "Para pedidos de RESUMO: produza um resumo apenas com o que estiver no contexto.\n"
            "Se faltar informação relevante, ainda assim entregue o melhor resumo possível e liste 'Limitações' no final."
        )
        messages = [
            SystemMessage(content=system),
            HumanMessage(content=f"{available}\n\n{context_text}\n\nPERGUNTA: {question}"),
        ]
        out = self._llm.invoke(messages)
        return getattr(out, "content", str(out))
