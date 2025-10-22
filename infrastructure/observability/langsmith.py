import os

from app.settings import Settings


def enable_langsmith(settings: Settings) -> None:
    """Configura variáveis de ambiente para LangSmith, se habilitado."""
    if settings.langchain_tracing_v2 and settings.langchain_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
        os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    else:
        os.environ.pop("LANGCHAIN_TRACING_V2", None)
