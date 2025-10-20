from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Observability (LangSmith)
    langchain_tracing_v2: bool = False
    langchain_api_key: str | None = None
    langsmith_project: str = "rag-fastapi-lc"

    # VectorStore / Chroma
    vector_store_provider: str = "chroma"
    chroma_dir: str = ".chroma"
    raw_dir: str = "data/raw"
    chroma_collection: str = "documents"

    # Embeddings
    embeddings_provider: str = "fake"  # fake | ollama | openai
    embeddings_model: str = "fake"     # e.g., nomic-embed-text (Ollama)

    # Query defaults
    retriever_search_type: str = "mmr"
    retriever_k: int = 5

    model_config = SettingsConfigDict(env_file=".env", env_prefix="", extra="ignore")

class AppState(BaseModel):
    settings: Settings
