from app.settings import AppState, Settings
from infrastructure.observability.langsmith import enable_langsmith


def build_app_state() -> AppState:
    settings = Settings()
    enable_langsmith(settings)
    return AppState(settings=settings)
