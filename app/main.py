from fastapi import FastAPI
from app.container import build_app_state
from interface_adapters.web.api.v1.echo import router as echo_router
from interface_adapters.web.api.v1.documents import router as documents_router
from interface_adapters.web.api.v1.rag import router as rag_router

def create_app() -> FastAPI:
    app = FastAPI(title="rag-fastapi-lc", version="0.1.0")
    app.state.container = build_app_state()
    app.include_router(echo_router, prefix="/v1")
    app.include_router(documents_router, prefix="/v1")
    app.include_router(rag_router, prefix="/v1")
    return app

app = create_app()
