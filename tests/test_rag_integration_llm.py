import os
from pathlib import Path

import fitz  # PyMuPDF
import pytest
from httpx import ASGITransport, AsyncClient

from app.main import create_app

pytestmark = pytest.mark.integration  # marca o módulo todo como "integration"


def _should_run_integration() -> tuple[bool, str]:
    """Decide se o teste deve rodar, com base nas variáveis de ambiente."""
    provider = (os.getenv("LLM_PROVIDER") or "fake").lower()
    if provider not in {"openai", "ollama"}:
        return False, f"LLM_PROVIDER must be 'openai' or 'ollama' (got '{provider}')."
    if provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        return False, "OPENAI_API_KEY is required for provider=openai."
    # Para ollama, assumo http://localhost:11434 por padrão; não verifico rede aqui.
    return True, ""


@pytest.mark.asyncio
async def test_rag_integration_real_llm(tmp_path: Path):
    should_run, reason = _should_run_integration()
    if not should_run:
        pytest.skip(reason)

    # 1) cria PDF e ingere
    pdf_path = tmp_path / "tiny_integration.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(
        (72, 72), "Este é um documento de teste para o fluxo de integração com LLM real."
    )
    doc.save(pdf_path)
    doc.close()

    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        with pdf_path.open("rb") as f:
            files = {"file": ("tiny_integration.pdf", f, "application/pdf")}
            resp = await ac.post("/v1/documents", files=files)
            assert resp.status_code == 201, resp.text

        # 2) consulta com generate=True (usa LLMProvider real)
        resp2 = await ac.post(
            "/v1/rag/query",
            json={"question": "Resuma o documento de teste.", "generate": True, "k": 3},
        )
        assert resp2.status_code == 200, resp2.text
        data2 = resp2.json()

        # Asserções brandas (não sei o texto exato do modelo)
        answer = data2.get("answer")
        assert isinstance(answer, str), "answer must be a string"
        assert len(answer.strip()) > 0, "answer should not be empty"
        assert "fake llm answer" not in answer.lower(), "should be a real LLM answer"
        assert isinstance(data2.get("hits"), list) and len(data2["hits"]) >= 1
