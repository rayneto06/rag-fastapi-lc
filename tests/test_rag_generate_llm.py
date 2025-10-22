from pathlib import Path

import fitz  # PyMuPDF
import pytest
from httpx import ASGITransport, AsyncClient

from app.main import create_app
from app.settings import Settings


@pytest.mark.asyncio
async def test_rag_generate_with_llm_provider(tmp_path: Path):
    # 1) cria PDF e ingere
    pdf_path = tmp_path / "tiny_gen.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "RAG test with FAKE/REAL LLM generation. Conteúdo de teste.")
    doc.save(pdf_path)
    doc.close()

    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        with pdf_path.open("rb") as f:
            files = {"file": ("tiny_gen.pdf", f, "application/pdf")}
            resp = await ac.post("/v1/documents", files=files)
            assert resp.status_code == 201, resp.text

        # 2) generate=true -> resposta via LLMProvider (fake ou real)
        resp2 = await ac.post("/v1/rag/query", json={"question": "Summarize", "generate": True})
        assert resp2.status_code == 200, resp2.text
        data2 = resp2.json()
        assert isinstance(data2.get("answer"), str)

        provider = (Settings().llm_provider or "fake").lower()
        ans = data2["answer"].lower()

        if provider == "fake":
            assert "fake llm answer" in ans
        else:
            # Real provider (ollama/openai): só garantir que há conteúdo não-vazio
            assert ans.strip()
            assert "fake llm answer" not in ans
