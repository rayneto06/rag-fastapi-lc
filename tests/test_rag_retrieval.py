import pytest
from pathlib import Path
import fitz  # PyMuPDF
from httpx import AsyncClient, ASGITransport
from app.main import create_app

TEXT = "LangChain & Chroma make RAG nice"

@pytest.mark.asyncio
async def test_rag_retrieval_flow(tmp_path: Path):
    # 1) cria PDF e ingere
    pdf_path = tmp_path / "tiny_rag.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), TEXT)
    doc.save(pdf_path)
    doc.close()

    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        with pdf_path.open("rb") as f:
            files = {"file": ("tiny_rag.pdf", f, "application/pdf")}
            resp = await ac.post("/v1/documents", files=files)
            assert resp.status_code == 201, resp.text

        # 2) retrieval-only
        resp2 = await ac.post("/v1/rag/query", json={"question": "What makes RAG nice?", "generate": False, "k": 3})
        assert resp2.status_code == 200, resp2.text
        data2 = resp2.json()
        assert len(data2["hits"]) >= 1

        # 3) generate=True usa o LLMProvider (FakeListChatModel)
        resp3 = await ac.post("/v1/rag/query", json={"question": "Summarize", "generate": True})
        assert resp3.status_code == 200, resp3.text
        data3 = resp3.json()
        assert isinstance(data3.get("answer"), str)
        assert "fake llm answer" in data3["answer"].lower()
