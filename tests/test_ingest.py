import pytest
import fitz  # PyMuPDF
from httpx import AsyncClient, ASGITransport
from pathlib import Path
from app.main import create_app

@pytest.mark.asyncio
async def test_ingest_pdf_ok(tmp_path: Path):
    pdf_path = tmp_path / "tiny.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Hello RAG with LangChain & Chroma!")
    doc.save(pdf_path)
    doc.close()

    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        with pdf_path.open("rb") as f:
            files = {"file": ("tiny.pdf", f, "application/pdf")}
            resp = await ac.post("/v1/documents", files=files)

    assert resp.status_code == 201, resp.text
    data = resp.json()
    assert data["filename"] == "tiny.pdf"
    assert data["num_docs"] >= 1
    assert data["num_chunks"] >= 1
