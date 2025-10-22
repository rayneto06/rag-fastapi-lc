import pytest
from httpx import ASGITransport, AsyncClient

from app.main import create_app


@pytest.mark.asyncio
async def test_echo_ok():
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.post("/v1/echo", json={"question": "ping"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"] == "echo: ping"
