from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from app.auth import make_verify_api_key
from app.config import Settings


def build_app(api_key: str) -> FastAPI:
    settings = Settings(
        model_id="x", device="cpu", compute_type="int8",
        max_concurrent_transcriptions=1, max_file_size_mb=1,
        api_key=api_key, host="0.0.0.0", port=8000,
    )
    verify_api_key = make_verify_api_key(settings)
    app = FastAPI()

    @app.get("/protected", dependencies=[Depends(verify_api_key)])
    async def protected():
        return {"ok": True}

    return app


def test_valid_api_key_passes():
    client = TestClient(build_app("secret123"))
    resp = client.get("/protected", headers={"Authorization": "Bearer secret123"})
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


def test_missing_api_key_rejected():
    client = TestClient(build_app("secret123"))
    resp = client.get("/protected")
    assert resp.status_code == 401
    assert resp.json()["detail"]["error"]["type"] == "invalid_api_key"


def test_wrong_api_key_rejected():
    client = TestClient(build_app("secret123"))
    resp = client.get("/protected", headers={"Authorization": "Bearer wrong"})
    assert resp.status_code == 401
