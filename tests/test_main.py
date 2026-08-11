import io
import time

from fastapi.testclient import TestClient

from app import main as main_module
from app.config import Settings
from app.main import app, get_transcription_service, job_store, verify_api_key
from app.transcribe import SegmentResult, TranscriptionResult, WordResult

app.dependency_overrides[verify_api_key] = lambda: None


class _FakeReadyService:
    queue_depth = 0


# Tests exercise the transcription endpoint via dependency_overrides on
# get_transcription_service rather than by running the real lifespan, so
# app.state.transcription_service is never populated by startup here.
# Simulate a "ready" app so the /health check behaves like production.
app.state.transcription_service = _FakeReadyService()
client = TestClient(app)


class FakeService:
    def __init__(self, result=None, raise_exc=None):
        self._result = result
        self._raise_exc = raise_exc

    async def transcribe(self, audio_path, language, word_timestamps):
        if self._raise_exc:
            raise self._raise_exc
        return self._result


def _sample_result(with_words=False):
    words = [WordResult(word="hi", start=0.0, end=0.3)] if with_words else None
    segments = [SegmentResult(id=0, start=0.0, end=1.0, text=" hi", words=words)]
    return TranscriptionResult(text="hi", language="en", duration=1.0, segments=segments)


def test_health_check():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "queue_depth": 0}


def test_health_check_not_ready():
    had_attr = hasattr(app.state, "transcription_service")
    previous = getattr(app.state, "transcription_service", None)
    if had_attr:
        delattr(app.state, "transcription_service")
    try:
        resp = client.get("/health")
        assert resp.status_code == 503
        assert resp.json() == {"status": "loading"}
    finally:
        if had_attr:
            app.state.transcription_service = previous


def test_json_response_default():
    app.dependency_overrides[get_transcription_service] = lambda: FakeService(_sample_result())
    resp = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("a.wav", io.BytesIO(b"fake-audio-bytes"), "audio/wav")},
    )
    assert resp.status_code == 200
    assert resp.json() == {"text": "hi"}


def test_verbose_json_response_includes_segments():
    app.dependency_overrides[get_transcription_service] = lambda: FakeService(_sample_result())
    resp = client.post(
        "/v1/audio/transcriptions",
        data={"response_format": "verbose_json"},
        files={"file": ("a.wav", io.BytesIO(b"fake-audio-bytes"), "audio/wav")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["language"] == "en"
    assert body["segments"][0]["text"] == " hi"
    assert body["words"] is None


def test_verbose_json_with_word_timestamps():
    app.dependency_overrides[get_transcription_service] = lambda: FakeService(_sample_result(with_words=True))
    resp = client.post(
        "/v1/audio/transcriptions",
        data={"response_format": "verbose_json", "timestamp_granularities[]": "word"},
        files={"file": ("a.wav", io.BytesIO(b"fake-audio-bytes"), "audio/wav")},
    )
    body = resp.json()
    assert body["words"][0]["word"] == "hi"


def test_text_response_format():
    app.dependency_overrides[get_transcription_service] = lambda: FakeService(_sample_result())
    resp = client.post(
        "/v1/audio/transcriptions",
        data={"response_format": "text"},
        files={"file": ("a.wav", io.BytesIO(b"fake-audio-bytes"), "audio/wav")},
    )
    assert resp.status_code == 200
    assert resp.text == "hi"


def test_invalid_response_format_rejected():
    app.dependency_overrides[get_transcription_service] = lambda: FakeService(_sample_result())
    resp = client.post(
        "/v1/audio/transcriptions",
        data={"response_format": "srt"},
        files={"file": ("a.wav", io.BytesIO(b"fake-audio-bytes"), "audio/wav")},
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["type"] == "invalid_request_error"


def test_empty_file_rejected():
    app.dependency_overrides[get_transcription_service] = lambda: FakeService(_sample_result())
    resp = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("a.wav", io.BytesIO(b""), "audio/wav")},
    )
    assert resp.status_code == 400


def test_missing_file_field_rejected():
    app.dependency_overrides[get_transcription_service] = lambda: FakeService(_sample_result())
    resp = client.post("/v1/audio/transcriptions", files={})
    assert resp.status_code == 400
    assert resp.json()["error"]["type"] == "invalid_request_error"


def test_file_too_large_rejected(monkeypatch):
    small_limit_settings = Settings(
        model_id=main_module.settings.model_id,
        device=main_module.settings.device,
        compute_type=main_module.settings.compute_type,
        max_concurrent_transcriptions=main_module.settings.max_concurrent_transcriptions,
        max_file_size_mb=0,
        api_key=main_module.settings.api_key,
        host=main_module.settings.host,
        port=main_module.settings.port,
    )
    monkeypatch.setattr(main_module, "settings", small_limit_settings)
    app.dependency_overrides[get_transcription_service] = lambda: FakeService(_sample_result())
    resp = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("a.wav", io.BytesIO(b"12345"), "audio/wav")},
    )
    assert resp.status_code == 413


def test_decode_failure_returns_400():
    app.dependency_overrides[get_transcription_service] = lambda: FakeService(raise_exc=ValueError("bad audio"))
    resp = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("a.wav", io.BytesIO(b"garbage"), "audio/wav")},
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["type"] == "invalid_request_error"


def test_internal_error_returns_503():
    app.dependency_overrides[get_transcription_service] = lambda: FakeService(raise_exc=RuntimeError("oom"))
    resp = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("a.wav", io.BytesIO(b"garbage"), "audio/wav")},
    )
    assert resp.status_code == 503
    assert resp.json()["error"]["type"] == "server_error"


def test_missing_api_key_rejected_without_override():
    app.dependency_overrides.pop(verify_api_key, None)
    try:
        app.dependency_overrides[get_transcription_service] = lambda: FakeService(_sample_result())
        resp = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("a.wav", io.BytesIO(b"fake-audio-bytes"), "audio/wav")},
        )
        assert resp.status_code == 401
    finally:
        app.dependency_overrides[verify_api_key] = lambda: None


def test_create_job_returns_202_with_queued_status():
    app.dependency_overrides[get_transcription_service] = lambda: FakeService(_sample_result())
    resp = client.post(
        "/v1/audio/transcriptions/jobs",
        files={"file": ("a.wav", io.BytesIO(b"fake-audio-bytes"), "audio/wav")},
    )
    assert resp.status_code == 202
    body = resp.json()
    assert body["status"] == "queued"
    assert body["job_id"]


def test_get_job_returns_queued_for_freshly_created_job():
    job = job_store.create()

    resp = client.get(f"/v1/audio/transcriptions/jobs/{job.job_id}")

    assert resp.status_code == 200
    assert resp.json() == {"job_id": job.job_id, "status": "queued", "result": None, "error": None}


def test_get_job_returns_completed_result():
    job = job_store.create()
    job_store.mark_completed(job.job_id, {"text": "hello"})

    resp = client.get(f"/v1/audio/transcriptions/jobs/{job.job_id}")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "completed"
    assert body["result"] == {"text": "hello"}
    assert body["error"] is None


def test_get_job_returns_failed_error():
    job = job_store.create()
    job_store.mark_failed(job.job_id, {"message": "boom", "type": "server_error"})

    resp = client.get(f"/v1/audio/transcriptions/jobs/{job.job_id}")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "failed"
    assert body["error"] == {"message": "boom", "type": "server_error"}
    assert body["result"] is None


def test_get_unknown_job_returns_404():
    resp = client.get("/v1/audio/transcriptions/jobs/does-not-exist")
    assert resp.status_code == 404
    assert resp.json()["error"]["type"] == "invalid_request_error"


def test_submit_job_missing_file_rejected():
    app.dependency_overrides[get_transcription_service] = lambda: FakeService(_sample_result())
    resp = client.post("/v1/audio/transcriptions/jobs", files={})
    assert resp.status_code == 400


def test_submit_job_empty_file_rejected():
    app.dependency_overrides[get_transcription_service] = lambda: FakeService(_sample_result())
    resp = client.post(
        "/v1/audio/transcriptions/jobs",
        files={"file": ("a.wav", io.BytesIO(b""), "audio/wav")},
    )
    assert resp.status_code == 400


def test_submit_job_file_too_large_rejected(monkeypatch):
    small_limit_settings = Settings(
        model_id=main_module.settings.model_id,
        device=main_module.settings.device,
        compute_type=main_module.settings.compute_type,
        max_concurrent_transcriptions=main_module.settings.max_concurrent_transcriptions,
        max_file_size_mb=0,
        api_key=main_module.settings.api_key,
        host=main_module.settings.host,
        port=main_module.settings.port,
    )
    monkeypatch.setattr(main_module, "settings", small_limit_settings)
    app.dependency_overrides[get_transcription_service] = lambda: FakeService(_sample_result())
    resp = client.post(
        "/v1/audio/transcriptions/jobs",
        files={"file": ("a.wav", io.BytesIO(b"12345"), "audio/wav")},
    )
    assert resp.status_code == 413


def test_submit_then_poll_completes_via_background_task():
    app.dependency_overrides[get_transcription_service] = lambda: FakeService(_sample_result())
    submit_resp = client.post(
        "/v1/audio/transcriptions/jobs",
        files={"file": ("a.wav", io.BytesIO(b"fake-audio-bytes"), "audio/wav")},
    )
    assert submit_resp.status_code == 202
    job_id = submit_resp.json()["job_id"]

    body = None
    for _ in range(20):
        poll_resp = client.get(f"/v1/audio/transcriptions/jobs/{job_id}")
        body = poll_resp.json()
        if body["status"] != "queued":
            break
        time.sleep(0.05)

    assert body["status"] == "completed"
    assert body["result"]["text"] == "hi"
    assert body["error"] is None
