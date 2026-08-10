import io

from fastapi.testclient import TestClient

from app import main as main_module
from app.main import app, get_transcription_service, verify_api_key
from app.transcribe import SegmentResult, TranscriptionResult, WordResult

app.dependency_overrides[verify_api_key] = lambda: None
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
    assert resp.json() == {"status": "ok"}


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
    small_limit_settings = main_module.Settings(
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
