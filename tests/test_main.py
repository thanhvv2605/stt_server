import asyncio
import glob
import io
import os
import tempfile
import time

import httpx
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


class SlowFakeService:
    """Transcription fake with a genuine await point, so tests can observe
    that a job is still in flight rather than finishing synchronously."""

    def __init__(self, result=None, delay=0.2):
        self._result = result
        self._delay = delay

    async def transcribe(self, audio_path, language, word_timestamps):
        await asyncio.sleep(self._delay)
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


def test_submit_job_rejected_when_pending_job_cap_reached(monkeypatch):
    low_cap_settings = Settings(
        model_id=main_module.settings.model_id,
        device=main_module.settings.device,
        compute_type=main_module.settings.compute_type,
        max_concurrent_transcriptions=main_module.settings.max_concurrent_transcriptions,
        max_file_size_mb=main_module.settings.max_file_size_mb,
        api_key=main_module.settings.api_key,
        host=main_module.settings.host,
        port=main_module.settings.port,
        max_pending_jobs=1,
    )
    monkeypatch.setattr(main_module, "settings", low_cap_settings)
    app.dependency_overrides[get_transcription_service] = lambda: FakeService(_sample_result())

    # Directly simulate 1 in-flight job (== the cap) without needing a real
    # background task, by inserting a placeholder into the same set the
    # handler checks the length of.
    placeholder = object()
    main_module._background_tasks.add(placeholder)
    try:
        resp = client.post(
            "/v1/audio/transcriptions/jobs",
            files={"file": ("a.wav", io.BytesIO(b"fake-audio-bytes"), "audio/wav")},
        )
        assert resp.status_code == 503
        assert resp.json()["error"] == {
            "message": "Too many pending jobs, try again later",
            "type": "server_error",
        }
    finally:
        main_module._background_tasks.discard(placeholder)


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


async def test_submit_does_not_block_on_transcription():
    # Uses httpx.AsyncClient + ASGITransport (not TestClient) so the whole
    # test runs on a single persistent event loop. TestClient opens a fresh
    # event-loop portal per call, which lets a synchronous fake finish
    # "instantly" and hides whether the handler actually awaited the
    # transcription before responding. A real await point (SlowFakeService)
    # plus a single loop is what lets this test actually distinguish
    # fire-and-forget submission from an inline-awaited (broken) one.
    app.dependency_overrides[get_transcription_service] = lambda: SlowFakeService(
        _sample_result(), delay=0.2
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client_:
        start = time.monotonic()
        submit_resp = await client_.post(
            "/v1/audio/transcriptions/jobs",
            files={"file": ("a.wav", io.BytesIO(b"fake-audio-bytes"), "audio/wav")},
        )
        elapsed = time.monotonic() - start

        assert submit_resp.status_code == 202
        assert elapsed < 0.15, f"submit took {elapsed:.3f}s — looks like it awaited transcription"
        job_id = submit_resp.json()["job_id"]

        # Poll immediately: the fake transcription takes 0.2s, so the job
        # must still be queued right after the (fast) submit response.
        poll_resp = await client_.get(f"/v1/audio/transcriptions/jobs/{job_id}")
        assert poll_resp.json()["status"] == "queued"

        body = None
        for _ in range(20):
            poll_resp = await client_.get(f"/v1/audio/transcriptions/jobs/{job_id}")
            body = poll_resp.json()
            if body["status"] != "queued":
                break
            await asyncio.sleep(0.05)

        assert body["status"] == "completed"
        assert body["result"]["text"] == "hi"
        assert body["error"] is None


async def _submit_and_wait_for_terminal_status(client_, service):
    app.dependency_overrides[get_transcription_service] = lambda: service
    submit_resp = await client_.post(
        "/v1/audio/transcriptions/jobs",
        files={"file": ("a.wav", io.BytesIO(b"fake-audio-bytes"), "audio/wav")},
    )
    assert submit_resp.status_code == 202
    job_id = submit_resp.json()["job_id"]

    body = None
    for _ in range(40):
        poll_resp = await client_.get(f"/v1/audio/transcriptions/jobs/{job_id}")
        body = poll_resp.json()
        if body["status"] != "queued":
            break
        await asyncio.sleep(0.05)

    return body


async def test_job_worker_maps_decode_failure_to_failed_status():
    # Exercises _run_transcription_job's own (OSError, ValueError, IndexError)
    # branch via the real background-task path, not JobStore directly.
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client_:
        body = await _submit_and_wait_for_terminal_status(
            client_, FakeService(raise_exc=ValueError("bad audio"))
        )

    assert body["status"] == "failed"
    assert body["error"] == {"message": "Could not decode audio file", "type": "invalid_request_error"}
    assert body["result"] is None


async def test_job_worker_maps_generic_failure_to_failed_status():
    # Exercises _run_transcription_job's own generic Exception branch via the
    # real background-task path, not JobStore directly.
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client_:
        body = await _submit_and_wait_for_terminal_status(
            client_, FakeService(raise_exc=RuntimeError("boom"))
        )

    assert body["status"] == "failed"
    assert body["error"] == {"message": "server busy, try again", "type": "server_error"}
    assert body["result"] is None


async def test_job_worker_result_build_failure_still_reaches_terminal_status():
    # Regression test for the bug where an exception raised while building
    # the result (after service.transcribe() already succeeded) was not
    # caught anywhere, leaving the job stuck on "queued" forever. A
    # TranscriptionResult with language=None makes TranscriptionVerboseJSON
    # validation fail inside _build_verbose_result/model_dump.
    bad_result = TranscriptionResult(text="hi", language=None, duration=1.0, segments=[])
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client_:
        body = await _submit_and_wait_for_terminal_status(client_, FakeService(bad_result))

    assert body is not None
    assert body["status"] == "failed", f"job never reached a terminal status: {body}"
    assert body["error"] == {"message": "server busy, try again", "type": "server_error"}
    assert body["result"] is None


async def test_job_temp_file_deleted_after_completion():
    # The job worker writes the upload to a NamedTemporaryFile(delete=False)
    # and must unlink it once the job reaches a terminal state. Use a
    # distinctive suffix so this test's temp file can be told apart from
    # anything else that happens to be in the system temp dir.
    marker_glob = os.path.join(tempfile.gettempdir(), "*.zigguratmarker")
    assert glob.glob(marker_glob) == []

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client_:
        app.dependency_overrides[get_transcription_service] = lambda: FakeService(_sample_result())
        submit_resp = await client_.post(
            "/v1/audio/transcriptions/jobs",
            files={"file": ("a.zigguratmarker", io.BytesIO(b"fake-audio-bytes"), "audio/wav")},
        )
        assert submit_resp.status_code == 202
        job_id = submit_resp.json()["job_id"]

        body = None
        for _ in range(40):
            poll_resp = await client_.get(f"/v1/audio/transcriptions/jobs/{job_id}")
            body = poll_resp.json()
            if body["status"] != "queued":
                break
            await asyncio.sleep(0.05)

    assert body["status"] == "completed"
    assert glob.glob(marker_glob) == []
