# Async Job API + queue_depth Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a submit-then-poll async job API (`POST /v1/audio/transcriptions/jobs`, `GET /v1/audio/transcriptions/jobs/{job_id}`) so no single HTTP request is ever held open long enough to hit the home-router's ~18s NAT idle timeout, and expose `queue_depth` on `GET /health` so clients can self-throttle.

**Architecture:** An in-memory `JobStore` (new `app/jobs.py`) tracks job records (`queued`/`completed`/`failed`). The submit endpoint validates and saves the upload synchronously (fast, bounded by size limit — no queue wait), then schedules the actual transcription as a background `asyncio.Task` and returns `202` immediately. The poll endpoint is a plain dict lookup. Both existing sync endpoint and the new job worker share the same `TranscriptionService` (unchanged concurrency mechanism), which gains a `queue_depth` counter incremented while a call waits for a semaphore slot.

**Tech Stack:** Same as the existing project — Python 3.12, FastAPI, uvicorn, faster-whisper, pytest + pytest-asyncio (`asyncio_mode = auto` already configured).

## Global Constraints

- Existing `POST /v1/audio/transcriptions` (sync) endpoint's behavior and tests are unchanged — only additive changes plus one internal refactor (extracting `_build_verbose_result`, behavior-preserving).
- New endpoints require the same `Authorization: Bearer <STT_API_KEY>` auth as the sync endpoint.
- Job status is exactly 3 values: `queued`, `completed`, `failed` — no `processing` sub-state.
- Job result (once `completed`) is always the full shape `{text, language, duration, segments, words}` — no `response_format` concept for jobs.
- `queue_depth` = count of calls (sync request or job worker) currently waiting for a `TranscriptionService` semaphore slot, not counting calls actively transcribing. Shared across sync + job paths since both use the same `TranscriptionService` instance.
- Job storage is in-memory only, TTL-purged via `JOB_RESULT_TTL_SECONDS` (default `3600`), purged opportunistically on every `JobStore.create()` call.
- All error responses keep the `{"error": {"message", "type"}}` shape; job-not-found is `404` with `type: "invalid_request_error"`.
- `GET /health` returns `{"status": "ok", "queue_depth": N}` when ready, `{"status": "loading"}` (unchanged, no `queue_depth` key) when the model hasn't finished loading.

Spec reference: `docs/superpowers/specs/2026-08-11-async-job-api-design.md`

---

### Task 1: In-memory JobStore

**Files:**
- Create: `app/jobs.py`
- Test: `tests/test_jobs.py`

**Interfaces:**
- Consumes: nothing (pure data structure, no dependency on other app modules).
- Produces: `app.jobs.Job` (dataclass: `job_id: str`, `status: str`, `created_at: float`, `result: Optional[dict] = None`, `error: Optional[dict] = None`), `app.jobs.JobStore` with `__init__(self, ttl_seconds: int, clock: Callable[[], float] = time.monotonic)`, `def create(self) -> Job`, `def get(self, job_id: str) -> Optional[Job]`, `def mark_completed(self, job_id: str, result: dict) -> None`, `def mark_failed(self, job_id: str, error: dict) -> None`, `def purge_expired(self) -> None`.

- [ ] **Step 1: Write the failing test** — `tests/test_jobs.py`

```python
from app.jobs import JobStore


def test_create_returns_queued_job_with_unique_id():
    store = JobStore(ttl_seconds=3600)
    job1 = store.create()
    job2 = store.create()

    assert job1.status == "queued"
    assert job1.result is None
    assert job1.error is None
    assert job1.job_id != job2.job_id


def test_get_returns_created_job():
    store = JobStore(ttl_seconds=3600)
    job = store.create()

    assert store.get(job.job_id) is job


def test_get_returns_none_for_unknown_id():
    store = JobStore(ttl_seconds=3600)

    assert store.get("does-not-exist") is None


def test_mark_completed_sets_status_and_result():
    store = JobStore(ttl_seconds=3600)
    job = store.create()

    store.mark_completed(job.job_id, {"text": "hello"})

    fetched = store.get(job.job_id)
    assert fetched.status == "completed"
    assert fetched.result == {"text": "hello"}
    assert fetched.error is None


def test_mark_failed_sets_status_and_error():
    store = JobStore(ttl_seconds=3600)
    job = store.create()

    store.mark_failed(job.job_id, {"message": "boom", "type": "server_error"})

    fetched = store.get(job.job_id)
    assert fetched.status == "failed"
    assert fetched.error == {"message": "boom", "type": "server_error"}
    assert fetched.result is None


def test_purge_expired_removes_old_jobs_using_injected_clock():
    fake_time = [0.0]
    store = JobStore(ttl_seconds=10, clock=lambda: fake_time[0])

    old_job = store.create()
    fake_time[0] = 20.0  # 20s later, past ttl_seconds=10
    new_job = store.create()  # create() purges expired jobs first

    assert store.get(old_job.job_id) is None
    assert store.get(new_job.job_id) is new_job
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_jobs.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.jobs'`

- [ ] **Step 3: Implement `app/jobs.py`**

```python
import time
import uuid
from dataclasses import dataclass
from typing import Callable, Dict, Optional


@dataclass
class Job:
    job_id: str
    status: str
    created_at: float
    result: Optional[dict] = None
    error: Optional[dict] = None


class JobStore:
    def __init__(self, ttl_seconds: int, clock: Callable[[], float] = time.monotonic):
        self._ttl_seconds = ttl_seconds
        self._clock = clock
        self._jobs: Dict[str, Job] = {}

    def create(self) -> Job:
        self.purge_expired()
        job = Job(job_id=uuid.uuid4().hex, status="queued", created_at=self._clock())
        self._jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def mark_completed(self, job_id: str, result: dict) -> None:
        job = self._jobs.get(job_id)
        if job is not None:
            job.status = "completed"
            job.result = result

    def mark_failed(self, job_id: str, error: dict) -> None:
        job = self._jobs.get(job_id)
        if job is not None:
            job.status = "failed"
            job.error = error

    def purge_expired(self) -> None:
        cutoff = self._clock() - self._ttl_seconds
        expired = [job_id for job_id, job in self._jobs.items() if job.created_at < cutoff]
        for job_id in expired:
            del self._jobs[job_id]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_jobs.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Run full suite**

Run: `pytest -v`
Expected: all previously-passing tests plus the 6 new ones pass (26 + 6 = 32).

- [ ] **Step 6: Commit**

```bash
git add app/jobs.py tests/test_jobs.py
git commit -m "feat: in-memory JobStore for async transcription jobs"
```

---

### Task 2: `queue_depth` on TranscriptionService

**Files:**
- Modify: `app/transcribe.py`
- Modify: `tests/test_transcribe.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `TranscriptionService.queue_depth` — read-only `int` property. No change to `transcribe()`'s public signature or behavior (still `async def transcribe(self, audio_path: str, language: Optional[str], word_timestamps: bool) -> TranscriptionResult`).

- [ ] **Step 1: Write the failing test** — append to `tests/test_transcribe.py`

```python
async def test_queue_depth_reflects_waiting_requests():
    import asyncio

    segments = [FakeSegment(0, 0.0, 1.0, " hi")]
    model = FakeWhisperModel(segments, delay=0.2)
    service = TranscriptionService(model, max_concurrent=2)

    assert service.queue_depth == 0

    tasks = [
        asyncio.create_task(service.transcribe("audio.wav", language=None, word_timestamps=False))
        for _ in range(5)
    ]
    await asyncio.sleep(0.05)  # 2 of 5 should have acquired a slot by now; 3 still waiting
    waiting_snapshot = service.queue_depth

    await asyncio.gather(*tasks)

    assert waiting_snapshot == 3
    assert service.queue_depth == 0
    service.shutdown()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_transcribe.py -v`
Expected: FAIL with `AttributeError: 'TranscriptionService' object has no attribute 'queue_depth'`

- [ ] **Step 3: Modify `app/transcribe.py`** — replace the `TranscriptionService.__init__` and `transcribe` methods with:

```python
    def __init__(self, model, max_concurrent: int):
        self._model = model
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._executor = ThreadPoolExecutor(max_workers=max_concurrent)
        self._waiting = 0

    @property
    def queue_depth(self) -> int:
        return self._waiting

    async def transcribe(
        self, audio_path: str, language: Optional[str], word_timestamps: bool
    ) -> TranscriptionResult:
        self._waiting += 1
        try:
            await self._semaphore.acquire()
        finally:
            self._waiting -= 1
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                self._executor, self._transcribe_sync, audio_path, language, word_timestamps
            )
        finally:
            self._semaphore.release()
```

(`_transcribe_sync` and `shutdown` are unchanged — leave them exactly as they are.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_transcribe.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Run full suite**

Run: `pytest -v`
Expected: all 33 tests pass (32 from Task 1 + this new one), zero regressions — especially confirm `test_transcribe_limits_concurrency_to_max_concurrent` (from the original plan) still passes, since the semaphore acquire/release mechanics changed shape (explicit acquire/release instead of `async with`) even though the observable behavior is identical.

- [ ] **Step 6: Commit**

```bash
git add app/transcribe.py tests/test_transcribe.py
git commit -m "feat: expose queue_depth on TranscriptionService"
```

---

### Task 3: Job endpoints + `queue_depth` on `/health`

**Files:**
- Modify: `app/config.py`
- Modify: `tests/test_config.py`
- Modify: `app/main.py`
- Modify: `tests/test_main.py`

**Interfaces:**
- Consumes: `app.jobs.JobStore` (Task 1), `TranscriptionService.queue_depth` (Task 2), existing `app.auth.make_verify_api_key`, `app.schemas.{Segment,Word,TranscriptionVerboseJSON,TranscriptionJSON}`.
- Produces: `app.main.job_store` (module-level `JobStore` instance, importable by tests), routes `POST /v1/audio/transcriptions/jobs` and `GET /v1/audio/transcriptions/jobs/{job_id}`, `GET /health` now includes `queue_depth`.

- [ ] **Step 1: Modify `app/config.py`** — add a new field with a default (so no existing direct `Settings(...)` construction in the test suite needs updating) to the `Settings` dataclass and `get_settings()`:

Change the `Settings` dataclass to:

```python
@dataclass(frozen=True)
class Settings:
    model_id: str
    device: str
    compute_type: str
    max_concurrent_transcriptions: int
    max_file_size_mb: int
    api_key: str
    host: str
    port: int
    job_result_ttl_seconds: int = 3600
```

Change `get_settings()` to add one line inside the `Settings(...)` call:

```python
def get_settings() -> Settings:
    return Settings(
        model_id=os.environ.get("STT_MODEL_ID", "deepdml/faster-whisper-large-v3-turbo-ct2"),
        device=os.environ.get("DEVICE", "cuda"),
        compute_type=os.environ.get("COMPUTE_TYPE", "float16"),
        max_concurrent_transcriptions=int(os.environ.get("MAX_CONCURRENT_TRANSCRIPTIONS", "4")),
        max_file_size_mb=int(os.environ.get("MAX_FILE_SIZE_MB", "200")),
        api_key=os.environ.get("STT_API_KEY", "changeme-please-set-STT_API_KEY"),
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
        job_result_ttl_seconds=int(os.environ.get("JOB_RESULT_TTL_SECONDS", "3600")),
    )
```

- [ ] **Step 2: Write the failing test** — append to `tests/test_config.py`, and extend the existing two tests:

In `test_defaults_when_no_env_set`, add `"JOB_RESULT_TTL_SECONDS"` to the tuple of env vars being deleted, and add this assertion at the end of the function:

```python
    assert settings.job_result_ttl_seconds == 3600
```

In `test_env_overrides_are_applied`, add this line among the other `monkeypatch.setenv(...)` calls:

```python
    monkeypatch.setenv("JOB_RESULT_TTL_SECONDS", "60")
```

and add this assertion among the others:

```python
    assert settings.job_result_ttl_seconds == 60
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL — `AttributeError: 'Settings' object has no attribute 'job_result_ttl_seconds'`

- [ ] **Step 4: Run test to verify it passes** (config.py was already changed in Step 1)

Run: `pytest tests/test_config.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Replace the full contents of `app/main.py`** with:

```python
import asyncio
import logging
import os
import shutil
import tempfile
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.requests import Request
from fastapi.responses import JSONResponse, PlainTextResponse
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.auth import make_verify_api_key
from app.config import get_settings
from app.jobs import JobStore
from app.schemas import Segment, TranscriptionJSON, TranscriptionVerboseJSON, Word
from app.transcribe import TranscriptionResult, TranscriptionService

logger = logging.getLogger("stt_server")

settings = get_settings()
verify_api_key = make_verify_api_key(settings)
job_store = JobStore(ttl_seconds=settings.job_result_ttl_seconds)
_background_tasks: set = set()


@asynccontextmanager
async def lifespan(app: FastAPI):
    from faster_whisper import WhisperModel

    logging.basicConfig(level=logging.INFO)

    if settings.api_key == "changeme-please-set-STT_API_KEY":
        logger.warning("STT_API_KEY not set — using insecure default dev key")

    model = WhisperModel(
        settings.model_id,
        device=settings.device,
        compute_type=settings.compute_type,
        num_workers=settings.max_concurrent_transcriptions,
    )
    app.state.transcription_service = TranscriptionService(model, settings.max_concurrent_transcriptions)
    yield
    app.state.transcription_service.shutdown()


app = FastAPI(lifespan=lifespan)


def get_transcription_service() -> TranscriptionService:
    return app.state.transcription_service


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    if isinstance(exc.detail, dict) and "error" in exc.detail:
        return JSONResponse(status_code=exc.status_code, content=exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"message": str(exc.detail), "type": "error"}},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning("Request validation failed: %s", exc.errors())
    return JSONResponse(
        status_code=400,
        content={"error": {"message": "Invalid request", "type": "invalid_request_error"}},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception while processing request")
    return JSONResponse(
        status_code=500,
        content={"error": {"message": "internal server error", "type": "server_error"}},
    )


@app.get("/health")
async def health():
    service = getattr(app.state, "transcription_service", None)
    if service is None:
        return JSONResponse(status_code=503, content={"status": "loading"})
    return {"status": "ok", "queue_depth": service.queue_depth}


def _build_verbose_result(result: TranscriptionResult, word_timestamps: bool) -> TranscriptionVerboseJSON:
    words = None
    if word_timestamps:
        words = [w for seg in result.segments for w in (seg.words or [])]
    return TranscriptionVerboseJSON(
        text=result.text,
        language=result.language,
        duration=result.duration,
        segments=[Segment(id=s.id, start=s.start, end=s.end, text=s.text) for s in result.segments],
        words=[Word(word=w.word, start=w.start, end=w.end) for w in words] if words else None,
    )


@app.post("/v1/audio/transcriptions", dependencies=[Depends(verify_api_key)])
async def create_transcription(
    file: UploadFile = File(...),
    model: str = Form(default=""),
    language: str = Form(default=""),
    response_format: str = Form(default="json"),
    timestamp_granularities: List[str] = Form(default=[], alias="timestamp_granularities[]"),
    service: TranscriptionService = Depends(get_transcription_service),
):
    if response_format not in ("json", "verbose_json", "text"):
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": f"Unsupported response_format: {response_format}",
                    "type": "invalid_request_error",
                }
            },
        )

    max_bytes = settings.max_file_size_mb * 1024 * 1024
    if file.size is not None and file.size > max_bytes:
        raise HTTPException(
            status_code=413,
            detail={
                "error": {
                    "message": f"File exceeds max size of {settings.max_file_size_mb}MB",
                    "type": "file_too_large",
                }
            },
        )

    suffix = os.path.splitext(file.filename or "")[1][:10] or ".audio"
    word_timestamps = response_format == "verbose_json" and "word" in timestamp_granularities

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        await run_in_threadpool(shutil.copyfileobj, file.file, tmp)
        tmp.flush()

        actual_size = os.path.getsize(tmp.name)
        if actual_size > max_bytes:
            raise HTTPException(
                status_code=413,
                detail={
                    "error": {
                        "message": f"File exceeds max size of {settings.max_file_size_mb}MB",
                        "type": "file_too_large",
                    }
                },
            )
        if actual_size == 0:
            raise HTTPException(
                status_code=400,
                detail={"error": {"message": "Uploaded file is empty", "type": "invalid_request_error"}},
            )

        try:
            result = await service.transcribe(
                tmp.name, language=language or None, word_timestamps=word_timestamps
            )
        except (OSError, ValueError, IndexError) as exc:
            logger.warning("Could not decode audio: %s", exc)
            raise HTTPException(
                status_code=400,
                detail={"error": {"message": "Could not decode audio file", "type": "invalid_request_error"}},
            )
        except Exception:
            logger.exception("Transcription failed")
            raise HTTPException(
                status_code=503,
                detail={"error": {"message": "server busy, try again", "type": "server_error"}},
            )

    if response_format == "text":
        return PlainTextResponse(result.text)

    if response_format == "verbose_json":
        return _build_verbose_result(result, word_timestamps)

    return TranscriptionJSON(text=result.text)


async def _run_transcription_job(
    job_id: str,
    tmp_path: str,
    language: Optional[str],
    word_timestamps: bool,
    service: TranscriptionService,
) -> None:
    try:
        result = await service.transcribe(tmp_path, language=language, word_timestamps=word_timestamps)
    except (OSError, ValueError, IndexError) as exc:
        logger.warning("Job %s: could not decode audio: %s", job_id, exc)
        job_store.mark_failed(
            job_id, {"message": "Could not decode audio file", "type": "invalid_request_error"}
        )
    except Exception:
        logger.exception("Job %s: transcription failed", job_id)
        job_store.mark_failed(job_id, {"message": "server busy, try again", "type": "server_error"})
    else:
        verbose = _build_verbose_result(result, word_timestamps)
        job_store.mark_completed(job_id, verbose.model_dump())
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@app.post("/v1/audio/transcriptions/jobs", status_code=202, dependencies=[Depends(verify_api_key)])
async def create_transcription_job(
    file: UploadFile = File(...),
    language: str = Form(default=""),
    timestamp_granularities: List[str] = Form(default=[], alias="timestamp_granularities[]"),
    service: TranscriptionService = Depends(get_transcription_service),
):
    max_bytes = settings.max_file_size_mb * 1024 * 1024
    if file.size is not None and file.size > max_bytes:
        raise HTTPException(
            status_code=413,
            detail={
                "error": {
                    "message": f"File exceeds max size of {settings.max_file_size_mb}MB",
                    "type": "file_too_large",
                }
            },
        )

    suffix = os.path.splitext(file.filename or "")[1][:10] or ".audio"
    word_timestamps = "word" in timestamp_granularities

    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        await run_in_threadpool(shutil.copyfileobj, file.file, tmp)
        tmp.flush()
        tmp.close()

        actual_size = os.path.getsize(tmp.name)
        if actual_size > max_bytes:
            raise HTTPException(
                status_code=413,
                detail={
                    "error": {
                        "message": f"File exceeds max size of {settings.max_file_size_mb}MB",
                        "type": "file_too_large",
                    }
                },
            )
        if actual_size == 0:
            raise HTTPException(
                status_code=400,
                detail={"error": {"message": "Uploaded file is empty", "type": "invalid_request_error"}},
            )
    except Exception:
        os.unlink(tmp.name)
        raise

    job = job_store.create()
    task = asyncio.create_task(
        _run_transcription_job(job.job_id, tmp.name, language or None, word_timestamps, service)
    )
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)

    return JSONResponse(status_code=202, content={"job_id": job.job_id, "status": job.status})


@app.get("/v1/audio/transcriptions/jobs/{job_id}", dependencies=[Depends(verify_api_key)])
async def get_transcription_job(job_id: str):
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail={"error": {"message": "Job not found", "type": "invalid_request_error"}},
        )
    return {"job_id": job.job_id, "status": job.status, "result": job.result, "error": job.error}
```

- [ ] **Step 6: Replace the full contents of `tests/test_main.py`** with:

```python
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
```

- [ ] **Step 7: Run the focused tests**

Run: `pytest tests/test_main.py -v`
Expected: PASS (21 passed — 12 existing + 9 new: create_job, get_job queued/completed/failed, unknown 404, missing/empty/too-large submit rejections, submit-then-poll).

- [ ] **Step 8: Run the full suite**

Run: `pytest -v`
Expected: all tests pass (33 from Tasks 1-2 minus the old `test_main.py` count, plus the new 21 in `test_main.py` — verify the total and that nothing regressed), output pristine, no warnings, no GPU/network access.

- [ ] **Step 9: Commit**

```bash
git add app/config.py tests/test_config.py app/main.py tests/test_main.py
git commit -m "feat: async job API (submit+poll) and queue_depth on /health"
```

---

### Task 4: Documentation

**Files:**
- Modify: `.env.example`
- Modify: `README.md`
- Modify: `API.md`

**Interfaces:**
- Consumes: nothing (docs only).
- Produces: nothing (docs only).

- [ ] **Step 1: Modify `.env.example`** — add this line after `MAX_FILE_SIZE_MB=200`:

```
JOB_RESULT_TTL_SECONDS=3600
```

- [ ] **Step 2: Modify `README.md`** — add a row to the Configuration table, right after the `MAX_FILE_SIZE_MB` row:

```
| `JOB_RESULT_TTL_SECONDS` | `3600` | How long a completed/failed async job's result stays available for polling before being purged from memory |
```

Add a short paragraph right after the existing `### Error responses` section, before `## Tests`:

```markdown
### Async job API

For batch/high-traffic use, `POST /v1/audio/transcriptions/jobs` and `GET /v1/audio/transcriptions/jobs/{job_id}` avoid holding a single HTTP connection open for the full queue-wait + transcription duration — every individual request completes in milliseconds. See `API.md` for the full contract with curl examples. Job results are kept in memory only (lost on restart) for `JOB_RESULT_TTL_SECONDS` after completion.
```

- [ ] **Step 3: Modify `API.md`** — update the `GET /health` response example to include `queue_depth`:

Replace:

```json
{ "status": "ok" }
```

With:

```json
{ "status": "ok", "queue_depth": 0 }
```

Add this paragraph right after that response example, before the "Response — server still starting up" section:

```markdown
`queue_depth` is the number of transcription requests (sync or async job) currently waiting for a free GPU slot — not counting ones actively transcribing. Use it to self-throttle batch submission instead of guessing.
```

Add a new `## POST /v1/audio/transcriptions/jobs` and `## GET /v1/audio/transcriptions/jobs/{job_id}` section, right after the existing `## POST /v1/audio/transcriptions` section's error-responses subsection and before the final `## Notes` section:

```markdown
## Async Job API

For batch or high-traffic use, prefer submit-then-poll over the synchronous endpoint above: every individual request (submit, poll) completes in milliseconds, so no connection is ever held open long enough to hit an intermediate NAT/proxy idle timeout.

### `POST /v1/audio/transcriptions/jobs`

Same auth as the sync endpoint. `multipart/form-data`.

| Field | Required | Description |
|---|---|---|
| `file` | Yes | Audio file. Same validation as the sync endpoint (`400` empty/missing, `413` oversized). |
| `language` | No | ISO code; omit for auto-detect. |
| `timestamp_granularities[]` | No | `word` — include word-level timestamps in the completed result. |

Returns immediately, before any GPU work happens.

#### curl

```bash
curl -s -X POST http://localhost:8000/v1/audio/transcriptions/jobs \
  -H "Authorization: Bearer $STT_API_KEY" \
  -F "file=@sample.wav"
```

#### Response

**`202 Accepted`**

```json
{ "job_id": "b3f1a2c4d5e6f7089a1b2c3d4e5f6789", "status": "queued" }
```

### `GET /v1/audio/transcriptions/jobs/{job_id}`

Same auth as the sync endpoint.

#### curl

```bash
curl -s http://localhost:8000/v1/audio/transcriptions/jobs/b3f1a2c4d5e6f7089a1b2c3d4e5f6789 \
  -H "Authorization: Bearer $STT_API_KEY"
```

#### Response — still queued

**`200 OK`**

```json
{ "job_id": "b3f1a2c4d5e6f7089a1b2c3d4e5f6789", "status": "queued", "result": null, "error": null }
```

#### Response — completed

**`200 OK`**

```json
{
  "job_id": "b3f1a2c4d5e6f7089a1b2c3d4e5f6789",
  "status": "completed",
  "result": {
    "text": "Xin chào, đây là bản ghi âm thử nghiệm.",
    "language": "vi",
    "duration": 4.2,
    "segments": [
      { "id": 0, "start": 0.0, "end": 2.1, "text": " Xin chào," },
      { "id": 1, "start": 2.1, "end": 4.2, "text": " đây là bản ghi âm thử nghiệm." }
    ],
    "words": null
  },
  "error": null
}
```

Job results always use this full shape (there is no `response_format` for jobs) — extract `.result.text` if you only need plain text.

#### Response — failed

**`200 OK`**

```json
{
  "job_id": "b3f1a2c4d5e6f7089a1b2c3d4e5f6789",
  "status": "failed",
  "result": null,
  "error": { "message": "Could not decode audio file", "type": "invalid_request_error" }
}
```

#### Response — unknown or expired job id

**`404 Not Found`**

```json
{ "error": { "message": "Job not found", "type": "invalid_request_error" } }
```

Job results are purged `JOB_RESULT_TTL_SECONDS` (default 1 hour) after creation — poll before then.
```

Also update the existing Error Responses table (in the sync-endpoint section) to add a note distinguishing the two different 404s — add this row right after the existing `404 | error | No route matches the request path` row:

```
| `404` | `invalid_request_error` | (Job API only) `job_id` not found or expired — distinct from the generic route-not-found 404 above, which uses `type: "error"` | unknown/expired job id on `GET /jobs/{id}` |
```

- [ ] **Step 4: Commit**

```bash
git add .env.example README.md API.md
git commit -m "docs: document async job API, queue_depth, and JOB_RESULT_TTL_SECONDS"
```

---

### Task 5: Real-model manual verification

**Files:** none (manual verification only, no code changes).

**Interfaces:** none.

- [ ] **Step 1: Confirm the server is running with the new code** — restart the running instance so it picks up the Task 1-3 changes:

```bash
cd /home/thanhdev/ai_dev/stt_server
pkill -f "uvicorn app.main:app.*--port 8266" || true
sleep 1
set -a; source .env; set +a
nohup .venv/bin/uvicorn app.main:app --host "$HOST" --port 8266 > /tmp/stt_server_8266.log 2>&1 &
sleep 4
curl -s http://localhost:8266/health
```

Expected: `{"status":"ok","queue_depth":0}` (note the new `queue_depth` key).

- [ ] **Step 2: Submit a real job and poll it to completion**

```bash
cd /home/thanhdev/ai_dev/stt_server
set -a; source .env; set +a
JOB_ID=$(curl -s -X POST http://localhost:8266/v1/audio/transcriptions/jobs \
  -H "Authorization: Bearer $STT_API_KEY" \
  -F "file=@sample.wav" | python3 -c "import sys,json; print(json.load(sys.stdin)['job_id'])")
echo "job_id: $JOB_ID"

for i in $(seq 1 20); do
  RESP=$(curl -s http://localhost:8266/v1/audio/transcriptions/jobs/$JOB_ID \
    -H "Authorization: Bearer $STT_API_KEY")
  STATUS=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['status'])")
  echo "poll $i: $STATUS"
  if [ "$STATUS" != "queued" ]; then
    echo "$RESP" | python3 -m json.tool
    break
  fi
  sleep 1
done
```

Expected: status transitions from `queued` to `completed` within a few seconds, final response has a non-null `result` with `text`/`language`/`duration`/`segments`.

- [ ] **Step 3: Verify submit latency stays low even when the queue is backed up** — this directly validates the fix for the original ~18s NAT-timeout production issue:

```bash
cd /home/thanhdev/ai_dev/stt_server
set -a; source .env; set +a
ffmpeg -y -f lavfi -i "sine=frequency=440:duration=300" -ar 16000 -ac 1 long_sample.wav 2>&1 | tail -2

for i in $(seq 1 12); do
  START=$(date +%s.%N)
  curl -s -o /dev/null -X POST http://localhost:8266/v1/audio/transcriptions/jobs \
    -H "Authorization: Bearer $STT_API_KEY" \
    -F "file=@long_sample.wav"
  END=$(date +%s.%N)
  echo "submit $i: $(echo "$END - $START" | bc)s"
done

curl -s http://localhost:8266/health
```

Expected: every one of the 12 submit calls completes in well under a second (they never wait on the transcription queue — only the earlier sync endpoint did that), even though `MAX_CONCURRENT_TRANSCRIPTIONS=8` means several of the 12 jobs are backed up behind GPU capacity. The final `/health` call should show `queue_depth > 0` while jobs are still draining (run it again a few times a couple seconds apart to see it fall back to `0`).

- [ ] **Step 4: Clean up generated files**

```bash
cd /home/thanhdev/ai_dev/stt_server
rm -f long_sample.wav
git status --short
```

Expected: clean (no untracked/modified files — `long_sample.wav` is gitignored and now removed).

- [ ] **Step 5: Record the result** — no commit needed for this task (no files changed), but note the outcome in the PR/summary: submit latency measurements and confirmation that `queue_depth` behaves as expected under load.

---

## Self-Review Notes

- **Spec coverage:** submit endpoint (Task 3), poll endpoint (Task 3), job statuses queued/completed/failed (Task 1 + 3), `queue_depth` on `/health` (Task 2 + 3), `JOB_RESULT_TTL_SECONDS` config (Task 3 config.py change + Task 4 docs), in-memory storage with TTL purge (Task 1), sync endpoint unchanged behavior (Task 3 — only DRY refactor via `_build_verbose_result`, verified behavior-preserving by keeping all existing sync-endpoint tests unmodified in Task 3 Step 6), real-model verification (Task 5) — all covered.
- **Placeholder scan:** no TBD/TODO markers; all code blocks are complete, runnable.
- **Type consistency:** `JobStore.create/get/mark_completed/mark_failed` signatures in Task 1 match every call site in Task 3's `app/main.py` (`job_store.create()`, `job_store.get(job_id)`, `job_store.mark_completed(job_id, dict)`, `job_store.mark_failed(job_id, dict)`). `TranscriptionService.queue_depth` (Task 2) matches its usage in Task 3's `/health` handler (`service.queue_depth`). `_build_verbose_result(result: TranscriptionResult, word_timestamps: bool) -> TranscriptionVerboseJSON` is defined once in Task 3 and used identically by both the sync endpoint and `_run_transcription_job`.
