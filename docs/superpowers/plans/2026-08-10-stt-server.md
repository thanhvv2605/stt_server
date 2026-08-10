# STT Server Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a FastAPI server that transcribes audio via `whisper-large-v3-turbo` (CTranslate2/faster-whisper backend), serving multiple concurrent requests while keeping GPU VRAM usage under a 6GB budget.

**Architecture:** A single faster-whisper `WhisperModel` instance is loaded once at startup (float16, CUDA). Each transcription request acquires an `asyncio.Semaphore` slot, then runs the blocking `model.transcribe()` call in a bounded `ThreadPoolExecutor` — CTranslate2 releases the GIL during inference, so up to `MAX_CONCURRENT_TRANSCRIPTIONS` requests genuinely run in parallel on the GPU while the event loop stays free to accept new connections. The HTTP contract mirrors OpenAI's `/v1/audio/transcriptions` so existing OpenAI SDK clients work by just changing `base_url`.

**Tech Stack:** Python 3.12, FastAPI, uvicorn, faster-whisper (CTranslate2), pytest + pytest-asyncio, ffmpeg (OS dependency).

## Global Constraints

- Model: `openai/whisper-large-v3-turbo`, served via CTranslate2 checkpoint `deepdml/faster-whisper-large-v3-turbo-ct2`.
- Default `compute_type=float16`; override via `COMPUTE_TYPE` env var (e.g. `int8_float16`) to trade quality for VRAM.
- Total GPU VRAM used by this service must stay under 6GB — enforced only via `MAX_CONCURRENT_TRANSCRIPTIONS` (default `4`) + `COMPUTE_TYPE` tuning and manual benchmarking (no hard cap exists in CTranslate2).
- API surface: `POST /v1/audio/transcriptions` (multipart: `file`, `model`, `language`, `response_format` ∈ {`json`,`verbose_json`,`text`}, `timestamp_granularities[]` ∈ {`segment`,`word`}) and `GET /health`.
- All error responses use the shape `{"error": {"message": "...", "type": "..."}}`.
- Auth: static API key via `STT_API_KEY` env var, checked against header `Authorization: Bearer <key>`. Default dev key is `changeme-please-set-STT_API_KEY`; server logs a warning at startup if this default is still in use.
- Deployment: plain Python venv (no Docker). Host must have `ffmpeg` installed.
- Explicitly out of scope: streaming/WebSocket transcription, multi-GPU, external job queue (Redis/Celery) — an in-process semaphore + threadpool is the only concurrency mechanism.
- Git: repo is initialized as part of Task 1; commit after every task.

Spec reference: `docs/superpowers/specs/2026-08-10-stt-server-design.md`

---

### Task 1: Project scaffolding & Config

**Files:**
- Create: `requirements.txt`
- Create: `.env.example`
- Create: `pytest.ini`
- Create: `.gitignore`
- Create: `app/__init__.py`
- Create: `app/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `app.config.Settings` (dataclass with fields `model_id: str`, `device: str`, `compute_type: str`, `max_concurrent_transcriptions: int`, `max_file_size_mb: int`, `api_key: str`, `host: str`, `port: int`) and `app.config.get_settings() -> Settings` (reads env vars fresh on every call).

- [ ] **Step 1: Init git repo and Python venv**

```bash
cd /home/thanhdev/ai_dev/stt_server
git init
python3 -m venv .venv
source .venv/bin/activate
```

- [ ] **Step 2: Create `.gitignore`**

```
.venv/
__pycache__/
*.pyc
.env
.pytest_cache/
```

- [ ] **Step 3: Create `requirements.txt`**

```
fastapi>=0.115,<0.116
uvicorn[standard]>=0.32,<0.33
python-multipart>=0.0.9,<0.1
faster-whisper>=1.0.3,<2.0
pytest>=8.0,<9.0
pytest-asyncio>=0.24,<0.25
httpx>=0.27,<0.28
```

- [ ] **Step 4: Install dependencies**

```bash
pip install -r requirements.txt
```

Expected: all packages install without error (faster-whisper pulls in `ctranslate2` and `av` as transitive deps).

- [ ] **Step 5: Create `.env.example`**

```
STT_MODEL_ID=deepdml/faster-whisper-large-v3-turbo-ct2
DEVICE=cuda
COMPUTE_TYPE=float16
MAX_CONCURRENT_TRANSCRIPTIONS=4
MAX_FILE_SIZE_MB=200
STT_API_KEY=changeme-please-set-STT_API_KEY
HOST=0.0.0.0
PORT=8000
```

- [ ] **Step 6: Create `pytest.ini`**

```ini
[pytest]
asyncio_mode = auto
pythonpath = .
```

- [ ] **Step 7: Create `app/__init__.py`** (empty file)

- [ ] **Step 8: Write the failing test** — `tests/test_config.py`

```python
from app.config import get_settings


def test_defaults_when_no_env_set(monkeypatch):
    for var in (
        "STT_MODEL_ID", "DEVICE", "COMPUTE_TYPE", "MAX_CONCURRENT_TRANSCRIPTIONS",
        "MAX_FILE_SIZE_MB", "STT_API_KEY", "HOST", "PORT",
    ):
        monkeypatch.delenv(var, raising=False)

    settings = get_settings()

    assert settings.model_id == "deepdml/faster-whisper-large-v3-turbo-ct2"
    assert settings.device == "cuda"
    assert settings.compute_type == "float16"
    assert settings.max_concurrent_transcriptions == 4
    assert settings.max_file_size_mb == 200
    assert settings.api_key == "changeme-please-set-STT_API_KEY"
    assert settings.host == "0.0.0.0"
    assert settings.port == 8000


def test_env_overrides_are_applied(monkeypatch):
    monkeypatch.setenv("STT_MODEL_ID", "custom/model")
    monkeypatch.setenv("DEVICE", "cpu")
    monkeypatch.setenv("COMPUTE_TYPE", "int8")
    monkeypatch.setenv("MAX_CONCURRENT_TRANSCRIPTIONS", "2")
    monkeypatch.setenv("MAX_FILE_SIZE_MB", "50")
    monkeypatch.setenv("STT_API_KEY", "secret")
    monkeypatch.setenv("PORT", "9000")

    settings = get_settings()

    assert settings.model_id == "custom/model"
    assert settings.device == "cpu"
    assert settings.compute_type == "int8"
    assert settings.max_concurrent_transcriptions == 2
    assert settings.max_file_size_mb == 50
    assert settings.api_key == "secret"
    assert settings.port == 9000
```

- [ ] **Step 9: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.config'`

- [ ] **Step 10: Implement `app/config.py`**

```python
import os
from dataclasses import dataclass


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
    )
```

- [ ] **Step 11: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: PASS (2 passed)

- [ ] **Step 12: Commit**

```bash
git add requirements.txt .env.example pytest.ini .gitignore app/__init__.py app/config.py tests/test_config.py
git commit -m "feat: project scaffolding and env-based config"
```

---

### Task 2: Response schemas

**Files:**
- Create: `app/schemas.py`
- Test: `tests/test_schemas.py`

**Interfaces:**
- Consumes: nothing (pure pydantic models, no dependency on other app modules).
- Produces: `app.schemas.Segment` (fields `id: int`, `start: float`, `end: float`, `text: str`), `app.schemas.Word` (fields `word: str`, `start: float`, `end: float`), `app.schemas.TranscriptionJSON` (field `text: str`), `app.schemas.TranscriptionVerboseJSON` (fields `text: str`, `language: str`, `duration: float`, `segments: list[Segment]`, `words: list[Word] | None = None`).

- [ ] **Step 1: Write the failing test** — `tests/test_schemas.py`

```python
from app.schemas import Segment, TranscriptionJSON, TranscriptionVerboseJSON, Word


def test_transcription_json_serializes_text_only():
    payload = TranscriptionJSON(text="hello world")
    assert payload.model_dump() == {"text": "hello world"}


def test_transcription_verbose_json_serializes_all_fields():
    payload = TranscriptionVerboseJSON(
        text="hi",
        language="en",
        duration=1.5,
        segments=[Segment(id=0, start=0.0, end=1.5, text="hi")],
        words=[Word(word="hi", start=0.0, end=0.3)],
    )
    dumped = payload.model_dump()
    assert dumped["language"] == "en"
    assert dumped["segments"][0]["text"] == "hi"
    assert dumped["words"][0]["word"] == "hi"


def test_transcription_verbose_json_words_default_none():
    payload = TranscriptionVerboseJSON(
        text="hi", language="en", duration=1.0,
        segments=[Segment(id=0, start=0.0, end=1.0, text="hi")],
    )
    assert payload.words is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_schemas.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.schemas'`

- [ ] **Step 3: Implement `app/schemas.py`**

```python
from typing import List, Optional

from pydantic import BaseModel


class Segment(BaseModel):
    id: int
    start: float
    end: float
    text: str


class Word(BaseModel):
    word: str
    start: float
    end: float


class TranscriptionJSON(BaseModel):
    text: str


class TranscriptionVerboseJSON(BaseModel):
    text: str
    language: str
    duration: float
    segments: List[Segment]
    words: Optional[List[Word]] = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_schemas.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add app/schemas.py tests/test_schemas.py
git commit -m "feat: response schemas for transcription endpoints"
```

---

### Task 3: Transcription service (concurrency control)

**Files:**
- Create: `app/transcribe.py`
- Test: `tests/test_transcribe.py`

**Interfaces:**
- Consumes: nothing from earlier tasks — takes any object exposing `.transcribe(audio, language=None, word_timestamps=False) -> (Iterable[segment], info)` where `segment` has `.id/.start/.end/.text/.words` and `info` has `.language/.duration` (this is the faster-whisper `WhisperModel` interface; tests use a fake).
- Produces: `app.transcribe.WordResult` (dataclass: `word: str`, `start: float`, `end: float`), `app.transcribe.SegmentResult` (dataclass: `id: int`, `start: float`, `end: float`, `text: str`, `words: list[WordResult] | None = None`), `app.transcribe.TranscriptionResult` (dataclass: `text: str`, `language: str`, `duration: float`, `segments: list[SegmentResult]`), `app.transcribe.TranscriptionService` with `__init__(self, model, max_concurrent: int)`, `async def transcribe(self, audio_path: str, language: str | None, word_timestamps: bool) -> TranscriptionResult`, `def shutdown(self) -> None`.

- [ ] **Step 1: Write the failing test** — `tests/test_transcribe.py`

```python
import threading
import time

from app.transcribe import TranscriptionService


class FakeWord:
    def __init__(self, word, start, end):
        self.word = word
        self.start = start
        self.end = end


class FakeSegment:
    def __init__(self, id, start, end, text, words=None):
        self.id = id
        self.start = start
        self.end = end
        self.text = text
        self.words = words


class FakeInfo:
    def __init__(self, language, duration):
        self.language = language
        self.duration = duration


class FakeWhisperModel:
    def __init__(self, segments, language="en", duration=1.0, delay=0.0):
        self._segments = segments
        self._language = language
        self._duration = duration
        self._delay = delay
        self._lock = threading.Lock()
        self.in_flight = 0
        self.max_in_flight = 0
        self.calls = []

    def transcribe(self, audio, language=None, word_timestamps=False):
        with self._lock:
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
        self.calls.append((audio, language, word_timestamps))
        if self._delay:
            time.sleep(self._delay)
        with self._lock:
            self.in_flight -= 1
        return iter(self._segments), FakeInfo(self._language, self._duration)


async def test_transcribe_returns_joined_text_and_segments():
    segments = [FakeSegment(0, 0.0, 1.0, " hello"), FakeSegment(1, 1.0, 2.0, " world")]
    model = FakeWhisperModel(segments, language="en", duration=2.0)
    service = TranscriptionService(model, max_concurrent=2)

    result = await service.transcribe("audio.wav", language=None, word_timestamps=False)

    assert result.text == "hello world"
    assert result.language == "en"
    assert result.duration == 2.0
    assert [s.text for s in result.segments] == [" hello", " world"]
    service.shutdown()


async def test_transcribe_includes_word_timestamps_when_requested():
    words = [FakeWord("hi", 0.0, 0.3)]
    segments = [FakeSegment(0, 0.0, 1.0, " hi", words=words)]
    model = FakeWhisperModel(segments)
    service = TranscriptionService(model, max_concurrent=2)

    result = await service.transcribe("audio.wav", language=None, word_timestamps=True)

    assert result.segments[0].words[0].word == "hi"
    service.shutdown()


async def test_transcribe_passes_language_through():
    segments = [FakeSegment(0, 0.0, 1.0, " hi")]
    model = FakeWhisperModel(segments)
    service = TranscriptionService(model, max_concurrent=2)

    await service.transcribe("audio.wav", language="vi", word_timestamps=False)

    assert model.calls[0] == ("audio.wav", "vi", False)
    service.shutdown()


async def test_transcribe_limits_concurrency_to_max_concurrent():
    import asyncio

    segments = [FakeSegment(0, 0.0, 1.0, " hi")]
    model = FakeWhisperModel(segments, delay=0.2)
    service = TranscriptionService(model, max_concurrent=2)

    await asyncio.gather(*(
        service.transcribe("audio.wav", language=None, word_timestamps=False)
        for _ in range(5)
    ))

    assert model.max_in_flight == 2
    service.shutdown()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_transcribe.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.transcribe'`

- [ ] **Step 3: Implement `app/transcribe.py`**

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class WordResult:
    word: str
    start: float
    end: float


@dataclass
class SegmentResult:
    id: int
    start: float
    end: float
    text: str
    words: Optional[List[WordResult]] = None


@dataclass
class TranscriptionResult:
    text: str
    language: str
    duration: float
    segments: List[SegmentResult]


class TranscriptionService:
    def __init__(self, model, max_concurrent: int):
        self._model = model
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._executor = ThreadPoolExecutor(max_workers=max_concurrent)

    async def transcribe(
        self, audio_path: str, language: Optional[str], word_timestamps: bool
    ) -> TranscriptionResult:
        async with self._semaphore:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                self._executor, self._transcribe_sync, audio_path, language, word_timestamps
            )

    def _transcribe_sync(
        self, audio_path: str, language: Optional[str], word_timestamps: bool
    ) -> TranscriptionResult:
        segments_iter, info = self._model.transcribe(
            audio_path, language=language, word_timestamps=word_timestamps
        )
        segments: List[SegmentResult] = []
        text_parts: List[str] = []
        for i, seg in enumerate(segments_iter):
            words = None
            if word_timestamps and getattr(seg, "words", None):
                words = [WordResult(word=w.word, start=w.start, end=w.end) for w in seg.words]
            segments.append(SegmentResult(id=i, start=seg.start, end=seg.end, text=seg.text, words=words))
            text_parts.append(seg.text)
        return TranscriptionResult(
            text="".join(text_parts).strip(),
            language=info.language,
            duration=info.duration,
            segments=segments,
        )

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_transcribe.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add app/transcribe.py tests/test_transcribe.py
git commit -m "feat: transcription service with bounded concurrency"
```

---

### Task 4: API key auth dependency

**Files:**
- Create: `app/auth.py`
- Test: `tests/test_auth.py`

**Interfaces:**
- Consumes: `app.config.Settings` (from Task 1) — reads `settings.api_key`.
- Produces: `app.auth.make_verify_api_key(settings: Settings) -> Callable` — an async FastAPI dependency that raises `fastapi.HTTPException(401, detail={"error": {"message": ..., "type": "invalid_api_key"}})` on mismatch, returns `None` on success.

- [ ] **Step 1: Write the failing test** — `tests/test_auth.py`

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_auth.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.auth'`

- [ ] **Step 3: Implement `app/auth.py`**

```python
from fastapi import Header, HTTPException

from app.config import Settings


def make_verify_api_key(settings: Settings):
    expected = f"Bearer {settings.api_key}"

    async def verify_api_key(authorization: str = Header(default="")) -> None:
        if authorization != expected:
            raise HTTPException(
                status_code=401,
                detail={"error": {"message": "Invalid or missing API key", "type": "invalid_api_key"}},
            )

    return verify_api_key
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_auth.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add app/auth.py tests/test_auth.py
git commit -m "feat: static API key auth dependency"
```

---

### Task 5: FastAPI app and endpoints

**Files:**
- Create: `app/main.py`
- Test: `tests/test_main.py`

**Interfaces:**
- Consumes: `app.config.get_settings` (Task 1), `app.schemas.{Segment,Word,TranscriptionJSON,TranscriptionVerboseJSON}` (Task 2), `app.transcribe.TranscriptionService` (Task 3), `app.auth.make_verify_api_key` (Task 4).
- Produces: `app.main.app` (the FastAPI instance), `app.main.settings`, `app.main.verify_api_key`, `app.main.get_transcription_service` — all used directly by tests via `app.dependency_overrides`.

- [ ] **Step 1: Write the failing test** — `tests/test_main.py`

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_main.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.main'`

- [ ] **Step 3: Implement `app/main.py`**

```python
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from typing import List

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.requests import Request
from fastapi.responses import JSONResponse, PlainTextResponse

from app.auth import make_verify_api_key
from app.config import Settings, get_settings
from app.schemas import Segment, TranscriptionJSON, TranscriptionVerboseJSON, Word
from app.transcribe import TranscriptionService

logger = logging.getLogger("stt_server")

settings = get_settings()
verify_api_key = make_verify_api_key(settings)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from faster_whisper import WhisperModel

    if settings.api_key == "changeme-please-set-STT_API_KEY":
        logger.warning("STT_API_KEY not set — using insecure default dev key")

    model = WhisperModel(settings.model_id, device=settings.device, compute_type=settings.compute_type)
    app.state.transcription_service = TranscriptionService(model, settings.max_concurrent_transcriptions)
    yield
    app.state.transcription_service.shutdown()


app = FastAPI(lifespan=lifespan)


def get_transcription_service() -> TranscriptionService:
    return app.state.transcription_service


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    if isinstance(exc.detail, dict) and "error" in exc.detail:
        return JSONResponse(status_code=exc.status_code, content=exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"message": str(exc.detail), "type": "error"}},
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


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

    contents = await file.read()
    max_bytes = settings.max_file_size_mb * 1024 * 1024
    if len(contents) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail={
                "error": {
                    "message": f"File exceeds max size of {settings.max_file_size_mb}MB",
                    "type": "file_too_large",
                }
            },
        )
    if len(contents) == 0:
        raise HTTPException(
            status_code=400,
            detail={"error": {"message": "Uploaded file is empty", "type": "invalid_request_error"}},
        )

    suffix = os.path.splitext(file.filename or "")[1] or ".audio"
    word_timestamps = "word" in timestamp_granularities

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        tmp.write(contents)
        tmp.flush()
        try:
            result = await service.transcribe(
                tmp.name, language=language or None, word_timestamps=word_timestamps
            )
        except (OSError, ValueError) as exc:
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

    return TranscriptionJSON(text=result.text)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_main.py -v`
Expected: PASS (10 passed). Note: this does not start the model lifespan (tests never enter `TestClient` as a context manager and always override `get_transcription_service`), so no GPU/model download happens during this run.

- [ ] **Step 5: Run full test suite**

Run: `pytest -v`
Expected: all tests across `tests/test_config.py`, `tests/test_schemas.py`, `tests/test_transcribe.py`, `tests/test_auth.py`, `tests/test_main.py` pass.

- [ ] **Step 6: Commit**

```bash
git add app/main.py tests/test_main.py
git commit -m "feat: FastAPI app wiring endpoints, auth, and error handling"
```

---

### Task 6: README, real-model smoke test, and concurrency/VRAM benchmark

**Files:**
- Create: `README.md`
- Create: `scripts/benchmark_concurrency.sh`

**Interfaces:**
- Consumes: the running server started from Task 5's `app.main:app`.
- Produces: none (documentation + manual verification scripts).

- [ ] **Step 1: Create `README.md`**

```markdown
# STT Server

Simple FastAPI speech-to-text server using `whisper-large-v3-turbo` via faster-whisper (CTranslate2), OpenAI-compatible `/v1/audio/transcriptions` endpoint.

## Prerequisites

- Python 3.12+
- NVIDIA GPU + driver with CUDA support
- `ffmpeg` installed on the host: `sudo apt install ffmpeg`

## Setup

\`\`\`bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env and set a real STT_API_KEY
\`\`\`

## Run

\`\`\`bash
set -a; source .env; set +a
uvicorn app.main:app --host "$HOST" --port "$PORT"
\`\`\`

First startup downloads the model from the Hugging Face Hub (requires internet access) and can take a few minutes.

## Usage

\`\`\`bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer $STT_API_KEY" \
  -F "file=@sample.wav" \
  -F "response_format=verbose_json" \
  -F "timestamp_granularities[]=word"
\`\`\`

## Tests

\`\`\`bash
pytest -v
\`\`\`

All automated tests use fake models and run on CPU without touching the GPU. Real-model verification is manual — see below.
```

- [ ] **Step 2: Generate a synthetic test audio file**

```bash
ffmpeg -f lavfi -i "sine=frequency=440:duration=5" -ar 16000 -ac 1 sample.wav
```

Expected: `sample.wav` created (5 seconds, 16kHz mono tone). This verifies the pipeline runs end-to-end without crashing; for meaningful transcription accuracy, replace with a real speech recording.

- [ ] **Step 3: Manual smoke test — start the server**

```bash
set -a; source .env; set +a
uvicorn app.main:app --host "$HOST" --port "$PORT"
```

Expected: log line confirming Uvicorn is running on `http://0.0.0.0:8000`, and (if `STT_API_KEY` was left at the default) a warning about the insecure default key.

- [ ] **Step 4: Manual smoke test — single request**

In a second terminal:

```bash
curl -s -X POST http://localhost:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer $STT_API_KEY" \
  -F "file=@sample.wav" \
  -F "response_format=verbose_json" | python3 -m json.tool
```

Expected: HTTP 200, JSON with non-empty `text`, `language`, `duration`, and a `segments` array with at least one entry.

- [ ] **Step 5: Create `scripts/benchmark_concurrency.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail

AUDIO_FILE="${1:-sample.wav}"
N="${2:-4}"
BASE_URL="${BASE_URL:-http://localhost:8000}"

for i in $(seq 1 "$N"); do
  curl -s -X POST "$BASE_URL/v1/audio/transcriptions" \
    -H "Authorization: Bearer $STT_API_KEY" \
    -F "file=@${AUDIO_FILE}" \
    -F "response_format=verbose_json" \
    -o "result_${i}.json" &
done
wait

echo "Requests completed. Checking for non-empty results:"
for i in $(seq 1 "$N"); do
  size=$(wc -c < "result_${i}.json")
  echo "result_${i}.json: ${size} bytes"
done
```

```bash
chmod +x scripts/benchmark_concurrency.sh
```

- [ ] **Step 6: Manual test — 4 concurrent requests within VRAM budget**

In one terminal, monitor VRAM (filtering to this process's PID to isolate it from other GPU users on the machine):

```bash
UVICORN_PID=$(pgrep -f "uvicorn app.main:app")
watch -n 1 "nvidia-smi --query-compute-apps=pid,used_memory --format=csv | grep $UVICORN_PID"
```

In another terminal, run the benchmark with a longer sample (record or download a real 5-10 minute speech file as `long_sample.wav`):

```bash
./scripts/benchmark_concurrency.sh long_sample.wav 4
```

Expected: all 4 `result_*.json` files are non-empty and contain valid transcription JSON; the `used_memory` value for the uvicorn PID stays under `6144` MiB throughout.

- [ ] **Step 7: Manual test — queueing beyond MAX_CONCURRENT_TRANSCRIPTIONS**

```bash
./scripts/benchmark_concurrency.sh long_sample.wav 6
```

Expected: with `MAX_CONCURRENT_TRANSCRIPTIONS=4`, all 6 requests still complete successfully (the last 2 wait for a free slot instead of erroring) — all 6 `result_*.json` files are non-empty.

If VRAM exceeds the 6GB budget in Step 6, lower `MAX_CONCURRENT_TRANSCRIPTIONS` in `.env` (e.g. to `3`) or set `COMPUTE_TYPE=int8_float16`, restart the server, and re-run Step 6.

- [ ] **Step 8: Commit**

```bash
git add README.md scripts/benchmark_concurrency.sh
git commit -m "docs: README, real-model smoke test, and concurrency/VRAM benchmark script"
```

---

## Self-Review Notes

- **Spec coverage:** model lifecycle (Task 5 lifespan), API contract incl. all 3 response formats (Task 5 + tests), concurrency & VRAM management (Task 3 semaphore/threadpool + Task 6 benchmark), error handling incl. all 5 documented cases (Task 5), auth with default dev key + startup warning (Task 4 + Task 5), config env vars (Task 1), project structure (Tasks 1-5), testing plan (automated in Tasks 1-5, manual in Task 6) — all covered.
- **Placeholder scan:** no TBD/TODO markers remain; the spec's "TBD ct2 repo name" was resolved to `deepdml/faster-whisper-large-v3-turbo-ct2` (verified against its Hugging Face model card).
- **Type consistency:** `TranscriptionService.transcribe` signature (`audio_path, language, word_timestamps`) matches its Task 3 definition and every Task 5 call site; `SegmentResult`/`WordResult` field names match how Task 5 reads them (`s.id/s.start/s.end/s.text`, `w.word/w.start/w.end`).
