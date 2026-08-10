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
