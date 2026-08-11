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
    job_result_ttl_seconds: int = 3600


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
