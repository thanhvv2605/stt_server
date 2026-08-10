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
