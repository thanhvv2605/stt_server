# STT Server

Simple FastAPI speech-to-text server using `whisper-large-v3-turbo` via faster-whisper (CTranslate2), OpenAI-compatible `/v1/audio/transcriptions` endpoint.

## Prerequisites

- Python 3.12+
- NVIDIA GPU + driver with CUDA support
- `ffmpeg` installed on the host: `sudo apt install ffmpeg`

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env and set a real STT_API_KEY
```

## Run

```bash
set -a; source .env; set +a
uvicorn app.main:app --host "$HOST" --port "$PORT"
```

First startup downloads the model from the Hugging Face Hub (requires internet access) and can take a few minutes.

## Usage

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer $STT_API_KEY" \
  -F "file=@sample.wav" \
  -F "response_format=verbose_json" \
  -F "timestamp_granularities[]=word"
```

## Tests

```bash
pytest -v
```

All automated tests use fake models and run on CPU without touching the GPU. Real-model verification is manual — see below.
