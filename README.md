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

## Configuration

All configuration is via environment variables (see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `STT_MODEL_ID` | `deepdml/faster-whisper-large-v3-turbo-ct2` | CTranslate2 model repo/path passed to `WhisperModel` |
| `DEVICE` | `cuda` | Inference device (`cuda` or `cpu`) |
| `COMPUTE_TYPE` | `float16` | CTranslate2 compute type (e.g. `float16`, `int8`) |
| `MAX_CONCURRENT_TRANSCRIPTIONS` | `4` | Max simultaneous transcriptions; also sets `WhisperModel(num_workers=...)` so CTranslate2 actually parallelizes |
| `MAX_FILE_SIZE_MB` | `200` | Max accepted upload size, rejected with `413` above this |
| `STT_API_KEY` | `changeme-please-set-STT_API_KEY` | Bearer token required on `/v1/audio/transcriptions`; a warning is logged at startup if left at the default |
| `HOST` | `0.0.0.0` | Bind address for uvicorn |
| `PORT` | `8000` | Bind port for uvicorn |

## Usage

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer $STT_API_KEY" \
  -F "file=@sample.wav" \
  -F "response_format=verbose_json" \
  -F "timestamp_granularities[]=word"
```

### Health check

```bash
curl http://localhost:8000/health
```

Returns `200 {"status": "ok"}` once the model has loaded and the transcription service is ready. Before that (or if startup fails), it returns `503 {"status": "loading"}`.

### Error responses

All error responses (auth failures, invalid requests, decode failures, internal errors, and framework-level 404/405s) share one shape:

```json
{"error": {"message": "human-readable message", "type": "error_type"}}
```

| HTTP status | `type` | When |
|---|---|---|
| 400 | `invalid_request_error` | Malformed request, unsupported `response_format`, empty file, or audio that can't be decoded |
| 401 | `invalid_api_key` | Missing or incorrect `Authorization: Bearer <key>` header |
| 413 | `file_too_large` | Upload exceeds `MAX_FILE_SIZE_MB` |
| 500 | `server_error` | Any unexpected unhandled exception |
| 503 | `server_error` | Transcription failed for a reason other than a decode error (e.g. transient GPU/model failure) |

## Tests

```bash
pytest -v
```

All automated tests use fake models and run on CPU without touching the GPU.

## Manual GPU Verification

The automated test suite never loads the real model or touches the GPU, so a manual pass is required to confirm real transcriptions and real concurrency:

1. Obtain a short sample audio file (e.g. `sample.wav`), or generate one with `ffmpeg -f lavfi -i "sine=frequency=440:duration=3" sample.wav`.
2. Start the server as described above and wait for it to finish loading the model (`curl http://localhost:8000/health` should return `{"status": "ok"}`).
3. Send a single request as shown in [Usage](#usage) and confirm you get back a real transcription.
4. Run the concurrency/VRAM benchmark while watching GPU utilization:

   ```bash
   nvidia-smi -l 1 &
   STT_API_KEY=$STT_API_KEY BASE_URL=http://localhost:8000 \
     scripts/benchmark_concurrency.sh sample.wav 4
   ```

   The script fires `N` concurrent requests, reports each response's HTTP status and byte size, fails loudly if any request didn't return `200`, and prints total wall-clock time for the batch. With `MAX_CONCURRENT_TRANSCRIPTIONS` set above 1, you should see multiple requests being processed on the GPU at once in `nvidia-smi` rather than serialized one-at-a-time.
