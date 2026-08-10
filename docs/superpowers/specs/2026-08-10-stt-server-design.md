# STT Server Design — whisper-large-v3-turbo

Date: 2026-08-10

## Goal

Server API speech-to-text đơn giản, xử lý nhiều request song song (2-4 concurrent, audio dài 1-10 phút), giữ tổng VRAM tiêu thụ trong khoảng 6GB trên GPU (RTX 3090, 24GB VRAM tổng, ~19GB rảnh hiện tại).

## Tech Stack

- **FastAPI + uvicorn** — HTTP server async.
- **faster-whisper** (CTranslate2) — dùng checkpoint đã convert của `openai/whisper-large-v3-turbo` (vd. `deepdml/faster-whisper-large-v3-turbo-ct2` trên HF Hub; verify tên repo cụ thể lúc implement). `device="cuda"`, `compute_type="float16"` mặc định.
- **ffmpeg** phải cài sẵn trên host — faster-whisper dùng để decode/resample audio input (mp3, wav, m4a, ogg...) về 16kHz mono.

Lý do chọn faster-whisper thay vì HuggingFace Transformers pipeline: CTranslate2 giải phóng GIL trong lúc infer, cho phép nhiều thread Python gọi `transcribe()` đồng thời và thực sự chạy song song trên GPU. VRAM footprint thấp hơn nhiều (model turbo ~1.6-2GB ở fp16) và dễ kiểm soát ngân sách 6GB hơn so với Transformers pipeline (vốn dễ bị block bởi GIL và tốn VRAM hơn khi phải serve nhiều request cùng lúc).

## Vòng đời model

Model được load **một lần duy nhất** lúc server khởi động (FastAPI lifespan hook), giữ 1 instance dùng chung cho toàn bộ request. Không load lại mỗi request, không load nhiều bản sao.

## API Contract

Tương thích OpenAI SDK — client có thể dùng thẳng OpenAI client, chỉ đổi `base_url`.

### `POST /v1/audio/transcriptions`

Request: `multipart/form-data`

| field | required | mô tả |
|---|---|---|
| `file` | có | file audio (mp3/wav/m4a/ogg...) |
| `model` | không | giá trị bị bỏ qua — server chỉ có 1 model (turbo) |
| `language` | không | mã ISO (vd `vi`, `en`); bỏ trống → auto-detect |
| `response_format` | không | `json` (default) \| `verbose_json` \| `text` |
| `timestamp_granularities[]` | không | `segment` (default) \| `word` — chỉ áp dụng khi `response_format=verbose_json` |

Response `json` (default):
```json
{ "text": "toàn bộ nội dung transcribe..." }
```

Response `verbose_json`:
```json
{
  "text": "...",
  "language": "vi",
  "duration": 123.4,
  "segments": [
    {"id": 0, "start": 0.0, "end": 4.2, "text": "xin chào..."}
  ],
  "words": [ {"word": "xin", "start": 0.0, "end": 0.3} ]
}
```
`words` chỉ xuất hiện khi `timestamp_granularities[]=word`.

Response `text`: plain text (`Content-Type: text/plain`).

### `GET /health`

Trả `{"status": "ok"}` khi model đã load xong và sẵn sàng nhận request.

## Concurrency & VRAM Management

- `MAX_CONCURRENT_TRANSCRIPTIONS` (mặc định `4`) — giới hạn bằng `asyncio.Semaphore`. Request vượt quá giới hạn sẽ **chờ theo FIFO**, không bị từ chối.
- `ThreadPoolExecutor` kích thước = `MAX_CONCURRENT_TRANSCRIPTIONS`, dùng để offload lệnh `transcribe()` (blocking) ra khỏi event loop — event loop luôn rảnh để nhận request/kết nối mới.
- `COMPUTE_TYPE` env var (`float16` mặc định, hoặc `int8_float16`) — đánh đổi chất lượng lấy VRAM nếu cần dồn thêm concurrency vào ngân sách 6GB.
- **Không có cơ chế ép cứng giới hạn VRAM** trong CTranslate2 (khác kiểu `memory_fraction` của TensorFlow). Ngân sách 6GB được đảm bảo bằng cách chọn `MAX_CONCURRENT_TRANSCRIPTIONS` + `COMPUTE_TYPE` hợp lý, và **benchmark thực tế** bằng `nvidia-smi` trong bước testing. Nếu VRAM vượt ngưỡng: giảm `MAX_CONCURRENT_TRANSCRIPTIONS` hoặc chuyển `int8_float16`.

## Error Handling

Format lỗi kiểu OpenAI: `{"error": {"message": "...", "type": "..."}}`

| Tình huống | HTTP status |
|---|---|
| Thiếu/sai field `file` | 400 |
| ffmpeg không decode được audio | 400 |
| File vượt `MAX_FILE_SIZE_MB` (default 200) | 413 |
| Sai/thiếu API key | 401 |
| Lỗi runtime lúc infer (kể cả GPU OOM) | 503 — bắt exception, KHÔNG crash process, log traceback đầy đủ ở server, trả message chung "server busy, try again" cho client |

## Auth

API key tĩnh qua env var `STT_API_KEY`, check header `Authorization: Bearer <key>`. Nếu `STT_API_KEY` chưa được set, dùng giá trị mặc định dev `changeme-please-set-STT_API_KEY` kèm cảnh báo log lúc khởi động.

## Config (env vars)

| var | default |
|---|---|
| `STT_MODEL_ID` | ct2 checkpoint của turbo (TBD tên repo cụ thể lúc implement) |
| `DEVICE` | `cuda` |
| `COMPUTE_TYPE` | `float16` |
| `MAX_CONCURRENT_TRANSCRIPTIONS` | `4` |
| `MAX_FILE_SIZE_MB` | `200` |
| `STT_API_KEY` | `changeme-please-set-STT_API_KEY` |
| `HOST` | `0.0.0.0` |
| `PORT` | `8000` |

## Cấu trúc project

```
stt_server/
  app/
    main.py        # FastAPI app, lifespan load model, routes
    transcribe.py  # wrap faster-whisper WhisperModel + concurrency control
    auth.py         # API key dependency
    schemas.py      # pydantic response models
  requirements.txt
  .env.example
  README.md
```

Chạy: venv + `pip install -r requirements.txt` (cần `ffmpeg` cài sẵn ở OS) → `uvicorn app.main:app --host 0.0.0.0 --port 8000`.

## Testing Plan

1. Smoke test: curl upload 1 file audio ngắn (vi + en), kiểm tra `verbose_json` trả đúng segments.
2. Concurrency test: bắn 4 request đồng thời với audio 5-10 phút, theo dõi `nvidia-smi -l 1` xác nhận VRAM < 6GB, đo latency.
3. Test hàng đợi: bắn 6 request cùng lúc (> `MAX_CONCURRENT_TRANSCRIPTIONS=4`), xác nhận 2 request cuối chờ rồi hoàn tất thay vì lỗi.
4. Test lỗi: file rỗng/hỏng, file quá lớn, sai/thiếu API key.

## Out of scope

- Streaming/WebSocket real-time transcription.
- Fine-tuning hoặc custom vocabulary.
- Multi-GPU / horizontal scaling.
- Persistent job queue (Redis/Celery) — dùng in-process semaphore + threadpool là đủ cho quy mô 2-4 concurrent.
