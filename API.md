# STT Server API Reference

Base URL: `http://<host>:<port>` (defaults to `http://localhost:8000`, configurable via `HOST`/`PORT` in `.env`).

All endpoints return JSON unless noted otherwise. All error responses share one shape:

```json
{ "error": { "message": "human-readable message", "type": "error_type" } }
```

## Authentication

Every request to `/v1/audio/transcriptions` requires a bearer token matching `STT_API_KEY` from `.env`:

```
Authorization: Bearer <STT_API_KEY>
```

`GET /health` does not require authentication.

---

## `GET /health`

Liveness/readiness check. No auth required.

### curl

```bash
curl -s http://localhost:8000/health
```

### Response — model loaded and ready

**`200 OK`**

```json
{ "status": "ok", "queue_depth": 0 }
```

`queue_depth` is the number of transcription requests (sync or async job) currently waiting for a free GPU slot — not counting ones actively transcribing. Use it to self-throttle batch submission instead of guessing.

### Response — server still starting up (model not loaded yet)

**`503 Service Unavailable`**

```json
{ "status": "loading" }
```

---

## `POST /v1/audio/transcriptions`

Transcribes an uploaded audio file. Request/response shape follows OpenAI's `/v1/audio/transcriptions` contract, so existing OpenAI SDK clients work by just pointing `base_url` at this server.

Content type: `multipart/form-data`.

### Form fields

| Field | Required | Default | Description |
|---|---|---|---|
| `file` | Yes | — | Audio file (`mp3`, `wav`, `m4a`, `ogg`, ...). Anything `ffmpeg` can decode. |
| `model` | No | *(ignored)* | Accepted for OpenAI SDK compatibility but ignored — this server only serves one model. |
| `language` | No | auto-detect | ISO language code (e.g. `vi`, `en`). Omit to let Whisper auto-detect. |
| `response_format` | No | `json` | One of `json`, `verbose_json`, `text`. |
| `timestamp_granularities[]` | No | *(none)* | `word` — only has an effect when `response_format=verbose_json`. Any other/omitted value returns segment-level timestamps only. |

### Limits

- Max upload size: `MAX_FILE_SIZE_MB` (default `200` MB) → rejected with `413` above this.
- Max concurrent transcriptions in flight: `MAX_CONCURRENT_TRANSCRIPTIONS` (default `4`) — extra requests queue and wait for a free slot rather than being rejected.

---

### `response_format=json` (default)

Returns just the transcribed text.

#### curl

```bash
curl -s -X POST http://localhost:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer $STT_API_KEY" \
  -F "file=@sample.wav"
```

#### Response

**`200 OK`** — `Content-Type: application/json`

```json
{
  "text": "Xin chào, đây là bản ghi âm thử nghiệm."
}
```

---

### `response_format=verbose_json`

Returns text plus language, duration, and segment-level timestamps.

#### curl

```bash
curl -s -X POST http://localhost:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer $STT_API_KEY" \
  -F "file=@sample.wav" \
  -F "response_format=verbose_json"
```

#### Response

**`200 OK`** — `Content-Type: application/json`

```json
{
  "text": "Xin chào, đây là bản ghi âm thử nghiệm.",
  "language": "vi",
  "duration": 4.2,
  "segments": [
    { "id": 0, "start": 0.0, "end": 2.1, "text": " Xin chào," },
    { "id": 1, "start": 2.1, "end": 4.2, "text": " đây là bản ghi âm thử nghiệm." }
  ],
  "words": null
}
```

`words` is `null` unless `timestamp_granularities[]=word` was also passed (see below).

---

### `response_format=verbose_json` + `timestamp_granularities[]=word`

Adds word-level timestamps flattened across all segments.

#### curl

```bash
curl -s -X POST http://localhost:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer $STT_API_KEY" \
  -F "file=@sample.wav" \
  -F "response_format=verbose_json" \
  -F "timestamp_granularities[]=word"
```

#### Response

**`200 OK`** — `Content-Type: application/json`

```json
{
  "text": "Xin chào, đây là bản ghi âm thử nghiệm.",
  "language": "vi",
  "duration": 4.2,
  "segments": [
    { "id": 0, "start": 0.0, "end": 2.1, "text": " Xin chào," },
    { "id": 1, "start": 2.1, "end": 4.2, "text": " đây là bản ghi âm thử nghiệm." }
  ],
  "words": [
    { "word": "Xin", "start": 0.0, "end": 0.3 },
    { "word": "chào,", "start": 0.3, "end": 0.8 },
    { "word": "đây", "start": 2.1, "end": 2.4 },
    { "word": "là", "start": 2.4, "end": 2.6 },
    { "word": "bản", "start": 2.6, "end": 2.9 },
    { "word": "ghi", "start": 2.9, "end": 3.1 },
    { "word": "âm", "start": 3.1, "end": 3.3 },
    { "word": "thử", "start": 3.3, "end": 3.7 },
    { "word": "nghiệm.", "start": 3.7, "end": 4.2 }
  ]
}
```

---

### `response_format=text`

Returns plain text, no JSON wrapper.

#### curl

```bash
curl -s -X POST http://localhost:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer $STT_API_KEY" \
  -F "file=@sample.wav" \
  -F "response_format=text"
```

#### Response

**`200 OK`** — `Content-Type: text/plain; charset=utf-8`

```
Xin chào, đây là bản ghi âm thử nghiệm.
```

---

### Specifying language explicitly

Skips auto-detection (slightly faster, and more reliable for short/ambiguous audio).

```bash
curl -s -X POST http://localhost:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer $STT_API_KEY" \
  -F "file=@sample.wav" \
  -F "language=vi"
```

---

## Error Responses

All errors share the `{"error": {"message", "type"}}` shape.

| HTTP status | `type` | When | Example trigger |
|---|---|---|---|
| `400` | `invalid_request_error` | Missing `file` field, empty file, unsupported `response_format`, or audio that can't be decoded | malformed multipart body, corrupt audio |
| `401` | `invalid_api_key` | Missing or incorrect `Authorization: Bearer <key>` header | wrong/omitted API key |
| `413` | `file_too_large` | Upload exceeds `MAX_FILE_SIZE_MB` | oversized file |
| `404` | `error` | No route matches the request path | typo in URL |
| `404` | `invalid_request_error` | (Job API only) `job_id` not found or expired — distinct from the generic route-not-found 404 above, which uses `type: "error"` | unknown/expired job id on `GET /jobs/{id}` |
| `405` | `error` | HTTP method not allowed for the matched route | `GET /v1/audio/transcriptions` |
| `500` | `server_error` | Any unexpected unhandled exception | bug / edge case |
| `503` | `server_error` | Transcription failed for a reason other than a decode error (e.g. transient GPU/model failure). Also returned by `POST /jobs` when `MAX_PENDING_JOBS` in-flight jobs are already queued/running | GPU OOM, runtime error; or too many pending jobs |

### Example — missing/wrong API key

```bash
curl -s -X POST http://localhost:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer wrong-key" \
  -F "file=@sample.wav"
```

**`401 Unauthorized`**

```json
{ "error": { "message": "Invalid or missing API key", "type": "invalid_api_key" } }
```

### Example — missing file field

```bash
curl -s -X POST http://localhost:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer $STT_API_KEY"
```

**`400 Bad Request`**

```json
{ "error": { "message": "Invalid request", "type": "invalid_request_error" } }
```

### Example — empty file

```bash
curl -s -X POST http://localhost:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer $STT_API_KEY" \
  -F "file=@/dev/null"
```

**`400 Bad Request`**

```json
{ "error": { "message": "Uploaded file is empty", "type": "invalid_request_error" } }
```

### Example — file too large

```bash
curl -s -X POST http://localhost:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer $STT_API_KEY" \
  -F "file=@huge_recording.wav"
```

**`413 Payload Too Large`**

```json
{ "error": { "message": "File exceeds max size of 200MB", "type": "file_too_large" } }
```

### Example — invalid `response_format`

```bash
curl -s -X POST http://localhost:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer $STT_API_KEY" \
  -F "file=@sample.wav" \
  -F "response_format=srt"
```

**`400 Bad Request`**

```json
{ "error": { "message": "Unsupported response_format: srt", "type": "invalid_request_error" } }
```

### Example — audio that can't be decoded

```bash
curl -s -X POST http://localhost:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer $STT_API_KEY" \
  -F "file=@not_actually_audio.txt"
```

**`400 Bad Request`**

```json
{ "error": { "message": "Could not decode audio file", "type": "invalid_request_error" } }
```

---

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

#### Response — too many pending jobs

**`503 Service Unavailable`**

Returned when `MAX_PENDING_JOBS` jobs are already queued/running. Retry later.

```json
{ "error": { "message": "Too many pending jobs, try again later", "type": "server_error" } }
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

---

## Notes

- Requests beyond `MAX_CONCURRENT_TRANSCRIPTIONS` are queued (not rejected) and processed as GPU slots free up — a burst of requests all eventually succeed, just with added latency for the queued ones.
- See `README.md` for setup, configuration, and how to run the manual concurrency/VRAM benchmark (`scripts/benchmark_concurrency.sh`).
