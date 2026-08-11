# Async Job API + queue_depth — Design

Date: 2026-08-11

## Goal

Fix a production issue: ~20% of `POST /v1/audio/transcriptions` requests hang for a fixed ~18s while queued behind other transcriptions (no bytes flow on the socket during that wait), then fail with `EPIPE`/`ETIMEDOUT`. Root cause: the server is exposed via home-router DDNS + port-forward with no reverse proxy; the router's NAT connection tracking almost certainly drops idle TCP flows with no traffic after a fixed ~15-30s window. The existing synchronous endpoint holds the client connection open — with no bytes sent — for the entire queue-wait + transcribe duration, which can exceed that window under load.

Fix: add an async submit-then-poll job API where every individual HTTP request (submit, poll) completes in milliseconds, so no connection is ever held open long enough to hit the NAT idle timeout. Also expose `queue_depth` on `/health` so batch clients can self-throttle instead of guessing.

## Non-goals

- Not touching the existing `POST /v1/audio/transcriptions` synchronous endpoint's behavior — it stays exactly as is, for callers who want a single request/response and can tolerate occasional timeouts on short/low-traffic use.
- Not adding webhook delivery (polling only for this iteration).
- Not adding persistent (cross-restart) job storage — in-memory only, accepted tradeoff.
- Not adding a `processing` job sub-state — jobs are `queued` until `completed`/`failed`. `queue_depth` on `/health` already exposes queue pressure globally.

## API Contract

### `POST /v1/audio/transcriptions/jobs`

Auth: same `Authorization: Bearer <STT_API_KEY>` as the sync endpoint.

Request: `multipart/form-data`

| Field | Required | Description |
|---|---|---|
| `file` | Yes | Audio file. Same validation as the sync endpoint: rejected with `400` if missing/empty, `413` if it exceeds `MAX_FILE_SIZE_MB`. |
| `language` | No | ISO code; omit for auto-detect. |
| `timestamp_granularities[]` | No | `word` — if present, word-level timestamps are computed and included in the completed result. |

These validation checks (file present, non-empty, within size limit) run synchronously in the request handler — fast, bounded, no queue wait involved, so no NAT-timeout risk at submit time. Once validated, the file is copied to a durable temp file (not auto-deleted when the handler returns), a `job_id` (uuid4 hex) is generated, a job record is stored, and the actual transcription is scheduled as a background `asyncio` task. The handler returns immediately without waiting on the transcription semaphore.

Response: **`202 Accepted`**

```json
{ "job_id": "b3f1...", "status": "queued" }
```

Error responses (400/413/401) use the same `{"error": {"message", "type"}}` shape as the sync endpoint, same trigger conditions (missing/empty file → 400 `invalid_request_error`; oversized → 413 `file_too_large`; bad/missing key → 401 `invalid_api_key`).

### `GET /v1/audio/transcriptions/jobs/{job_id}`

Auth: same as above.

Response: **`200 OK`**

```json
{
  "job_id": "b3f1...",
  "status": "queued",
  "result": null,
  "error": null
}
```

Once finished:

```json
{
  "job_id": "b3f1...",
  "status": "completed",
  "result": {
    "text": "...",
    "language": "vi",
    "duration": 12.4,
    "segments": [ { "id": 0, "start": 0.0, "end": 2.1, "text": " ..." } ],
    "words": null
  },
  "error": null
}
```

Or on failure:

```json
{
  "job_id": "b3f1...",
  "status": "failed",
  "result": null,
  "error": { "message": "Could not decode audio file", "type": "invalid_request_error" }
}
```

Failure error types/messages mirror the sync endpoint's transcription failure handling: `(OSError, ValueError, IndexError)` during transcription → `invalid_request_error` / "Could not decode audio file"; any other exception → `server_error` / "server busy, try again".

Unknown or expired `job_id`: **`404 Not Found`**

```json
{ "error": { "message": "Job not found", "type": "invalid_request_error" } }
```

### `GET /health` — add `queue_depth`

```json
{ "status": "ok", "queue_depth": 2 }
```

`queue_depth` = number of transcription calls (sync endpoint **or** job worker) currently blocked waiting for a `TranscriptionService` semaphore slot — i.e. accepted but not yet started on the GPU. Does not include calls actively transcribing. Omitted (as today) when the model hasn't finished loading — response stays `503 {"status": "loading"}` with no `queue_depth` key in that case.

## Component Changes

- **`app/transcribe.py` (`TranscriptionService`):** add an internal waiting counter, incremented before acquiring the semaphore and decremented immediately once acquired (via explicit `try/finally` around `semaphore.acquire()`, not a nested `async with`, so the counter is correct even if the wait is cancelled). Expose it as a read-only `queue_depth` property. This is the only change to existing code — the semaphore/executor concurrency mechanism itself is unchanged.
- **`app/jobs.py` (new):** in-memory job store. A `Job` dataclass (`job_id`, `status`, `result`, `error`, `created_at`). A `JobStore` class: `create() -> Job`, `get(job_id) -> Job | None`, `mark_completed(job_id, result)`, `mark_failed(job_id, error)`, and `purge_expired()` (drops jobs older than `JOB_RESULT_TTL_SECONDS`, called opportunistically on every `create()`). No dependency on FastAPI/transcribe — plain data structure, independently testable.
- **`app/main.py`:** add the two new routes, wire a module-level `JobStore` instance (parallel to how `TranscriptionService` is wired via `app.state`), add a background-task runner function that calls `service.transcribe(...)`, maps exceptions to `Job.error` the same way the sync endpoint maps them to `HTTPException`, and deletes the temp file in a `finally` block. Keep a module-level `set()` of in-flight `asyncio.Task` objects with a `add_done_callback` to discard themselves, to prevent the "task garbage-collected mid-run" asyncio pitfall from unreferenced `create_task()` calls. Add `queue_depth` to the `/health` handler's 200-branch.

## Config

New env vars, both with defaults, both documented in `.env.example`/README:

| Var | Default | Description |
|---|---|---|
| `JOB_RESULT_TTL_SECONDS` | `3600` | How long a completed/failed job's result stays available for polling before being purged from memory. |

## Error Handling

- Job submit validation errors (missing/empty file, oversized) — identical to sync endpoint, returned synchronously at submit time, job is never created.
- Job worker failures (decode error, runtime error) — captured into `Job.error`, never raised/propagated (no client is synchronously waiting), logged server-side the same way the sync endpoint logs them today.
- Temp file cleanup: `finally` block in the background runner always attempts `os.unlink`, tolerating the file already being gone.
- Server shutdown: unchanged from today — `TranscriptionService.shutdown()` still drains the executor. Background job tasks still waiting on the semaphore when the process is killed are not gracefully cancelled (accepted simplification, matches "simple server" scope — the process exiting takes them down regardless).

## Testing Plan

- `app/jobs.py`: unit tests for `JobStore` — create/get, mark_completed, mark_failed, purge_expired (using an injectable "now" or a short TTL + real short sleep, whichever keeps the test fast and deterministic).
- `TranscriptionService.queue_depth`: extend the existing fake-model concurrency test to assert `queue_depth` rises while requests wait and drops as slots free up.
- `app/main.py`: `TestClient`-based tests for the new endpoints using the existing fake-service-via-`dependency_overrides` pattern — submit → 202 + job_id; poll immediately after a fake service that completes synchronously → `completed` with the right result shape; poll unknown id → 404; submit with missing/oversized/empty file → same 400/413 as sync endpoint; `/health` includes `queue_depth`.
- No new manual/GPU verification step is strictly required beyond what Task 6 (original plan) already covered, but a quick manual submit+poll cycle against the real model is worth doing once, given this is the fix for a real production symptom.

## Out of Scope (explicitly deferred)

- Webhook/callback delivery.
- Persistent (SQLite/file-backed) job storage.
- `processing` job sub-state.
- TCP keepalive tuning or any other network-layer mitigation — the async job pattern is the chosen fix; other options are not being pursued in parallel.
