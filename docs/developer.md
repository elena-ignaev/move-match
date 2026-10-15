# Developer notes

## Routes

| Route | Purpose |
|---|---|
| `GET /`, `GET /library`, `GET /privacy` | home (journey, upload), video list, data policy |
| `POST /upload` | save + validate a reference (ffprobe), queue processing, redirect to its page |
| `GET /videos/<id>` | reference player with skeleton, processing progress, attempts table, delete |
| `GET /api/videos/<id>/status` | JSON progress for polling |
| `POST /videos/<id>/reprocess`, `POST /videos/<id>/delete` | retry processing, delete everything for a video |
| `GET /videos/<id>/media` | browser playback file (H.264 + AAC, constant frame rate, range requests) |
| `GET /api/videos/<id>/landmarks` | image-space landmarks for drawing skeletons on a canvas |
| `GET /videos/<id>/practice` | section loop, slow playback, frame stepping, mirror, skeleton |
| `GET /videos/<id>/ghost` | camera + ghost, countdown, recording (accepts `?start&end&speed&mirror&follow`) |
| `POST /videos/<id>/attempts` | multipart upload of a recording + sync log; JSON in and out; idempotent by `upload_id` |
| `GET /attempts/<id>` | replay side by side / overlay, feedback, comparison with the previous attempt |
| `GET /api/attempts/<id>` (alias `/attempts/<id>/status`) | status, replay map, feedback, comparison |
| `GET /attempts/<id>/recording` | the replay MP4 (or the raw upload while it is processing) |
| `POST /attempts/<id>/reevaluate`, `POST /attempts/<id>/delete` | recompute feedback, delete |
| `/dev/landmarks/...` | landmark inspector, only with `ENABLE_DEV_TOOLS=1` |

All POSTs need a CSRF token (form field `csrf_token` or header `X-CSRFToken`).

## Processing pipeline

```
upload ──ffprobe──> Video(queued) ──job──> transcode CFR + AAC ──> MediaPipe ──> landmarks .npz ──> completed
record ──sync log──> Attempt(uploaded) ──job──> transcode CFR ──> replay map ──> MediaPipe ──> feedback ──> ready | withheld
```

Jobs run on a per-app thread pool (`jobs.py`); `JOBS_SYNC=True` runs them inline (tests). Interrupted jobs
are re-queued on the next start. Status values: video `queued → processing → completed | failed`
(`needs_reprocessing` for prototype rows); attempt `uploaded → processing → ready | withheld | failed`.

## Storage layout (`storage/`)

`uploads/` original references · `playback/` browser-ready references · `landmarks/` `.npz` caches for
references and attempts · `attempts/` raw recordings (deleted after transcoding) · `replays/` replay MP4s.

## Sync log sent with each attempt

```json
{"recorder_start_ms": 412.3,
 "samples": [[1450.1, 10.0], [1483.4, 10.0167]],
 "events": [{"type": "reference_play", "session_time_ms": 1440.2, "reference_time": 10.0, "playback_rate": 0.5}]}
```

All times share one `performance.now()` clock started just before the recorder. See `sync_timeline.py`.

## Changing the schema

Edit `models.py`, then `flask db migrate -m "..."`, read the generated file, `flask db upgrade`, and commit it.
`flask db check` confirms the models and migrations agree. Startup never creates or alters tables.

## Changing the feedback method

Change `feedback.py`, run `python evaluation/run_eval_set.py`, update the expectations or the docs if
behaviour changed on purpose, and bump `ALGORITHM_VERSION` (stored with every evaluation). Use
`POST /attempts/<id>/reevaluate` to recompute old attempts.
