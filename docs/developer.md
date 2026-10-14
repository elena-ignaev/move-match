# MoveMatch Developer Notes

## Current Flow

- `/` lets the single user upload a new dance video or open the past-video library.
- `/library` lists uploaded videos.
- `/videos/<id>` shows the original video, annotated video, analysis details, and ghost attempts.
- `/dance-practice/<id>` is the frame-by-frame/slowed annotated-video player.
- `/videos/<id>/ghost` is ghost mode: camera preview, selected reference speed, translucent annotated-reference overlay, learner recording, and upload.

## Ghost Mode

Ghost mode uses browser APIs:

- `navigator.mediaDevices.getUserMedia()` requests webcam access.
- `MediaRecorder` records the learner stream at the camera's normal frame rate.
- The reference video playback rate is set to `0.25`, `0.5`, or `0.75`.
- Recording starts only after camera preview and reference media are prepared.
- Recording stops when the reference ends or when the user presses Stop.
- Camera tracks are stopped after recording or when leaving the page.
- Each attempt stores a small `sync_events` timeline in `Attempt.details_json`.

The translucent overlay currently uses the annotated reference video as the overlay source, which places the reference skeleton over the learner preview. Browser-side learner pose detection is not implemented yet.

## Synchronization Notes

For the current prototype, the learner recording starts at the controlled ghost-mode start and the selected reference speed is saved on the `Attempt`.

Do not score future evaluations by assuming reference frame `N` equals learner frame `N`. Cameras can use variable frame rates, users can react late, and the reference can pause or stall. Instead, use the stored sync metadata:

- `reference_started_at` and `reference_stopped_at`
- `reference_time` at stop
- `sync_events[]`, including `recording_start`, `reference_play`, `reference_pause`, `reference_waiting`, `reference_playing`, `reference_ended`, `manual_stop`, and `recording_stop`

Each sync event includes a monotonic `session_time_ms`, the reference video's `currentTime`, the playback rate, and a wall-clock timestamp. Future evaluation code should map learner recording time to reference timeline time through these events.

## Backend Recording State

Attempts are tracked separately from videos:

- `Attempt.video_id`
- `Attempt.recording_filename`
- `Attempt.speed`
- `Attempt.status`
- `Attempt.details_json`
- `Attempt.created_at`

Routes:

- `GET /videos/<id>/ghost`: ghost-mode page.
- `POST /videos/<id>/attempts`: stores the learner recording in `storage/attempts`.
- `GET /attempts/<id>/status`: returns attempt status JSON.

Evaluation is intentionally not implemented yet. New attempts are saved with `status="recorded"`.

## Local Notes

- The app uses SQLite at `videos.sqlite3`.
- Uploaded source videos live in `storage/uploads`.
- Annotated videos live in `storage/annotated`.
- Ghost recordings live in `storage/attempts`.
- Webcam access works on `localhost` during development. Production should use HTTPS.
