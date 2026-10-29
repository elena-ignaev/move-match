# MoveMatch

**Learn a dance routine section by section with your webcam, then see where your attempt differed and when.**

MoveMatch is a local web app for recreational dancers preparing for a performance. You pick a short
reference video, practise one section slowly with the music and a mirrored "ghost" skeleton, record an
attempt, replay it next to the reference, and get specific feedback: how many tenths of a second you were
behind, and which joints differed at the key moments of the section. Then you retry and compare.

## Who it is for, and the problem

Learning choreography from a video is hard on your own. You can't see yourself while you dance, it's easy to
mirror the wrong side, the music is too fast to catch details, and "it looks a bit off" doesn't say what to
fix. In a group, instructors spend most rehearsal time correcting individuals one by one.

## Why I built it

Throughout my performance preparation at NUS High, cleaning movements and synchronizing dancers was the most
time-consuming part: more than 100 hours for a 4.5-minute performance. Much of that time went into the gap
between how instructors (my co-leaders and I) saw a move and described it, and how each dancer, with their own
body and their own reading of an instruction, actually performed it.

I want each dancer to be able to practise against the same reference on their own, with feedback that is
specific enough to act on, so that group rehearsals can focus on bringing the performance together.

## Demo

*A 60–90 second demo video of one complete journey is planned; see [`docs/demo_script.md`](docs/demo_script.md)
for the shot list.*

## Screenshots

| Practise a section | Record with the ghost | Feedback |
|---|---|---|
| ![Practice page](docs/images/manual/04_practice.jpg) | ![Recording](docs/images/manual/10_ghost_recording.jpg) | ![Feedback](docs/images/manual/14_feedback.jpg) |

More in [`docs/images/manual/`](docs/images/manual/).

## The journey

1. **Choose a short reference** (≤ 90 s; one dancer, full body, front view). Processing runs in the background
   with a progress bar. The music is kept.
2. **Practise a section**: loop it, slow it to 0.25–0.75× (pitch preserved), step frame by frame, toggle the
   skeleton overlay and mirror view.
3. **Record an attempt**: a full-body positioning guide, a 3/5/10-second countdown with beeps, and the
   reference skeleton as a ghost over your camera at any speed, including normal speed.
4. **Replay side by side** (or as a ghost overlay): the reference follows the exact moment it was showing
   while you recorded, including slow speed and buffering pauses.
5. **Read feedback**: a provisional score out of 100, your timing offset, the key moments worth reviewing
   (with "Show in replay"), and the joint-angle differences behind each note. If tracking was poor you get
   the reason ("your legs were below the bottom edge of the frame"), not a low score.
6. **Retry and compare**: every attempt at the same section is judged at the same moments, so you can see
   which moments improved and how the score changed.

Each attempt also gets a **provisional score out of 100** for tracking your progress on a section:
posture at key moments 70%, timing 20%, completion 10%, always shown with that breakdown. There is no
score when feedback is withheld, so a camera problem never turns into a low mark. The score has not yet
been checked against a teacher's judgment (see [`docs/evaluation.md`](docs/evaluation.md)), so the notes
explain more than the number.

## Quick start

Requires Python 3.11 and FFmpeg (`brew install ffmpeg`). Full guide, including troubleshooting:
[`docs/setup.md`](docs/setup.md).

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
python scripts/download_model.py
cp .env.example .env
flask db upgrade
flask run
```

Open <http://localhost:5000>. There is no sample dance footage in the repository (no clip with clear
redistribution rights). Record a 10–20 s clip of yourself, or see `flask add-sample` in the setup guide.

```bash
pytest
```

240 automated tests, including one complete journey on a generated fixture video.

## How it works

| Step | What happens | Code |
|---|---|---|
| Pose detection | Google's pretrained **MediaPipe Pose Landmarker** finds 33 body landmarks per frame. MoveMatch does not train a model. | `process_video.py` |
| Constant frame rate | Every video is transcoded once to a constant frame rate (keeping the audio), so landmark timestamps equal the browser's playback time. | `media.py` |
| Synchronization | While recording, the browser logs which reference frame was on screen at each moment. The server maps every recorded frame to that reference time; stalls and pauses are excluded. | `sync_timeline.py`, `static/js/ghost_mode.js` |
| Normalization | Hip-centred, scaled by a smoothed torso length; orientation kept (turns are choreography). Joint angles do not depend on body size or camera distance. | `pose_utils.py`, `pose_pipeline.py` |
| Feedback | Tracking check → timing offset → joint angles at reference key moments, with timing removed; a flag needs the 3-D and 2-D views to agree. | `feedback.py` |
| Background jobs | Uploads return immediately; processing reports progress and can be retried. | `jobs.py` |

The reasoning, including what was tried and changed, is in
[`docs/technical_decisions.md`](docs/technical_decisions.md): normalization, synchronization, missing
landmarks, mirror mode, and the limitations of one camera.

## Status

**Working**
* The full journey above, on a laptop with a webcam (Chrome, Edge, Firefox or Safari).
* Feedback: timing offset with drift detection, key-moment joint angles, missed-turn detection, partial or
  withheld feedback when parts of the body are not visible.
* Reliability: background processing with progress, content-based upload validation with duration,
  resolution and size limits, retryable uploads that never duplicate, cleanup of failed and intermediate
  files, Alembic-only migrations.
* Privacy: delete controls, a privacy page, and an optional public-demo mode (anonymous ownership, CSRF
  protection, rate limits, 24-hour retention, strict security headers, required production secret).

**Limitations**
* Tested so far on synthetic cases with known answers (20 of 21 behave as expected) and on a handful of real
  recordings, **not yet against a dance teacher's judgment** ([`docs/evaluation.md`](docs/evaluation.md)).
* One camera: depth is guessed, so reaching forward vs. sideways is invisible to joint angles; back-facing
  parts can swap left and right; symmetric moves hide mirror mistakes.
* One dancer, a fixed camera, the whole body visible, short sections.
* Thresholds (20°/35°, ±0.15 s) and the score weights (70/20/10) are provisional.

**Planned**
* Teacher comparison study and user testing with dancers from my group ([`docs/user_testing.md`](docs/user_testing.md)).
* A live "you're out of frame" warning during recording (in-browser pose detection).
* Beat-aware timing (offsets in beats instead of seconds).
* A 3-D body-mesh model. I first tried HybrIK (hybrid analytical-neural inverse kinematics for body mesh
  recovery), which promises more precise 3-D poses, but I could not install its dependencies on macOS.

## Evidence of iteration

* **Prototype** ([`assets/prototype/`](assets/prototype/)): uploads processed inside the request, a silent
  annotated video, two unsynchronized video players for ghost mode, attempts saved but never evaluated.
* **Problems found and fixed:** music dropped by the conversion; ghost drift after buffering; the recording
  could keep running after a playback error; double clicks started overlapping recordings; failed uploads were
  lost; variable-frame-rate timing; a migration chain that only worked on an existing database.
* **Changes driven by the evaluation set:** false "bent knee" flags from depth error led to the 3-D/2-D
  agreement rule; moments after an early stop are no longer judged; per-side tracking regions. Details in
  [`docs/evaluation.md`](docs/evaluation.md).
* **Feedback from other users:** none recorded yet; the plan is in [`docs/user_testing.md`](docs/user_testing.md).

## Repository layout

```
app.py, config.py, models.py, security.py   Flask app, settings, database models, ownership/CSRF/limits
jobs.py, media.py, process_video.py         background processing, FFmpeg, MediaPipe extraction
sync_timeline.py, feedback.py               synchronization and the feedback method
pose_utils.py, pose_pipeline.py             normalization, angles, mirror convention
landmark_store.py                           versioned landmark cache (.npz) with stale-cache detection
templates/, static/                         pages and browser code (no build step)
migrations/                                 Alembic schema history
evaluation/                                 synthetic evaluation set and teacher-comparison tooling
research/                                   DTW alignment and score-calibration experiments (not used for feedback)
scripts/                                    model download; scripts/demos/ = manual MediaPipe demos (not tests)
tests/                                      automated pytest suite
docs/                                       setup, technical decisions, evaluation, user testing, developer notes
```

## License and credits

Code: MIT ([`LICENSE`](LICENSE)). Third-party models and libraries: [`CREDITS.md`](CREDITS.md).
