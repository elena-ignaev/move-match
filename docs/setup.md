# Setup guide

These steps were verified on **macOS 15 (Apple Silicon) with Python 3.11.15 and FFmpeg 9.0** in a
brand-new virtual environment and a fresh clone (see "Verification log" at the end). Linux works the same way;
on Windows use `py -3.11` and `.venv\Scripts\activate`.

## 1. Requirements

| What | Version | Why |
|---|---|---|
| Python | 3.11 (`.python-version`) | MediaPipe 1.1 wheels; 3.12+ is not verified |
| FFmpeg + ffprobe | 5.1 or newer | inspecting uploads, transcoding with audio at a constant frame rate |
| A webcam and a recent Chrome, Edge, Firefox or Safari | | recording attempts (camera access works on `http://localhost` or HTTPS) |

FFmpeg is a system program, not a Python package (the old `ffmpeg` entry in `requirements.txt` installed an
unrelated library). Install it with:

```bash
brew install ffmpeg
```

Ubuntu/Debian: `sudo apt install ffmpeg`. Windows: `winget install Gyan.FFmpeg`. Check with `ffmpeg -version`.

## 2. Install

```bash
git clone https://github.com/elena-ignaev/move-match.git
```

```bash
cd move-match
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

`requirements.txt` pins every runtime dependency. PyTorch is **not** needed: MoveMatch uses Google's pretrained
MediaPipe Pose Landmarker, which ships its own runtime. (`torch`, `torchvision`, `pymysql` and `ffmpeg` were
removed from the requirements; they were left over from trying HybrIK, which never ran on macOS.)

## 3. Download the model

```bash
python scripts/download_model.py
```

This downloads `pose_landmarker_lite.task` (5.8 MB, version 1 of the float16 model) from Google's official
storage into `pretrained_models/`, and verifies its SHA-256 checksum
(`59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a`). The file is not committed because it is a
binary we did not create (it is Apache-2.0 licensed; see `CREDITS.md`). `python scripts/download_model.py --check`
verifies an existing copy.

## 4. Configure

```bash
cp .env.example .env
```

The defaults are fine for local use. Every setting is documented in `.env.example` (limits, storage folder,
public-demo mode). Flask reads `.env` automatically.

## 5. Create the database

```bash
flask db upgrade
```

The schema is managed only by Alembic migrations in `migrations/`; the app no longer creates or patches tables
at startup. If you start the server before this step, every page explains what to run.

### Existing databases (from the first prototype)

Databases created by the prototype's `db.create_all()` already contain the tables but no migration record.
Back up, mark the database at the revision it matches, then upgrade:

```bash
cp videos.sqlite3 videos.backup.sqlite3
flask db stamp b57605eb2ee6
flask db upgrade
```

Old videos become "needs reprocessing" (their annotated videos had no audio and their landmarks were timed by
nominal frame rate); press **Reprocess** on the video page. Old attempts are evaluated with their saved
play/pause events on the next start.

## 6. Run

```bash
flask run
```

Open <http://localhost:5000>. Upload a short clip (≤ 90 s) of one dancer, full body visible, filmed from the
front; processing runs in the background with a progress bar.

## 7. Try it with a sample video

There is no third-party dance footage in the repository, because we could not confirm permission to
redistribute any. To try the full journey:

1. Record yourself (or a friend who agrees) dancing for 10–20 seconds: phone on a stand, whole body in frame,
   good light, plain background. You own that clip.
2. Upload it on the home page, or register it as a shared sample:
   ```bash
   flask add-sample path/to/your_clip.mp4 --name "Warm-up combo"
   ```
3. Open it, choose a 4–8 s section on **Practise**, then **Record** a take at 0.5×.

If you put your clip in `samples/` and commit it, add a line to `CREDITS.md` saying who is in it and that they
agreed to share it.

## 8. Run the tests

```bash
pytest
```

240 tests, about 35 seconds. Tests that need FFmpeg skip themselves if it is missing; the MediaPipe smoke test
skips if the model has not been downloaded. Fixture videos are generated with FFmpeg test patterns, and pose
detection is replaced by synthetic choreography, so the complete journey test needs no camera or footage.

```bash
python evaluation/run_eval_set.py
```

runs the feedback method on the synthetic evaluation set (see `docs/evaluation.md`).

## Public demo (optional)

The local app is single-user. For a public demo with **prepared sample footage**:

```bash
export MOVEMATCH_ENV=production MOVEMATCH_PUBLIC_DEMO=1
export SECRET_KEY="$(python -c 'import secrets; print(secrets.token_hex(32))')"
flask db upgrade && flask add-sample path/to/your_clip.mp4
```

Then serve it with a production WSGI server behind HTTPS (cameras require HTTPS), with **one** process (the
upload rate limiter and job queue are in-memory), e.g. `gunicorn -w 1 --threads 4 'app:create_app()'`.
In this mode the app refuses to start without a strong `SECRET_KEY` or with debug on, visitors only see the
samples and their own attempts, reference uploads are off, and recordings are deleted after 24 hours. See
`docs/technical_decisions.md` ("Protecting recordings") for what is and is not covered.

## Troubleshooting

| Symptom | Fix |
|---|---|
| "The pose model is not installed" | `python scripts/download_model.py` |
| "FFmpeg is not installed" | install FFmpeg and make sure `ffmpeg -version` works in the same terminal |
| Page says "The database is not ready" | `flask db upgrade`, then restart |
| Camera permission denied | allow the camera in the browser's address bar; use `localhost`, not a LAN IP, unless you have HTTPS |
| "Feedback withheld: your legs were below the bottom edge" | step back or lower the camera: this is a tracking problem, not a score |

## Verification log

2026-10-14, macOS 15 / Apple Silicon, Python 3.11.15, FFmpeg 9.0. In an empty directory containing only the
files Git tracks: `python3.11 -m venv .venv`, `pip install -r requirements-dev.txt` (from PyPI, no PyTorch),
`python scripts/download_model.py` (downloaded from Google, checksum matched), `cp .env.example .env`,
`flask db upgrade`, `pytest` (231 passed), `flask run`, then a 40 s reference uploaded over HTTP and processed
with real MediaPipe (status `completed`; home, video, practice, record and privacy pages returned 200). The
browser journey (practice loop, countdown, recording, upload, replay sync, feedback) was checked in Chrome on
the development copy; the camera was substituted by a canvas stream for that test.
