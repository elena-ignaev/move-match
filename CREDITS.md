# Credits

## Model

* **MediaPipe Pose Landmarker (lite), float16, version 1** by Google. Apache License 2.0.
  Downloaded by `scripts/download_model.py` from Google's official model storage; not redistributed here.
  Model card: <https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker>.
  MoveMatch uses this pretrained model as-is. It does not train or fine-tune any model.

## Software

| Package | License |
|---|---|
| MediaPipe | Apache 2.0 |
| OpenCV (opencv-contrib-python) | Apache 2.0 |
| NumPy | BSD-3-Clause |
| Flask, Werkzeug, Jinja2 | BSD-3-Clause |
| Flask-SQLAlchemy, SQLAlchemy, Alembic, Flask-Migrate | MIT |
| Flask-WTF | BSD-3-Clause |
| python-dotenv | BSD-3-Clause |
| pytest | MIT |
| matplotlib (research plots) | Matplotlib License (PSF-style) |
| FFmpeg (system program, not bundled) | LGPL 2.1+ / GPL depending on build |

## Media

* No third-party dance footage is included. Test videos are generated at test time from FFmpeg's built-in
  `testsrc2` pattern and `sine` tone.
* `assets/prototype/` contains screenshots of the first prototype, showing footage from the author's
  own test upload.
* If you add a sample clip under `samples/`, list here who appears in it and confirm they agreed to share it.
