"""Pose extraction: run MediaPipe Pose Landmarker over a video and cache the landmarks.

Extraction always runs on the constant-frame-rate playback file produced by media.py (for the
reference) or on the CFR transcode of a recording (for an attempt). The skeleton is NOT burned into
a second video any more: the browser draws it from the cached image landmarks on a transparent
canvas, which keeps the music, allows an opacity control, and needs no extra transcode.

MediaPipe is a pretrained pose detector. MoveMatch does not train a model.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from landmark_store import LandmarkRecorder, build_metadata, is_current, read_metadata, save_landmarks

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "pretrained_models" / "pose_landmarker_lite.task"

# MediaPipe Pose Landmarker landmark connections, indexed by the 33 pose landmarks.
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (27, 29), (29, 31),
    (27, 31), (24, 26), (26, 28), (28, 30), (30, 32), (28, 32)
]


class ModelMissingError(FileNotFoundError):
    pass


def require_model(model_path: Path = MODEL_PATH) -> None:
    if not Path(model_path).is_file():
        raise ModelMissingError(
            f"Pose Landmarker model not found at {model_path}. "
            "Run `python scripts/download_model.py` (it saves the model under pretrained_models/ "
            "and verifies its checksum). See docs/setup.md.")


def mediapipe_landmarker(model_path: Path = MODEL_PATH):
    """A MediaPipe PoseLandmarker in VIDEO mode (a context manager with detect_for_video)."""
    import mediapipe as mp

    require_model(model_path)
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp.tasks.vision.PoseLandmarker.create_from_options(options)


def _mediapipe_version() -> str | None:
    try:
        import mediapipe as mp
        return getattr(mp, "__version__", None)
    except ImportError:          # pragma: no cover
        return None


def draw_pose(frame, pose_landmarks):
    """Draw normalized MediaPipe pose landmarks and connections onto a BGR frame (demo scripts)."""
    height, width = frame.shape[:2]
    points = []
    for landmark in pose_landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        points.append((max(0, min(width - 1, x)), max(0, min(height - 1, y))))
    for start, end in POSE_CONNECTIONS:
        if start < len(points) and end < len(points):
            cv2.line(frame, points[start], points[end], (0, 220, 80), 2, cv2.LINE_AA)
    for point in points:
        cv2.circle(frame, point, 3, (0, 100, 255), -1, cv2.LINE_AA)
    return frame


def _open_video(input_path):
    """Open a video with OpenCV; returns (cap, fps, width, height, frame_count). Caller releases cap."""
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError("OpenCV could not open this video.")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or fps > 240:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        raise ValueError("Could not read the video's dimensions.")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    return cap, fps, width, height, frame_count


def extract_landmarks(
    video_path: Path,
    landmarks_path: Path,
    *,
    landmarker_factory: Callable[[], object] | None = None,
    progress: Callable[[float], None] | None = None,
    model_path: Path = MODEL_PATH,
) -> dict:
    """Run pose detection on every frame and save a versioned landmark file.

    Timestamps are the decoder's presentation times (CAP_PROP_POS_MSEC). On the CFR files this is
    used with, they equal frame_index * 1000 / fps; if the decoder reports something unusable
    (missing or not increasing), the frame-index time is used and the metadata says so.
    """
    factory = landmarker_factory or (lambda: mediapipe_landmarker(model_path))
    cap, fps, width, height, frame_count = _open_video(video_path)
    recorder = LandmarkRecorder()
    index_times_used = False
    last_ts = -1.0
    last_mp_ts = -1
    frame_index = 0
    try:
        with factory() as landmarker:
            import mediapipe as mp
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                ts = cap.get(cv2.CAP_PROP_POS_MSEC)
                index_ts = frame_index * 1000.0 / fps
                if not math.isfinite(ts) or ts <= last_ts or abs(ts - index_ts) > 1000.0 / fps * 1.5:
                    ts = index_ts
                    index_times_used = True
                last_ts = ts
                # VIDEO mode needs strictly increasing integer milliseconds.
                mp_ts = max(int(round(ts)), last_mp_ts + 1)
                last_mp_ts = mp_ts
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = landmarker.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), mp_ts)
                brightness = float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())
                recorder.add(ts, result, brightness)
                frame_index += 1
                if progress and frame_count and frame_index % 10 == 0:
                    progress(min(1.0, frame_index / frame_count))
    finally:
        cap.release()

    if frame_index == 0:
        raise ValueError("No readable frames were found in this video.")

    metadata = build_metadata(
        fps=fps, width=width, height=height, num_frames=frame_index,
        model_path=model_path if landmarker_factory is None else None,
        mediapipe_version=_mediapipe_version(),
        timestamp_source=("frame_index * 1000 / fps (decoder times unusable)" if index_times_used
                          else "decoder presentation time of a constant-frame-rate file"),
    )
    sequence = recorder.build(metadata)
    save_landmarks(landmarks_path, sequence)
    if progress:
        progress(1.0)
    return {
        "landmarks_path": str(landmarks_path),
        "frame_count": frame_index,
        "fps": fps,
        "width": width,
        "height": height,
        "detected_frames": recorder.detected_count,
        "detected_ratio": recorder.detected_count / frame_index,
        "mean_brightness": float(np.nanmean(sequence.brightness)),
    }


def precompute_landmarks(input_path: Path, landmarks_path: Path, force: bool = False):
    """Extract and cache landmarks for one file unless a current cache exists (dev tools).

    Skips work if a current cache already exists (see landmark_store.is_current), unless force=True.
    """
    landmarks_path = Path(landmarks_path)
    if not force and is_current(landmarks_path, model_path=MODEL_PATH if MODEL_PATH.is_file() else None):
        meta = read_metadata(landmarks_path)
        return {
            "skipped": True,
            "landmarks_path": str(landmarks_path),
            "frame_count": meta["num_frames"],
            "fps": meta["extraction_fps"],
            "width": meta["width"],
            "height": meta["height"],
            "detected_frames": None,
        }
    require_model()
    info = extract_landmarks(Path(input_path), landmarks_path)
    info["skipped"] = False
    return info
