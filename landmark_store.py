"""Versioned on-disk format for per-frame pose landmarks (.npz).

Arrays stored in the file
-------------------------
timestamps       (T,)     float64  frame time in milliseconds on the playback timeline. Extraction runs on
                                   the constant-frame-rate transcode the browser plays (media.py), so this
                                   equals frame_index * 1000 / fps AND the browser's currentTime * 1000.
landmarks        (T,33,3) float32  MediaPipe *world* landmarks (metres, hip-centred, y down), raw
visibility       (T,33)   float32  per-landmark visibility in [0, 1] (0 where undetected)
detected         (T,)     bool     False where MediaPipe found no pose (landmarks are NaN)
image_landmarks  (T,33,3) float32  normalized image coordinates: overlays and in-frame checks
brightness       (T,)     float32  mean luma of the frame, 0..255 (NaN if not measured); used to explain
                                   poor tracking ("the room was dark"), never to score
metadata_json    ()       str      JSON metadata (see build_metadata)

Landmarks are stored RAW. Normalization (hip-centring, torso scaling, mirroring)
happens at evaluation time via pose_utils, so changing it never invalidates the cache.

The file contains only plain numeric/string arrays, so it is always loaded with
allow_pickle=False.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pose_utils import landmarks_to_array

SCHEMA_VERSION = 2            # bump when the file layout changes (2: brightness array)
EXTRACTION_VERSION = "2"      # bump when extraction logic changes (2: run on the CFR playback transcode)
LANDMARK_CONVENTION = "mediapipe_pose_33_world"
NUM_LANDMARKS = 33


class LandmarkSchemaError(ValueError):
    """The file is not a landmark file this code version understands."""


@dataclass
class LandmarkSequence:
    timestamps: np.ndarray
    landmarks: np.ndarray
    visibility: np.ndarray
    detected: np.ndarray
    image_landmarks: np.ndarray
    metadata: dict = field(default_factory=dict)
    brightness: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.brightness is None:
            self.brightness = np.full(self.timestamps.shape[0], np.nan, dtype=np.float32)

    @property
    def num_frames(self) -> int:
        return int(self.timestamps.shape[0])

    def validate(self) -> None:
        t = self.num_frames
        expected = {
            "timestamps": (t,),
            "landmarks": (t, NUM_LANDMARKS, 3),
            "visibility": (t, NUM_LANDMARKS),
            "detected": (t,),
            "image_landmarks": (t, NUM_LANDMARKS, 3),
            "brightness": (t,),
        }
        for name, shape in expected.items():
            actual = getattr(self, name).shape
            if actual != shape:
                raise LandmarkSchemaError(f"{name} has shape {actual}, expected {shape}")
        if t > 1 and not np.all(np.diff(self.timestamps) > 0):
            raise LandmarkSchemaError("timestamps must be strictly increasing")


def file_sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def build_metadata(
    *,
    fps: float,
    width: int,
    height: int,
    num_frames: int,
    model_path: str | Path | None = None,
    mediapipe_version: str | None = None,
    timestamp_source: str = "frame_index * 1000 / fps",
    extra: dict | None = None,
) -> dict:
    meta = {
        "schema_version": SCHEMA_VERSION,
        "extraction_version": EXTRACTION_VERSION,
        "landmark_convention": LANDMARK_CONVENTION,
        "coordinate_units": "metres",
        "coordinate_origin": "hip midpoint (MediaPipe world), y axis points down",
        "timestamp_unit": "ms",
        "timestamp_source": timestamp_source,
        "extraction_fps": float(fps),
        "width": int(width),
        "height": int(height),
        "num_frames": int(num_frames),
        "normalization": "none (raw; normalize with pose_utils at evaluation time)",
        "model_file": Path(model_path).name if model_path else None,
        "model_sha256": file_sha256(model_path) if model_path and Path(model_path).is_file() else None,
        "mediapipe_version": mediapipe_version,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        meta.update(extra)
    return meta


def save_landmarks(path: str | Path, seq: LandmarkSequence) -> Path:
    """Validate and write atomically (temp file, then rename)."""
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError("Landmark files must use the .npz extension")
    seq.validate()
    meta = dict(seq.metadata)
    meta.setdefault("schema_version", SCHEMA_VERSION)
    meta["num_frames"] = seq.num_frames

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(
        tmp,
        timestamps=seq.timestamps.astype(np.float64),
        landmarks=seq.landmarks.astype(np.float32),
        visibility=seq.visibility.astype(np.float32),
        detected=seq.detected.astype(bool),
        image_landmarks=seq.image_landmarks.astype(np.float32),
        brightness=np.asarray(seq.brightness, dtype=np.float32),
        metadata_json=np.array(json.dumps(meta)),
    )
    os.replace(tmp, path)
    return path


def read_metadata(path: str | Path) -> dict:
    """Read only the metadata (cheap; does not load the arrays)."""
    with np.load(path, allow_pickle=False) as data:
        if "metadata_json" not in data.files:
            raise LandmarkSchemaError("Missing metadata_json")
        return json.loads(str(data["metadata_json"]))


def load_landmarks(path: str | Path) -> LandmarkSequence:
    with np.load(path, allow_pickle=False) as data:
        if "metadata_json" in data.files:
            version = json.loads(str(data["metadata_json"])).get("schema_version")
            if version != SCHEMA_VERSION:
                raise LandmarkSchemaError(
                    f"Schema version {version} != supported {SCHEMA_VERSION}; regenerate the landmarks.")
        missing = {"timestamps", "landmarks", "visibility", "detected",
                   "image_landmarks", "brightness", "metadata_json"} - set(data.files)
        if missing:
            raise LandmarkSchemaError(f"Missing arrays: {sorted(missing)}")
        meta = json.loads(str(data["metadata_json"]))
        if meta.get("schema_version") != SCHEMA_VERSION:
            raise LandmarkSchemaError(
                f"Schema version {meta.get('schema_version')} != supported {SCHEMA_VERSION}; "
                "regenerate the landmarks."
            )
        seq = LandmarkSequence(
            timestamps=data["timestamps"],
            landmarks=data["landmarks"],
            visibility=data["visibility"],
            detected=data["detected"],
            image_landmarks=data["image_landmarks"],
            metadata=meta,
            brightness=data["brightness"],
        )
    seq.validate()
    return seq


def is_current(path: str | Path, model_path: str | Path | None = None) -> bool:
    """True if the cache exists and matches this code's schema, extraction version and model.

    Use before reusing a cache; if False, re-run extraction.
    """
    path = Path(path)
    if not path.is_file():
        return False
    try:
        meta = read_metadata(path)
    except Exception:
        return False
    if meta.get("schema_version") != SCHEMA_VERSION:
        return False
    if meta.get("extraction_version") != EXTRACTION_VERSION:
        return False
    if model_path is not None and Path(model_path).is_file():
        if meta.get("model_sha256") != file_sha256(model_path):
            return False
    return True


# --------------------------------------------------------------------------
# Recording during extraction
# --------------------------------------------------------------------------
def frame_arrays_from_result(result):
    """Convert one MediaPipe PoseLandmarkerResult (or None) to per-frame arrays.

    Returns (world (33,3), image (33,3), visibility (33,), detected bool).
    Undetected frames give NaN coordinates, zero visibility and detected=False.
    Visibility is read from the image landmarks, where it is reliably populated.
    """
    world = np.full((NUM_LANDMARKS, 3), np.nan)
    image = np.full((NUM_LANDMARKS, 3), np.nan)
    vis = np.zeros(NUM_LANDMARKS)
    detected = False
    if result is not None and result.pose_landmarks and result.pose_world_landmarks:
        img_arr = landmarks_to_array(result.pose_landmarks[0])
        world_arr = landmarks_to_array(result.pose_world_landmarks[0])
        image, vis, world = img_arr[:, :3], img_arr[:, 3], world_arr[:, :3]
        detected = True
    return world, image, vis, detected


class LandmarkRecorder:
    """Collects per-frame results during extraction and builds a LandmarkSequence."""

    def __init__(self) -> None:
        self._ts, self._world, self._image, self._vis, self._det = [], [], [], [], []
        self._brightness = []

    def add(self, timestamp_ms: float, result, brightness: float = float("nan")) -> bool:
        """Record one frame; returns whether a pose was detected."""
        world, image, vis, detected = frame_arrays_from_result(result)
        self._brightness.append(float(brightness))
        self._ts.append(float(timestamp_ms))
        self._world.append(world)
        self._image.append(image)
        self._vis.append(vis)
        self._det.append(detected)
        return detected

    def __len__(self) -> int:
        return len(self._ts)

    @property
    def detected_count(self) -> int:
        return int(sum(self._det))

    def build(self, metadata: dict) -> LandmarkSequence:
        if not self._ts:
            raise ValueError("No frames were recorded.")
        return LandmarkSequence(
            timestamps=np.asarray(self._ts, dtype=np.float64),
            landmarks=np.stack(self._world),
            visibility=np.stack(self._vis),
            detected=np.asarray(self._det, dtype=bool),
            image_landmarks=np.stack(self._image),
            metadata=metadata,
            brightness=np.asarray(self._brightness, dtype=np.float32),
        )