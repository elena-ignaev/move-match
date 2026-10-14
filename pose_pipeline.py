"""One coordinate convention for the reference dancer and the learner.

Every pose that is ever compared goes through prepare_reference() or prepare_learner()
(prepare_learner_frame() for live frames). Nothing else should normalize poses for
comparison, so the two sides cannot drift apart.

THE CONVENTION (version 1)
--------------------------
1. Source: MediaPipe WORLD landmarks (metres), landmark order and left/right labels as
   MediaPipe reports them.
2. Extraction never flips anything. The reference is extracted from the frames the learner
   is shown; the learner from the frames the camera delivers. Mirroring is applied later,
   here, from explicit flags, never baked into the stored landmark files.
3. Translation: the hip midpoint is the origin.
4. Scale: torso length, as ONE smoothed value per recording (pose_utils.estimate_body_scale).
   Reference and learner use the same NormalizationSettings.
5. Orientation: NOT normalized. Facing direction is part of the dance. Axes stay those of
   the camera, so a dancer facing 30 degrees off-camera stays rotated by 30 degrees.
6. Mirroring has two independent causes, kept as two separate arguments:
     capture_mirrored  a FACT about the data: the learner frames were flipped before
                       extraction (e.g. a canvas drawn with scaleX(-1) and recorded).
     follow_mode       a POLICY about the dance: does the learner copy the screen like a
                       mirror ("mirror") or move the same side as the reference ("same_side")?
   The learner's array is flipped when exactly one of the two is true (flags cancel).

              capture_mirrored   follow_mode   flip learner?
              False              mirror        yes
              False              same_side     no
              True               mirror        no
              True               same_side     yes
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import NamedTuple

import numpy as np

import pose_utils as pu

CONVENTION_VERSION = 1

_LEFT_WRIST, _RIGHT_WRIST = 15, 16      # MediaPipe pose landmark indices

FOLLOW_MIRROR = "mirror"
FOLLOW_SAME_SIDE = "same_side"
FOLLOW_MODES = (FOLLOW_MIRROR, FOLLOW_SAME_SIDE)


@dataclass(frozen=True)
class NormalizationSettings:
    """Everything that must be identical for reference and learner. Store as_dict() with each score."""

    scale: str = "smoothed"          # "smoothed" (rolling window, needs fps) | "global" (one constant)
    window_s: float = 3.0            # smoothing window in seconds (longer than your longest bend)
    percentile: float = 75.0         # high, because bending only shortens the torso
    min_visibility: float = 0.5      # joints below this are treated as missing

    def __post_init__(self) -> None:
        if self.scale not in ("smoothed", "global"):
            raise ValueError(f"Unknown scale mode {self.scale!r}")
        if self.window_s <= 0:
            raise ValueError("window_s must be positive")
        if not 0 < self.percentile <= 100:
            raise ValueError("percentile must be in (0, 100]")
        if not 0 <= self.min_visibility <= 1:
            raise ValueError("min_visibility must be in [0, 1]")

    def as_dict(self) -> dict:
        return {"convention_version": CONVENTION_VERSION, **asdict(self)}


DEFAULT_SETTINGS = NormalizationSettings()


class PreparedPoses(NamedTuple):
    pose: np.ndarray                 # (T, 33, 3) or (33, 3): centred, scaled, mirror-corrected
    visibility: np.ndarray | None    # same leading shape + 33; left/right swapped if the pose was


# --------------------------------------------------------------------- mirroring
def mirror_needed(capture_mirrored: bool, follow_mode: str) -> bool:
    """Single source of truth for whether the learner's array must be flipped (see table above)."""
    if not isinstance(capture_mirrored, (bool, np.bool_)):
        raise ValueError(
            "capture_mirrored must be True or False, not "
            f"{capture_mirrored!r}: never let a missing value silently mean 'not mirrored'"
        )
    if follow_mode not in FOLLOW_MODES:
        raise ValueError(f"follow_mode must be one of {FOLLOW_MODES}, got {follow_mode!r}")
    return bool(capture_mirrored) != (follow_mode == FOLLOW_MIRROR)


def _flip(pose: np.ndarray, vis: np.ndarray | None) -> PreparedPoses:
    return PreparedPoses(pu.mirror_pose(pose), None if vis is None else pu.mirror_visibility(vis))


# ------------------------------------------------------------------- sequences
def _normalize(xyz_seq, vis, fps, settings: NormalizationSettings, causal: bool) -> np.ndarray:
    seq = np.asarray(xyz_seq, dtype=np.float64)
    if vis is not None:
        vis = np.asarray(vis, dtype=np.float64)
        seq = pu.mask_low_visibility(seq, vis, settings.min_visibility)
    if settings.scale == "global":
        window = None                                            # one constant for the recording
    elif fps is None:
        raise ValueError("fps is required for scale='smoothed': the window is defined in seconds, "
                         "and a frame count means something different at 30 and 60 fps")
    else:
        window = pu.window_from_seconds(settings.window_s, fps)
    return pu.normalize_sequence(
        seq,
        align_yaw=False,                                         # convention 5: keep facing direction
        vis=vis,
        window=window,
        percentile=settings.percentile,
        causal=causal,
        min_visibility=settings.min_visibility,
    )


def prepare_reference(
    xyz_seq: np.ndarray,
    vis: np.ndarray | None = None,
    *,
    fps: float | None = None,
    settings: NormalizationSettings = DEFAULT_SETTINGS,
    causal: bool = False,
) -> PreparedPoses:
    """Reference sequence (T, 33, 3) from a landmark cache. Never mirrored."""
    return PreparedPoses(_normalize(xyz_seq, vis, fps, settings, causal),
                         None if vis is None else np.asarray(vis, dtype=np.float64))


def prepare_learner(
    xyz_seq: np.ndarray,
    vis: np.ndarray | None = None,
    *,
    capture_mirrored: bool,
    follow_mode: str,
    fps: float | None = None,
    settings: NormalizationSettings = DEFAULT_SETTINGS,
    causal: bool = False,
) -> PreparedPoses:
    """Learner sequence (T, 33, 3), in the same convention as the reference.

    capture_mirrored and follow_mode are required keywords on purpose: they are decisions,
    and a default would hide them. See the table in the module docstring.
    """
    flip = mirror_needed(capture_mirrored, follow_mode)
    pose = _normalize(xyz_seq, vis, fps, settings, causal)
    vis = None if vis is None else np.asarray(vis, dtype=np.float64)
    return _flip(pose, vis) if flip else PreparedPoses(pose, vis)


def prepare_learner_frame(
    xyz: np.ndarray,
    vis: np.ndarray | None = None,
    *,
    scale: float,
    capture_mirrored: bool,
    follow_mode: str,
    settings: NormalizationSettings = DEFAULT_SETTINGS,
) -> PreparedPoses:
    """One live learner frame (33, 3). `scale` comes from pose_utils.StreamingBodyScale."""
    flip = mirror_needed(capture_mirrored, follow_mode)
    xyz = np.asarray(xyz, dtype=np.float64)
    if vis is not None:
        vis = np.asarray(vis, dtype=np.float64)
        xyz = pu.mask_low_visibility(xyz, vis, settings.min_visibility)
    pose = pu.normalize_pose(xyz, align_yaw=False, scale=scale)
    return _flip(pose, vis) if flip else PreparedPoses(pose, vis)


# ------------------------------------------------------------------ diagnostics
def detect_capture_mirroring(
    image_landmarks: np.ndarray, margin: float = 0.05, percentile: float = 10.0
) -> bool | None:
    """Calibration: was this recording mirrored? Record a clip in which you raise your
    anatomical RIGHT hand (keep the left one down), extract it, and pass its image_landmarks.

    Returns False if MediaPipe's right wrist (16) went highest (labels are correct),
    True if the left wrist (15) did (the recording is mirrored), None if the clip does not
    decide it (both or neither hand raised, or no data). Image y grows downward, so the
    raised hand has the smaller y; `margin` is in normalized image units.
    """
    y = np.asarray(image_landmarks, dtype=np.float64)[..., 1]
    if y.ndim != 2:
        raise ValueError(f"Expected image landmarks of shape (T, 33, >=2), got {np.shape(image_landmarks)}")
    left, right = y[:, _LEFT_WRIST], y[:, _RIGHT_WRIST]
    if not (np.isfinite(left).any() and np.isfinite(right).any()):
        return None
    left_top = np.nanpercentile(left, percentile)       # how high each wrist got
    right_top = np.nanpercentile(right, percentile)
    if left_top - right_top > margin:
        return False
    if right_top - left_top > margin:
        return True
    return None


def facing_angle_deg(pose_seq: np.ndarray, n_frames: int | None = None) -> float:
    """Circular-mean facing angle (degrees) of prepared poses; 0 = squarely facing the camera.

    Use the first second or two (n_frames): over a whole dance with turns an average is
    meaningless. NaN if no frame has a usable shoulder/hip axis.
    """
    seq = np.asarray(pose_seq, dtype=np.float64)
    if seq.ndim == 2:
        seq = seq[None]
    if n_frames:
        seq = seq[:n_frames]
    angles = np.array([pu.yaw_angle(frame) for frame in seq])
    angles = angles[np.isfinite(angles)]
    if angles.size == 0:
        return float("nan")
    return float(np.degrees(np.arctan2(np.sin(angles).mean(), np.cos(angles).mean())))


def facing_difference_deg(reference_pose: np.ndarray, learner_pose: np.ndarray,
                          n_frames: int | None = None) -> float:
    """learner - reference facing angle in degrees, wrapped to [-180, 180).

    Facing direction is deliberately NOT normalized away, so a large value here means the
    pose score will be pulled down by camera placement or stance, not only by the moves.
    Use it to warn the learner; it never changes the data.
    """
    diff = facing_angle_deg(learner_pose, n_frames) - facing_angle_deg(reference_pose, n_frames)
    return float((diff + 180.0) % 360.0 - 180.0)