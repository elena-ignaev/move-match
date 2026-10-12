"""Shared pose utilities for the dance app.

Pure NumPy: no MediaPipe, OpenCV or Flask imports, so it can be used by
offline processing, live feedback and evaluation, and is easy to unit test.

Conventions
-----------
* A pose is an array of shape (33, 3) holding x, y, z in MediaPipe order.
  Use MediaPipe *world* landmarks (result.pose_world_landmarks) for scoring.
  Image-space landmarks (result.pose_landmarks) are for drawing only.
* Visibility is an array of shape (33,) with values in [0, 1].
* "Normalized" means: hip-centred, scaled so the torso length
  (hip midpoint -> shoulder midpoint) is 1.0, optionally yaw-aligned.
  A distance of 1.0 therefore means "average joint error of one torso length".
"""

from __future__ import annotations

import numpy as np

NUM_LANDMARKS = 33

NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_HIP, R_HIP = 23, 24

# (left, right) index pairs, used to mirror a pose.
_MIRROR_PAIRS = [
    (1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16),
    (17, 18), (19, 20), (21, 22), (23, 24), (25, 26), (27, 28), (29, 30),
    (31, 32),
]
MIRROR_INDEX = np.arange(NUM_LANDMARKS)
for _l, _r in _MIRROR_PAIRS:
    MIRROR_INDEX[_l], MIRROR_INDEX[_r] = _r, _l


def _default_joint_weights() -> np.ndarray:
    w = np.ones(NUM_LANDMARKS)
    w[0] = 0.5                                  # nose
    w[1:11] = 0.0                               # eyes, ears, mouth: noisy, rarely choreography
    w[[17, 18, 19, 20, 21, 22]] = 0.5           # fingers
    w[[29, 30, 31, 32]] = 0.5                   # heels and foot index
    return w


DEFAULT_JOINT_WEIGHTS = _default_joint_weights()


# --------------------------------------------------------------------------
# Conversion
# --------------------------------------------------------------------------
def landmarks_to_array(landmarks) -> np.ndarray:
    """Convert a MediaPipe landmark list to an array of shape (33, 4).

    Columns are x, y, z, visibility. Works with any object exposing
    .x .y .z (and optionally .visibility), so no MediaPipe import is needed.
    Missing visibility is treated as 1.0.
    """
    rows = []
    for lm in landmarks:
        vis = getattr(lm, "visibility", None)
        rows.append([lm.x, lm.y, lm.z, 1.0 if vis is None else vis])
    arr = np.asarray(rows, dtype=np.float64)
    if arr.shape != (NUM_LANDMARKS, 4):
        raise ValueError(f"Expected {NUM_LANDMARKS} landmarks, got {arr.shape[0]}")
    return arr


# --------------------------------------------------------------------------
# Rotation
# --------------------------------------------------------------------------
def yaw_angle(xyz: np.ndarray) -> float:
    """Angle (radians) of the body's left-right axis in the horizontal x-z plane.

    0 for a person facing the camera squarely. Returns nan if the axis is degenerate.
    """
    v = (xyz[L_SHOULDER] - xyz[R_SHOULDER]) + (xyz[L_HIP] - xyz[R_HIP])
    if np.hypot(v[0], v[2]) < 1e-6:
        return float("nan")
    return float(np.arctan2(v[2], v[0]))


def rotate_yaw(xyz: np.ndarray, theta: float) -> np.ndarray:
    """Rotate points about the vertical (y) axis by theta radians."""
    c, s = np.cos(theta), np.sin(theta)
    out = np.array(xyz, dtype=np.float64, copy=True)
    out[..., 0] = xyz[..., 0] * c + xyz[..., 2] * s
    out[..., 2] = -xyz[..., 0] * s + xyz[..., 2] * c
    return out


# --------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------
def normalize_pose(xyz: np.ndarray, align_yaw: bool = False, eps: float = 1e-6) -> np.ndarray:
    """Hip-centre, scale by torso length, and optionally align yaw.

    Returns an all-NaN (33, 3) array if the torso cannot be measured, which
    pose_distance() treats as "no usable information".

    align_yaw: per-frame yaw alignment removes camera-angle differences but also
    erases *intentional* turns in the choreography. Leave it off for dances with
    turns, or align with one global angle instead (see yaw_angle / rotate_yaw).
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.shape != (NUM_LANDMARKS, 3):
        raise ValueError(f"Expected shape (33, 3), got {xyz.shape}")

    hip_mid = (xyz[L_HIP] + xyz[R_HIP]) / 2.0
    shoulder_mid = (xyz[L_SHOULDER] + xyz[R_SHOULDER]) / 2.0
    torso = np.linalg.norm(shoulder_mid - hip_mid)
    if not np.isfinite(torso) or torso < eps:
        return np.full((NUM_LANDMARKS, 3), np.nan)

    out = (xyz - hip_mid) / torso
    if align_yaw:
        angle = yaw_angle(out)
        if np.isfinite(angle):
            out = rotate_yaw(out, angle)
    return out


def normalize_sequence(xyz_seq: np.ndarray, align_yaw: bool = False) -> np.ndarray:
    """Apply normalize_pose to every frame of a (T, 33, 3) array."""
    xyz_seq = np.asarray(xyz_seq, dtype=np.float64)
    return np.stack([normalize_pose(f, align_yaw=align_yaw) for f in xyz_seq])


# --------------------------------------------------------------------------
# Mirroring
# --------------------------------------------------------------------------
def mirror_pose(xyz: np.ndarray) -> np.ndarray:
    """Mirror a *normalized* pose: swap left/right landmarks and negate x.

    Apply after normalize_pose (the pose must be centred on the hips).
    Mirroring twice returns the original pose.
    """
    out = np.asarray(xyz, dtype=np.float64)[..., MIRROR_INDEX, :].copy()
    out[..., 0] *= -1.0
    return out


def mirror_visibility(vis: np.ndarray) -> np.ndarray:
    """Swap left/right entries of a visibility array to match mirror_pose."""
    return np.asarray(vis)[..., MIRROR_INDEX]


# --------------------------------------------------------------------------
# Distance
# --------------------------------------------------------------------------
def pose_distance(
    a: np.ndarray,
    b: np.ndarray,
    vis_a: np.ndarray | None = None,
    vis_b: np.ndarray | None = None,
    joint_weights: np.ndarray | None = None,
) -> tuple[float, float]:
    """Weighted mean joint distance between two normalized poses.

    Each joint is weighted by joint_weights * min(vis_a, vis_b), so poorly
    tracked joints contribute little instead of being counted as errors.

    Returns (distance, tracking_confidence):
      distance            mean joint error in torso lengths (nan if nothing usable)
      tracking_confidence fraction of the intended joint weight that was visible, 0..1
    """
    jw = DEFAULT_JOINT_WEIGHTS if joint_weights is None else np.asarray(joint_weights, float)

    d = np.linalg.norm(np.asarray(a, float) - np.asarray(b, float), axis=-1)  # (33,)
    va = np.ones(NUM_LANDMARKS) if vis_a is None else np.clip(np.asarray(vis_a, float), 0, 1)
    vb = np.ones(NUM_LANDMARKS) if vis_b is None else np.clip(np.asarray(vis_b, float), 0, 1)

    w = jw * np.minimum(va, vb)
    valid = np.isfinite(d)
    w = np.where(valid, w, 0.0)
    d = np.where(valid, d, 0.0)

    total = w.sum()
    full = jw.sum()
    confidence = float(total / full) if full > 0 else 0.0
    if total < 1e-9:
        return float("nan"), 0.0
    return float((w * d).sum() / total), confidence


# --------------------------------------------------------------------------
# Missing points
# --------------------------------------------------------------------------
def mask_low_visibility(
    xyz: np.ndarray, vis: np.ndarray, min_visibility: float = 0.5
) -> np.ndarray:
    """Return a copy of xyz (..., 33, 3) with joints below min_visibility set to NaN.

    Use this *before* normalize_pose so an unreliable hip or shoulder cannot
    distort the torso scale; normalize_pose then returns an all-NaN pose.
    """
    out = np.array(xyz, dtype=np.float64, copy=True)
    out[np.asarray(vis, dtype=float) < min_visibility] = np.nan
    return out


def interpolate_missing(
    xyz_seq: np.ndarray,
    timestamps: np.ndarray | None = None,
    max_gap: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Linearly fill short NaN gaps in a (T, 33, 3) sequence, per joint and axis.

    Only gaps of at most `max_gap` frames that have valid data on both sides are
    filled; longer gaps and leading/trailing gaps stay NaN (never extrapolated).
    Apply to raw world landmarks *before* normalize_sequence.

    Returns (filled_sequence, was_filled) where was_filled has shape (T, 33).
    Treat filled points as lower confidence (e.g. cap their visibility) rather
    than as real detections.
    """
    seq = np.asarray(xyz_seq, dtype=np.float64)
    if seq.ndim != 3 or seq.shape[1:] != (NUM_LANDMARKS, 3):
        raise ValueError(f"Expected shape (T, 33, 3), got {seq.shape}")
    T = seq.shape[0]
    t = np.arange(T, dtype=np.float64) if timestamps is None else np.asarray(timestamps, float)
    if t.shape != (T,):
        raise ValueError("timestamps must have shape (T,)")

    out = seq.copy()
    filled = np.zeros((T, NUM_LANDMARKS), dtype=bool)
    for j in range(NUM_LANDMARKS):
        for c in range(3):
            idx = np.flatnonzero(~np.isfinite(seq[:, j, c]))
            if idx.size == 0:
                continue
            for run in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
                s, e = run[0], run[-1]
                if s == 0 or e == T - 1 or run.size > max_gap:
                    continue
                out[run, j, c] = np.interp(
                    t[run], [t[s - 1], t[e + 1]], [seq[s - 1, j, c], seq[e + 1, j, c]]
                )
                filled[run, j] = True
    return out, filled


# --------------------------------------------------------------------------
# Joint angles
# --------------------------------------------------------------------------
# name -> (point_a, vertex_b, point_c); the angle is measured at b, in degrees.
# Left/right entries are adjacent so ANGLE_MIRROR_INDEX is a simple pair swap.
JOINT_ANGLE_DEFS = {
    "left_elbow": (11, 13, 15),
    "right_elbow": (12, 14, 16),
    "left_shoulder": (13, 11, 23),
    "right_shoulder": (14, 12, 24),
    "left_hip": (11, 23, 25),
    "right_hip": (12, 24, 26),
    "left_knee": (23, 25, 27),
    "right_knee": (24, 26, 28),
    "left_ankle": (25, 27, 31),
    "right_ankle": (26, 28, 32),
}
JOINT_ANGLE_NAMES = list(JOINT_ANGLE_DEFS)
ANGLE_MIRROR_INDEX = np.array([i ^ 1 for i in range(len(JOINT_ANGLE_NAMES))])


def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Angle at vertex b between rays b->a and b->c, in degrees (0..180).

    Works on points of shape (..., 3). Returns NaN for NaN input or a
    zero-length segment. Invariant to translation, scale and rotation.
    """
    ba = np.asarray(a, float) - np.asarray(b, float)
    bc = np.asarray(c, float) - np.asarray(b, float)
    denom = np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.sum(ba * bc, axis=-1) / denom
    ang = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    return np.where(denom < eps, np.nan, ang)


def compute_joint_angles(
    xyz: np.ndarray, vis: np.ndarray | None = None, min_visibility: float = 0.5
) -> np.ndarray:
    """Joint angles in degrees for xyz of shape (..., 33, 3) -> (..., n_angles).

    Column order follows JOINT_ANGLE_NAMES. An angle is NaN if any of its three
    landmarks is NaN or (when vis is given) has visibility below min_visibility.
    """
    xyz = np.asarray(xyz, float)
    cols = []
    for a, b, c in JOINT_ANGLE_DEFS.values():
        ang = joint_angle(xyz[..., a, :], xyz[..., b, :], xyz[..., c, :])
        if vis is not None:
            v = np.asarray(vis, float)
            ok = np.minimum(np.minimum(v[..., a], v[..., b]), v[..., c]) >= min_visibility
            ang = np.where(ok, ang, np.nan)
        cols.append(ang)
    return np.stack(cols, axis=-1)


def mirror_angles(angles: np.ndarray) -> np.ndarray:
    """Swap left/right entries of an angle vector (..., n_angles)."""
    return np.asarray(angles)[..., ANGLE_MIRROR_INDEX]


def angle_distance(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Mean absolute angle difference in degrees over angles valid in both poses.

    Returns (distance, confidence) where confidence is the fraction of angles
    that could be compared; distance is nan when none could.
    """
    diff = np.abs(np.asarray(a, float) - np.asarray(b, float))
    valid = np.isfinite(diff)
    if not valid.any():
        return float("nan"), 0.0
    return float(diff[valid].mean()), float(valid.mean())