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
from collections import deque

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
# Step 1: centering
# --------------------------------------------------------------------------
def _as_pose_array(xyz) -> np.ndarray:
    """Float array whose last two axes are (33 landmarks, 3 coordinates)."""
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.ndim < 2 or xyz.shape[-2:] != (NUM_LANDMARKS, 3):
        raise ValueError(f"Expected shape (33, 3) or (T, 33, 3), got {xyz.shape}")
    return xyz
 
 
def hip_center(xyz: np.ndarray) -> np.ndarray:
    """Midpoint of the left and right hips:  c_t = (p_t,left_hip + p_t,right_hip) / 2.
 
    Input (33, 3) -> output (3,);  input (T, 33, 3) -> output (T, 3).
    NaN if either hip is missing in that frame.
    """
    xyz = _as_pose_array(xyz)
    return (xyz[..., L_HIP, :] + xyz[..., R_HIP, :]) / 2.0
 
 
def center_pose(xyz: np.ndarray) -> np.ndarray:
    """Subtract the hip midpoint from every joint:  p'_t,j = p_t,j - c_t.
 
    Works on one pose (33, 3) or a whole sequence (T, 33, 3) and returns a new array.
    This is a pure translation: distances between joints, limb directions and
    scale are unchanged. Missing data: if a hip is NaN the whole frame becomes NaN
    (there is no origin to subtract); a NaN in any other joint stays confined to
    that joint. Low-visibility joints should be masked beforehand with
    mask_low_visibility().
    """
    xyz = _as_pose_array(xyz)
    return xyz - hip_center(xyz)[..., np.newaxis, :]

# --------------------------------------------------------------------------
# Step 2: scale
# --------------------------------------------------------------------------
# Design (evidence in scale_benchmark.py):
#   * Measure the torso length in every frame, but do NOT divide each frame by its own
#     value: bending shortens the torso, and per-frame jitter would be injected into
#     every joint as fake motion.
#   * Estimate one smooth body scale instead: a rolling HIGH percentile of the valid torso
#     lengths (high, because bending only ever shortens the torso).
#   * The window must be longer than the longest bend you expect (default 3 seconds).
TORSO_JOINTS = (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP)
 
 
def window_from_seconds(seconds: float, fps: float) -> int:
    """Window length in frames for a duration. Odd and >= 3, so a centred window is symmetric.
 
    Specify windows in seconds: a frame count means something different at 30 and 60 fps.
    """
    if seconds <= 0 or fps <= 0:
        raise ValueError("seconds and fps must be positive")
    frames = int(round(seconds * fps))
    return max(3, frames if frames % 2 == 1 else frames + 1)
 
def shoulder_center(xyz: np.ndarray) -> np.ndarray:
    """Midpoint of the left and right shoulders. (33, 3) -> (3,);  (T, 33, 3) -> (T, 3)."""
    xyz = _as_pose_array(xyz)
    return (xyz[..., L_SHOULDER, :] + xyz[..., R_SHOULDER, :]) / 2.0

def torso_length(
    xyz: np.ndarray,
    vis: np.ndarray | None = None,
    min_visibility: float = 0.5,
    eps: float = 1e-6,
) -> np.ndarray:
    """Distance from the hip midpoint to the shoulder midpoint.
 
    Input (33, 3) -> scalar array; input (T, 33, 3) -> (T,). NaN when a torso joint is
    missing, when (given vis) any of the four torso joints is below min_visibility, or when
    the length is ~0. Frames flagged NaN are ignored by estimate_body_scale, so partially
    visible frames do not distort the scale.
    """
    xyz = _as_pose_array(xyz)
    length = np.linalg.norm(shoulder_center(xyz) - hip_center(xyz), axis=-1)
    if vis is not None:
        vis = np.asarray(vis, dtype=np.float64)
        if vis.shape != xyz.shape[:-1]:
            raise ValueError(f"vis has shape {vis.shape}, expected {xyz.shape[:-1]}")
        length = np.where(vis[..., list(TORSO_JOINTS)].min(axis=-1) >= min_visibility, length, np.nan)
    return np.where(length > eps, length, np.nan)
 
def _fill_gaps(x: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaNs, holding the first/last valid value at the edges (uses future frames)."""
    ok = np.isfinite(x)
    if not ok.any():
        return x
    idx = np.arange(len(x))
    return np.interp(idx, idx[ok], x[ok])
 
 
def _ffill(x: np.ndarray) -> np.ndarray:
    """Hold the last valid value forward; leading NaNs stay NaN. Never looks at future frames."""
    out = x.copy()
    last = np.nan
    for i in range(len(out)):
        if np.isfinite(out[i]):
            last = out[i]
        else:
            out[i] = last
    return out
 
 
def _check_scale_params(window, percentile, min_count) -> None:
    if not 0 < percentile <= 100:
        raise ValueError("percentile must be in (0, 100]")
    if window is not None and window < 1:
        raise ValueError("window must be a positive number of frames (or None for one global value)")
    if min_count < 1:
        raise ValueError("min_count must be at least 1")
 
 
def estimate_body_scale(
    lengths: np.ndarray,
    window: int | None = 91,
    percentile: float = 75.0,
    causal: bool = False,
    min_count: int = 5,
) -> np.ndarray:
    """Smooth body-scale estimate for every frame, from per-frame torso lengths.
 
    lengths  (T,) torso length per frame, NaN where unusable (see torso_length).
    window   frames in the rolling window; None gives one constant for the whole recording.
             Use window_from_seconds() to convert from seconds.
    percentile  high on purpose: bending only shortens the torso (75 by default).
    causal   False: window centred on each frame; gaps filled by interpolation (offline,
             e.g. the reference video). True: window is the last `window` frames only; gaps
             hold the last estimate; leading frames stay NaN until min_count valid
             measurements exist (matches StreamingBodyScale, usable for live feedback).
    min_count  valid measurements a window needs before it produces an estimate.
 
    Returns a (T,) float array. Raises ValueError if there is too little valid data, rather
    than silently returning NaNs.
    """
    _check_scale_params(window, percentile, min_count)
    L = np.asarray(lengths, dtype=np.float64)
    if L.ndim != 1:
        raise ValueError(f"lengths must be 1-D, got shape {L.shape}")
    L = np.where(np.isfinite(L) & (L > 0), L, np.nan)
    n_valid = int(np.isfinite(L).sum())
    if n_valid < min_count:
        raise ValueError(f"Only {n_valid} valid torso measurements; need at least {min_count} to estimate a scale")
 
    T = L.size
    if window is None:
        return np.full(T, float(np.nanpercentile(L, percentile)))
 
    out = np.full(T, np.nan)
    half = window // 2
    for t in range(T):
        lo, hi = (max(0, t - window + 1), t + 1) if causal else (max(0, t - half), min(T, t + half + 1))
        w = L[lo:hi]
        w = w[np.isfinite(w)]
        if w.size >= min_count:
            out[t] = np.percentile(w, percentile)
    if not np.isfinite(out).any():
        raise ValueError("No window contained enough valid torso measurements; use a longer window or lower min_count")
    return _ffill(out) if causal else _fill_gaps(out)
 
 
class StreamingBodyScale:
    """Frame-by-frame version of estimate_body_scale(causal=True), for live camera feedback.
 
        scaler = StreamingBodyScale(window=window_from_seconds(3.0, fps))
        s = scaler.update(torso_length(frame_xyz, vis=frame_vis))   # NaN until enough data
    """
 
    def __init__(self, window: int = 91, percentile: float = 75.0, min_count: int = 5) -> None:
        _check_scale_params(window, percentile, min_count)
        self.percentile, self.min_count = percentile, min_count
        self._buffer: deque = deque(maxlen=window)
        self._last = float("nan")
 
    def update(self, length: float) -> float:
        """Add one torso length (NaN if unusable) and return the current scale estimate."""
        length = float(length)
        self._buffer.append(length if np.isfinite(length) and length > 0 else np.nan)
        valid = [v for v in self._buffer if np.isfinite(v)]
        if len(valid) >= self.min_count:
            self._last = float(np.percentile(valid, self.percentile))
        return self._last
 
    def reset(self) -> None:
        self._buffer.clear()
        self._last = float("nan")
 
 
def scale_pose(xyz: np.ndarray, scale, eps: float = 1e-6) -> np.ndarray:
    """Divide poses by a body scale: a scalar, or one value per frame for a (T, 33, 3) array.
 
    Call on already centred poses (center_pose). A missing or ~0 scale gives a NaN frame.
    """
    xyz = _as_pose_array(xyz)
    scale = np.asarray(scale, dtype=np.float64)
    if scale.ndim != 0 and scale.shape != xyz.shape[:-2]:
        raise ValueError(f"scale has shape {scale.shape}, expected a scalar or {xyz.shape[:-2]}")
    safe = np.where(np.isfinite(scale) & (scale > eps), scale, np.nan)
    return xyz / safe[..., np.newaxis, np.newaxis]
 
 
    

# --------------------------------------------------------------------------
# Rotation
# --------------------------------------------------------------------------
def yaw_angle(xyz: np.ndarray) -> float:
    """Angle (radians) of the body's left-right axis in the horizontal x-z plane.
 
    0 for a person facing the camera squarely. Returns nan if the axis is degenerate.
    Takes a single pose of shape (33, 3).
    """
    xyz = _as_pose_array(xyz)
    if xyz.ndim != 2:
        raise ValueError(f"yaw_angle takes a single (33, 3) pose, got {xyz.shape}")
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
 
 
def _align_yaw_frame(pose: np.ndarray) -> np.ndarray:
    """Rotate one pose so its left-right axis lies along +x; unchanged if the axis is degenerate."""
    angle = yaw_angle(pose)
    return rotate_yaw(pose, angle) if np.isfinite(angle) else pose
 


# --------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------
def normalize_pose(xyz: np.ndarray, align_yaw: bool = False, scale=None) -> np.ndarray:
    """Hip-centre, scale and optionally yaw-align ONE pose of shape (33, 3).
 
    Composition of center_pose and scale_pose. Returns an all-NaN pose if the hips or the
    scale are unusable, which pose_distance() treats as "no usable information".
 
    scale: body scale for this frame, e.g. the value from StreamingBodyScale.update() in
    live use. If None, this frame's own torso length is used. That is only a fallback for
    single images and tests: it is jittery (see Step 2 design note). For video, prefer
    normalize_sequence() offline or StreamingBodyScale live.
 
    align_yaw: per-frame yaw alignment removes camera-angle differences but also
    erases *intentional* turns in the choreography. Leave it off for dances with
    turns, or align with one global angle instead (see yaw_angle / rotate_yaw).
    """
    xyz = _as_pose_array(xyz)
    if xyz.ndim != 2:
        raise ValueError(f"normalize_pose takes a single (33, 3) pose, got {xyz.shape}")
    s = torso_length(xyz) if scale is None else scale
    out = scale_pose(center_pose(xyz), s)
    return _align_yaw_frame(out) if align_yaw else out


def normalize_sequence(
    xyz_seq: np.ndarray,
    align_yaw: bool = False,
    vis: np.ndarray | None = None,
    window: int | None = 91,
    percentile: float = 75.0,
    causal: bool = False,
    min_count: int = 5,
    min_visibility: float = 0.5,
) -> np.ndarray:
    """Normalize a (T, 33, 3) sequence: center, estimate one smooth body scale, divide.
 
    Pipeline: center_pose -> torso_length -> estimate_body_scale -> scale_pose, then optional
    per-frame yaw alignment (same caveat as normalize_pose). vis (T, 33) is only used to
    discard unreliable torso measurements; mask other low-visibility joints beforehand with
    mask_low_visibility(). window/percentile/causal/min_count: see estimate_body_scale.
    Raises ValueError if there is too little valid torso data.
    """
    xyz_seq = _as_pose_array(xyz_seq)
    if xyz_seq.ndim != 3:
        raise ValueError(f"normalize_sequence takes (T, 33, 3), got {xyz_seq.shape}")
    lengths = torso_length(xyz_seq, vis=vis, min_visibility=min_visibility)
    scale = estimate_body_scale(lengths, window=window, percentile=percentile,
                                causal=causal, min_count=min_count)
    out = scale_pose(center_pose(xyz_seq), scale)
    if align_yaw:
        out = np.stack([_align_yaw_frame(f) for f in out])
    return out

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

import numpy as np


# --------------------------------------------------------------------------
# Missing points
# --------------------------------------------------------------------------

def mask_low_visibility(
    xyz: np.ndarray,
    vis: np.ndarray,
    min_visibility: float = 0.5,
) -> np.ndarray:
    """
    Replace unreliable or non-finite landmarks with NaN.

    Parameters
    ----------
    xyz:
        Landmark coordinates with shape (..., 33, 3).
        Supports either a single frame (33, 3) or a sequence
        (T, 33, 3).
    vis:
        Visibility scores with shape (..., 33), matching xyz.
    min_visibility:
        Minimum acceptable visibility in [0, 1].

    Returns
    -------
    np.ndarray
        Float64 copy of xyz, with unreliable joints set to NaN.
        A joint is considered valid only if all three coordinates
        are finite and its visibility meets the threshold.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    vis = np.asarray(vis, dtype=np.float64)

    if xyz.ndim < 2 or xyz.shape[-2:] != (NUM_LANDMARKS, 3):
        raise ValueError(
            f"Expected xyz shape (..., {NUM_LANDMARKS}, 3), "
            f"got {xyz.shape}"
        )

    expected_vis_shape = xyz.shape[:-1]
    if vis.shape != expected_vis_shape:
        raise ValueError(
            f"Expected vis shape {expected_vis_shape}, got {vis.shape}"
        )

    if not np.isfinite(min_visibility) or not 0 <= min_visibility <= 1:
        raise ValueError("min_visibility must be between 0 and 1")

    out = xyz.copy()

    # Require all XYZ coordinates to be finite for a joint to be valid.
    valid_xyz = np.all(np.isfinite(out), axis=-1)

    valid_visibility = (
        np.isfinite(vis)
        & (vis >= min_visibility)
    )

    valid = valid_xyz & valid_visibility

    # Set all three coordinates of an invalid joint to NaN.
    out[~valid] = np.nan

    return out


def interpolate_missing(
    xyz_seq: np.ndarray,
    timestamps: np.ndarray | None = None,
    max_gap: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Interpolate short gaps in a sequence of 3D landmarks.

    Only missing joint observations bounded by valid observations
    on both sides are filled. Gaps longer than max_gap, as well as
    leading and trailing gaps, remain missing.

    A joint observation is valid only when all three XYZ coordinates
    are finite.

    Parameters
    ----------
    xyz_seq:
        Array with shape (T, NUM_LANDMARKS, 3).
    timestamps:
        Optional frame timestamps with shape (T,), strictly increasing.
        If omitted, frame indices are used.
    max_gap:
        Maximum number of consecutive missing frames to fill.

    Returns
    -------
    filled_sequence:
        Copy of the sequence with eligible gaps interpolated.
    was_filled:
        Boolean array of shape (T, NUM_LANDMARKS).
        True indicates that the joint was interpolated.
    """
    seq = np.asarray(xyz_seq, dtype=np.float64)

    if seq.ndim != 3 or seq.shape[1:] != (NUM_LANDMARKS, 3):
        raise ValueError(
            f"Expected shape (T, {NUM_LANDMARKS}, 3), got {seq.shape}"
        )

    T = seq.shape[0]

    if not isinstance(max_gap, (int, np.integer)) or isinstance(
        max_gap, (bool, np.bool_)
    ):
        raise TypeError("max_gap must be a non-negative integer")

    if max_gap < 0:
        raise ValueError("max_gap must be non-negative")

    if timestamps is None:
        t = np.arange(T, dtype=np.float64)
    else:
        t = np.asarray(timestamps, dtype=np.float64)

        if t.shape != (T,):
            raise ValueError(f"Expected timestamps shape ({T},), got {t.shape}")

        if not np.all(np.isfinite(t)):
            raise ValueError("timestamps must all be finite")

        if T > 1 and np.any(np.diff(t) <= 0):
            raise ValueError("timestamps must be strictly increasing")

    out = seq.copy()
    was_filled = np.zeros((T, NUM_LANDMARKS), dtype=bool)

    if T == 0 or max_gap == 0:
        return out, was_filled

    # A joint observation is valid only when the complete XYZ vector
    # is finite. This avoids mixing partially missing coordinates.
    joint_valid = np.all(np.isfinite(seq), axis=-1)

    for j in range(NUM_LANDMARKS):
        valid_indices = np.flatnonzero(joint_valid[:, j])

        if valid_indices.size < 2:
            continue

        # Examine gaps between consecutive valid observations.
        for left, right in zip(valid_indices[:-1], valid_indices[1:]):
            gap_length = right - left - 1

            if gap_length <= 0 or gap_length > max_gap:
                continue

            # Interpolate only internal gaps, never extrapolate.
            missing_indices = np.arange(left + 1, right)

            alpha = (
                (t[missing_indices] - t[left])
                / (t[right] - t[left])
            )

            out[missing_indices, j, :] = (
                seq[left, j, :][None, :] * (1.0 - alpha[:, None])
                + seq[right, j, :][None, :] * alpha[:, None]
            )

            was_filled[missing_indices, j] = True

    return out, was_filled


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


def trunk_inclination(
    xyz: np.ndarray,
    vis: np.ndarray | None = None,
    min_visibility: float = 0.5,
    up: tuple[float, float, float] = (0.0, -1.0, 0.0),
) -> np.ndarray:
    """Angle in degrees (0..180) between the trunk (hip midpoint -> shoulder midpoint) and "up".
 
    0 = upright, 90 = horizontal. xyz (..., 33, 3) -> (...,). NaN when the torso cannot be
    measured (see torso_length). Not part of JOINT_ANGLE_DEFS: it has no left/right
    partner and is not mirrored (a forward lean is a forward lean in a mirror).
 
    up: the world "up" direction. MediaPipe world landmarks have y pointing DOWN, so the
    default is (0, -1, 0). Check on a standing frame: shoulder_center - hip_center should have
    a negative y. Unlike joint_angle this is measured against a fixed axis, so it is not
    rotation-invariant: camera tilt changes it (yaw about y does not).
    """
    xyz = _as_pose_array(xyz)
    up = np.asarray(up, dtype=np.float64)
    up = up / np.linalg.norm(up)
    length = torso_length(xyz, vis=vis, min_visibility=min_visibility)
    trunk = shoulder_center(xyz) - hip_center(xyz)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = (trunk @ up) / length
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


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