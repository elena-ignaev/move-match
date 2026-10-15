"""Explainable practice feedback: tracking check, timing offset, posture at key moments.

This is deliberately NOT a single "dance accuracy" percentage. There is no evidence yet that such a
number would mean the same thing to a learner as it does to a teacher (see docs/evaluation.md).
Instead each attempt gets three separate, checkable answers:

1. Could we see you?  (tracking)
   Per body region, the share of frames in which its joints were detected, confident and INSIDE
   the camera frame. If the torso was not visible, or nobody was found, feedback is WITHHELD with
   the reason ("your legs were below the bottom edge of the frame"), rather than given as a low
   score. If only the legs were missing, leg feedback is withheld and arm feedback is still given.

2. Were you early or late?  (timing, separated from posture)
   Each learner frame already has the reference time that was on screen (sync_timeline.py). We try
   constant delays L between -0.6 s and +1.0 s and keep the one at which the learner's joint angles
   best match the reference's: "you were about 0.3 s behind". If the cost curve is flat (little
   movement, or noisy tracking) we say the timing could not be measured instead of guessing.

3. Did the shape match at the moments that matter?  (posture, with the timing removed)
   Key moments are picked from the REFERENCE alone: instants where the reference dancer's joints
   are momentarily still (a hit, a held pose), spread over the section. Because they depend only
   on the reference and the section, every attempt at the same section is judged at the same
   moments, so attempts can be compared moment by moment. At each moment we compare joint angles
   (elbows, shoulders, hips, knees) in 3-D world coordinates, after shifting by the measured delay
   and allowing +/-0.2 s of music for small timing jitter. Angles do not depend on where the
   learner stands, how far from the camera, or how tall they are.

   Joint angles cannot see facing direction (a turn changes no elbow angle), and orientation is
   deliberately NOT normalized away (pose_pipeline.py, convention 5). Facing is therefore checked
   separately at each moment, so a missed turn is reported instead of hidden.

Thresholds are provisional and stated in FeedbackSettings; evaluation/run_eval_set.py shows how
they behave on correct, wrong, late, mirrored and missing-body cases, including where they fail.
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass

import numpy as np

import pose_pipeline as pp
import pose_utils as pu

ALGORITHM_VERSION = "1.1"   # 1.1: a flag needs the 3-D AND the 2-D picture angle to differ

FEEDBACK_ANGLES = ("left_elbow", "right_elbow", "left_shoulder", "right_shoulder",
                   "left_hip", "right_hip", "left_knee", "right_knee")
_ANGLE_COL = {name: pu.JOINT_ANGLE_NAMES.index(name) for name in FEEDBACK_ANGLES}
# Body regions checked for visibility. An angle is only compared when the learner's region that
# contains it was visible often enough (the region of the learner's OWN side: see evaluate_attempt).
REGIONS = {
    "torso": (11, 12, 23, 24),
    "left_arm": (13, 15),
    "right_arm": (14, 16),
    "left_leg": (25, 27),
    "right_leg": (26, 28),
}
ANGLE_REGION = {
    "left_elbow": "left_arm", "left_shoulder": "left_arm", "right_elbow": "right_arm", "right_shoulder": "right_arm",
    "left_hip": "left_leg", "left_knee": "left_leg", "right_hip": "right_leg", "right_knee": "right_leg",
}


@dataclass(frozen=True)
class FeedbackSettings:
    min_visibility: float = 0.5        # below this a joint counts as not seen
    frame_margin: float = 0.02         # image coords outside [-margin, 1 + margin] = outside the frame
    close_deg: float = 20.0            # every compared joint within this: "close"
    review_deg: float = 35.0           # any joint beyond this: "worth reviewing"
    lag_min_s: float = -0.6            # timing search range, wall-clock seconds (+ = learner late)
    lag_max_s: float = 1.0
    on_time_s: float = 0.15            # |delay| below this is reported as on time
    min_lag_contrast_deg: float = 2.0  # cost-curve depth needed to trust the delay
    moment_window_s: float = 0.2       # posture tolerance around each moment, seconds of music
    min_moment_gap_s: float = 1.0
    max_moments: int = 8
    edge_s: float = 0.3                # ignore the first/last 0.3 s of the section
    facing_review_deg: float = 60.0    # facing differs by more: flag the moment
    min_mapped_s: float = 2.0          # need at least this much recording overlapping playback
    min_detected: float = 0.5          # fraction of frames with a person found
    region_ok: float = 0.6             # region visible in at least this share of frames
    region_good: float = 0.85
    dark_luma: float = 50.0            # mean frame brightness (0-255) below this: "dark"
    min_joints_per_moment: int = 3

    def as_dict(self) -> dict:
        return asdict(self)


DEFAULT_SETTINGS = FeedbackSettings()


# --------------------------------------------------------------------------- helpers
def _visible_mask(image: np.ndarray, vis: np.ndarray, s: FeedbackSettings) -> np.ndarray:
    """(T, 33) True where a joint was confidently detected inside the camera frame."""
    x, y = image[..., 0], image[..., 1]
    with np.errstate(invalid="ignore"):
        inside = (x >= -s.frame_margin) & (x <= 1 + s.frame_margin) & \
                 (y >= -s.frame_margin) & (y <= 1 + s.frame_margin)
        return np.isfinite(x) & np.isfinite(y) & inside & (np.nan_to_num(vis) >= s.min_visibility)


def _angles(world: np.ndarray, visible: np.ndarray) -> np.ndarray:
    """(T, 10) joint angles in degrees, NaN where any of the three joints was not visible."""
    return pu.compute_joint_angles(world, vis=visible.astype(float), min_visibility=0.5)


def _image_angles(seq, visible: np.ndarray) -> np.ndarray:
    """(T, 10) joint angles measured in the 2-D picture (pixels, so x and y have the same scale).

    Depth (z) from one camera is a guess, and slowly drifting depth errors bend 3-D angles by tens of
    degrees (evaluation/cases.py, depth_noise_*). The picture itself has no depth error, but it does
    shorten limbs that point at the camera. Neither view is trusted alone: see joint_status().
    """
    w = float(seq.metadata.get("width") or 16)
    h = float(seq.metadata.get("height") or 9)
    img = np.asarray(seq.image_landmarks, dtype=np.float64)
    flat = np.stack([img[..., 0] * w, img[..., 1] * h, np.zeros(img.shape[:-1])], axis=-1)
    return pu.compute_joint_angles(flat, vis=visible.astype(float), min_visibility=0.5)


def joint_status(d3: float, d2: float, s: "FeedbackSettings") -> str:
    """close / minor / review for a 3-D difference d3 and a picture-plane difference d2 (degrees).

    'review' needs BOTH views to disagree with the reference, in the same direction: a large 3-D
    difference that the picture does not show (or contradicts) is more likely a depth-estimation
    error than a real mistake, so it is reported as a small difference instead of a flag.
    """
    if abs(d3) <= s.close_deg:
        return "close"
    if abs(d3) > s.review_deg and np.isfinite(d2) and abs(d2) > s.close_deg and np.sign(d2) == np.sign(d3):
        return "review"
    return "minor"


def _smooth(x: np.ndarray, k: int = 5) -> np.ndarray:
    """NaN-aware running median along axis 0 (reduces landmark jitter, keeps sharp hits)."""
    if len(x) < k:
        return x.copy()
    pad = k // 2
    padded = np.concatenate([np.repeat(x[:1], pad, 0), x, np.repeat(x[-1:], pad, 0)])
    windows = np.lib.stride_tricks.sliding_window_view(padded, k, axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = np.nanmedian(windows, axis=-1)
    return np.where(np.isfinite(x), out, np.nan)      # do not invent values where there were none


def _sample(times: np.ndarray, values: np.ndarray, at: np.ndarray) -> np.ndarray:
    """Linear interpolation of rows of `values` (T, K) at times `at` (N,); NaN outside or if missing."""
    at = np.asarray(at, dtype=np.float64)
    out = np.full((len(at), values.shape[1]), np.nan)
    ok = np.isfinite(at) & (at >= times[0]) & (at <= times[-1])
    if not ok.any():
        return out
    idx = np.clip(np.searchsorted(times, at[ok]) - 1, 0, len(times) - 2)
    t0, t1 = times[idx], times[idx + 1]
    w = np.where(t1 > t0, (at[ok] - t0) / (t1 - t0), 0.0)[:, None]
    a, b = values[idx], values[idx + 1]
    blended = a * (1 - w) + b * w
    # If one neighbour is missing, use the other one when it is the nearer frame.
    nearest = np.where(w < 0.5, a, b)
    blended = np.where(np.isfinite(blended), blended, nearest)
    out[ok] = blended
    return out


def _fmt_time(seconds: float) -> str:
    m, s = divmod(max(0.0, float(seconds)), 60.0)
    return f"{int(m)}:{s:04.1f}"


def _r(x, nd: int = 2):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


def _swap_side(name: str) -> str:
    if name.startswith("left_"):
        return "right_" + name[5:]
    if name.startswith("right_"):
        return "left_" + name[6:]
    return name


def _pretty(name: str) -> str:
    side, joint = name.split("_", 1)
    return f"{side} {joint}"


def describe_difference(joint: str, diff: float) -> str:
    """Plain-language direction for learner-minus-reference angle `diff` (degrees)."""
    kind = joint.split("_", 1)[1]
    amount = f"about {abs(diff):.0f}°"
    if kind in ("elbow", "knee"):
        return f"straighter by {amount}" if diff > 0 else f"more bent by {amount}"
    if kind == "shoulder":
        return (f"raised higher / farther from your body by {amount}" if diff > 0
                else f"lower / closer to your body by {amount}")
    if kind == "hip":
        return (f"more open by {amount} (leg lower or upper body more upright)" if diff > 0
                else f"more closed by {amount} (leg lifted higher or leaning forward more)")
    return f"different by {amount}"


# --------------------------------------------------------------------------- tracking
def tracking_report(image, vis, detected, brightness, frames: np.ndarray, s: FeedbackSettings) -> dict:
    """How well the learner was seen in `frames` (indices), with reasons a person can act on."""
    n = int(len(frames))
    if n == 0:
        return {"status": "poor", "detected": 0.0, "regions": {}, "messages": [], "dark": False}
    img, v, det = image[frames], vis[frames], detected[frames]
    visible = _visible_mask(img, v, s) & det[:, None]
    detected_frac = float(det.mean())
    regions = {name: float(visible[:, list(idx)].all(axis=1).mean()) for name, idx in REGIONS.items()}

    problems: dict[tuple[str, str], list[tuple[str, float]]] = {}
    with np.errstate(invalid="ignore"):
        for name, idx in REGIONS.items():
            if regions[name] >= s.region_ok or detected_frac < s.min_detected:
                continue
            pts = img[det][:, list(idx)]
            share = 1.0 - regions[name]
            if len(pts):
                below = float(np.mean((pts[..., 1] > 1 + s.frame_margin).any(axis=1)))
                above = float(np.mean((pts[..., 1] < -s.frame_margin).any(axis=1)))
                side = float(np.mean(((pts[..., 0] < -s.frame_margin) | (pts[..., 0] > 1 + s.frame_margin)).any(axis=1)))
            else:
                below = above = side = 0.0
            reason = ("below" if below >= 0.3 else "above" if above >= 0.3 else "side" if side >= 0.3 else "unclear")
            kind = "torso" if name == "torso" else name.split("_")[1]           # arm | leg | torso
            side_name = None if name == "torso" else name.split("_")[0]
            problems.setdefault((kind, reason), []).append((side_name, share))

    messages = []
    for (kind, reason), items in problems.items():
        share = max(x for _, x in items)
        if kind == "torso":
            label = "Your upper body was"
        elif len(items) == 2:
            label = f"Your {kind}s were"
        else:
            label = f"Your {items[0][0]} {kind} was"
        where = {
            "below": "below the bottom edge of the frame",
            "above": "above the top of the frame",
            "side": "outside the side of the frame",
            "unclear": "hard to track (something in front of it, loose clothing, or low contrast)",
        }[reason]
        fix = {
            "below": (" Step back or tilt the camera down so your feet are visible." if kind == "leg"
                      else " Step back or tilt the camera down so your whole body fits in the frame."),
            "above": " Step back or tilt the camera up.",
            "side": " Stay inside the guide box.",
            "unclear": "",
        }[reason]
        messages.append(f"{label} {where} in {share:.0%} of the recording.{fix}")

    b = brightness[frames] if brightness is not None else np.array([np.nan])
    mean_b = float(np.nanmean(b)) if np.isfinite(b).any() else float("nan")
    dark = bool(np.isfinite(mean_b) and mean_b < s.dark_luma)

    if detected_frac < s.min_detected:
        messages.insert(0, f"MoveMatch found a person in only {detected_frac:.0%} of the recording. "
                           "Make sure your whole body is in the camera view.")
    if dark and (detected_frac < s.min_detected or min(regions.values()) < s.region_ok):
        messages.append("The video looks dark, which makes tracking unreliable. Turn on more light or face a window.")

    if detected_frac < s.min_detected or regions["torso"] < s.region_ok:
        status = "poor"
    elif min(regions.values()) >= s.region_good:
        status = "good"
    else:
        status = "partial"
    return {"status": status, "detected": round(detected_frac, 3),
            "regions": {k: round(x, 3) for k, x in regions.items()},
            "messages": messages, "dark": dark, "mean_brightness": _r(mean_b, 1)}


# --------------------------------------------------------------------------- timing
def _lag_cost(learner_ang, ref_t, ref_ang, mapped, shift: int, cols) -> tuple[float, int]:
    """Mean |angle difference| when learner frame i is compared with the reference at mapped[i - shift]."""
    n = len(mapped)
    idx = np.arange(n) - shift
    ok = (idx >= 0) & (idx < n)
    target = np.full(n, np.nan)
    target[ok] = mapped[idx[ok]]
    ref_vals = _sample(ref_t, ref_ang[:, cols], target)
    diff = np.abs(learner_ang[:, cols] - ref_vals)
    valid = np.isfinite(diff)
    return (float(diff[valid].mean()) if valid.any() else float("nan")), int(valid.sum())


def estimate_delay(learner_ang, ref_t, ref_ang, mapped, fps, s: FeedbackSettings, cols,
                   lag_range: tuple[float, float] | None = None) -> dict:
    """Best constant delay (wall-clock seconds, + = learner late) and whether to trust it."""
    lo, hi = lag_range or (s.lag_min_s, s.lag_max_s)
    shifts = np.arange(int(np.floor(lo * fps)), int(np.ceil(hi * fps)) + 1)
    results = [_lag_cost(learner_ang, ref_t, ref_ang, mapped, int(k), cols) for k in shifts]
    costs = np.array([c for c, _ in results])
    counts = np.array([n for _, n in results])
    base = counts.max() if len(counts) else 0
    costs = np.where(counts >= 0.3 * max(base, 1), costs, np.nan)
    if not np.isfinite(costs).any():
        return {"offset_s": None, "confident": False, "reason": "not enough tracked frames", "curve": []}
    best = int(np.nanargmin(costs))
    offset = shifts[best] / fps
    # Sub-frame refinement (parabola through the minimum and its neighbours).
    if 0 < best < len(costs) - 1 and np.isfinite(costs[best - 1:best + 2]).all():
        a, b, c = costs[best - 1], costs[best], costs[best + 1]
        denom = a - 2 * b + c
        if denom > 1e-9:
            offset += 0.5 * (a - c) / denom / fps
    contrast = float(np.nanmedian(costs) - costs[best])
    at_edge = best in (0, len(costs) - 1)
    confident = contrast >= s.min_lag_contrast_deg and not at_edge
    reason = None
    if at_edge:
        reason = "the best match was at the edge of the searched range"
    elif not confident:
        reason = "the movement was too small or tracking too noisy to measure timing"
    curve = [[round(float(k) / fps, 3), _r(c, 2)] for k, c in zip(shifts, costs)]
    return {"offset_s": round(float(offset), 3), "confident": bool(confident), "reason": reason,
            "contrast_deg": round(contrast, 2), "cost_deg": _r(costs[best], 2), "curve": curve}


def _timing_message(offset: float | None, confident: bool, speed: float, s: FeedbackSettings, reason) -> str:
    if offset is None or not confident:
        return f"Timing could not be measured reliably ({reason})." if reason else "Timing could not be measured."
    music = abs(offset) * speed
    speed_note = "" if abs(speed - 1.0) < 1e-6 else f" at {speed:g}× speed (≈{music:.2f} s of music)"
    if abs(offset) < s.on_time_s:
        return f"On time: within {s.on_time_s:.2f} s of the reference{speed_note}."
    if offset > 0:
        return (f"About {offset:.2f} s behind the reference{speed_note}. Following a video live usually adds "
                "some reaction delay; try to anticipate the next move.")
    return f"About {abs(offset):.2f} s ahead of the reference{speed_note}."


# --------------------------------------------------------------------------- key moments
def select_key_moments(ref_t: np.ndarray, ref_ang: np.ndarray, ref_ok_frames: np.ndarray,
                       section: tuple[float, float], s: FeedbackSettings) -> list[float]:
    """Reference times (s) of the moments to judge. Depends ONLY on the reference and the section."""
    a, b = section
    lo, hi = a + s.edge_s, b - s.edge_s
    in_sec = (ref_t >= lo) & (ref_t <= hi) & ref_ok_frames
    if in_sec.sum() < 3:
        return []
    cols = [_ANGLE_COL[n] for n in FEEDBACK_ANGLES]
    ang = ref_ang[:, cols]
    dt = np.gradient(ref_t)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        speed = np.nanmean(np.abs(np.gradient(ang, axis=0)), axis=1) / np.maximum(dt, 1e-6)
    speed = _smooth(speed[:, None], 5)[:, 0]
    fps = 1.0 / float(np.median(np.diff(ref_t))) if len(ref_t) > 1 else 30.0
    half = max(1, int(round(0.25 * fps)))
    candidates = []
    for i in np.flatnonzero(in_sec):
        if not np.isfinite(speed[i]):
            continue
        window = speed[max(0, i - half):i + half + 1]
        if speed[i] <= np.nanmin(window) + 1e-9:
            candidates.append((speed[i], ref_t[i]))
    gap = max(s.min_moment_gap_s, (hi - lo) / max(1, s.max_moments))
    chosen: list[float] = []
    for _, t in sorted(candidates):
        if len(chosen) >= s.max_moments:
            break
        if all(abs(t - c) >= gap for c in chosen):
            chosen.append(float(t))
    # Continuous movement has few still points: fill evenly so the whole section is covered.
    if len(chosen) < min(3, s.max_moments):
        for t in np.linspace(lo, hi, min(s.max_moments, max(3, int((hi - lo) / gap) + 1))):
            nearest = ref_t[in_sec][np.argmin(np.abs(ref_t[in_sec] - t))]
            if all(abs(nearest - c) >= gap * 0.75 for c in chosen):
                chosen.append(float(nearest))
            if len(chosen) >= s.max_moments:
                break
    return sorted(chosen)


def _yaw_deg(world: np.ndarray, visible: np.ndarray) -> float:
    if not visible[list(pu.TORSO_JOINTS)].all():
        return float("nan")
    return float(np.degrees(pu.yaw_angle(world)))


# --------------------------------------------------------------------------- main entry
def evaluate_attempt(ref, learner, learner_ref_times: np.ndarray, *, section: tuple[float, float],
                     speed: float, follow_mode: str = pp.FOLLOW_SAME_SIDE, capture_mirrored: bool = False,
                     settings: FeedbackSettings = DEFAULT_SETTINGS) -> dict:
    """Feedback for one attempt.

    ref, learner        landmark_store.LandmarkSequence (raw world + image landmarks, visibility)
    learner_ref_times   (T_learner,) reference time in seconds on screen at each learner frame
                        (NaN = no reference playing), from sync_timeline.SyncTimeline
    section             (start_s, end_s) of the reference that was practised
    follow_mode         "same_side" or "mirror" (learner deliberately dances the mirror image)
    No Flask, no database: returns a JSON-serializable dict.
    """
    s = settings
    flip = pp.mirror_needed(capture_mirrored, follow_mode)
    ref_t = np.asarray(ref.timestamps, dtype=np.float64) / 1000.0
    lt = np.asarray(learner.timestamps, dtype=np.float64) / 1000.0
    mapped = np.asarray(learner_ref_times, dtype=np.float64)
    if mapped.shape != lt.shape:
        raise ValueError("learner_ref_times must have one value per learner frame")
    a, b = float(section[0]), float(section[1])
    l_fps = 1.0 / float(np.median(np.diff(lt))) if len(lt) > 1 else 30.0

    result = {
        "algorithm_version": ALGORITHM_VERSION,
        "settings": s.as_dict(),
        "section": [round(a, 3), round(b, 3)],
        "speed": speed,
        "follow_mode": follow_mode,
        "learner_flipped": bool(flip),
        "withheld": False,
        "withheld_reasons": [],
    }

    # --- which learner frames saw the reference playing inside the section
    in_section = np.isfinite(mapped) & (mapped >= a - 0.05) & (mapped <= b + 0.05)
    frames = np.flatnonzero(in_section)
    mapped_s = len(frames) / l_fps
    tracking = tracking_report(learner.image_landmarks, learner.visibility, learner.detected,
                               learner.brightness, frames, s)
    result["tracking"] = tracking
    result["coverage"] = {"mapped_seconds": round(mapped_s, 2),
                          "section_seconds": round((b - a) / max(speed, 1e-6), 2)}

    ref_in = (ref_t >= a) & (ref_t <= b)
    ref_detect = float(ref.detected[ref_in].mean()) if ref_in.any() else 0.0
    if mapped_s < min(s.min_mapped_s, 0.5 * (b - a) / max(speed, 1e-6)):
        result["withheld_reasons"].append(
            "The recording barely overlapped the reference playback (it may have been stopped early), "
            "so there is nothing to compare.")
    if ref_detect < s.min_detected:
        result["withheld_reasons"].append(
            f"The dancer in the reference video was tracked in only {ref_detect:.0%} of this section, "
            "so MoveMatch cannot give reliable feedback here.")
    if tracking["status"] == "poor":
        result["withheld_reasons"].extend(tracking["messages"] or
                                          ["Your upper body was not visible enough to compare."])
    if result["withheld_reasons"]:
        result["withheld"] = True
        result["headline"] = "Feedback withheld: " + result["withheld_reasons"][0]
        return result

    # --- angles (visibility- and frame-masked), mirrored to the reference's sides if needed
    ref_vis_mask = _visible_mask(ref.image_landmarks, ref.visibility, s) & ref.detected[:, None]
    l_vis_mask = _visible_mask(learner.image_landmarks, learner.visibility, s) & learner.detected[:, None]
    ref_ang = _smooth(_angles(ref.landmarks, ref_vis_mask))
    l_ang = _smooth(_angles(learner.landmarks, l_vis_mask))
    ref_ang2 = _smooth(_image_angles(ref, ref_vis_mask))
    l_ang2 = _smooth(_image_angles(learner, l_vis_mask))
    l_world = np.asarray(learner.landmarks, dtype=np.float64)
    if flip:
        l_ang = pu.mirror_angles(l_ang)
        l_ang2 = pu.mirror_angles(l_ang2)
        l_vis_mask = pu.mirror_visibility(l_vis_mask)
        l_world = pu.mirror_pose(l_world)

    # Regions seen too rarely are excluded from feedback instead of counted as mistakes.
    # Regions are measured on the learner's own body; after a mirror flip the angle "left_elbow"
    # (reference side) is the learner's RIGHT elbow, so look up the swapped region.
    own = _swap_side if flip else (lambda n: n)
    use = [n for n in FEEDBACK_ANGLES if tracking["regions"].get(ANGLE_REGION[own(n)], 0) >= s.region_ok]
    excluded = sorted(set(FEEDBACK_ANGLES) - set(use))
    result["excluded_joints"] = excluded
    if not use:
        result["withheld"] = True
        result["withheld_reasons"].append("Neither your arms nor your legs were visible enough to compare.")
        result["withheld_reasons"].extend(tracking["messages"])
        result["headline"] = "Feedback withheld: " + result["withheld_reasons"][0]
        return result
    cols = [_ANGLE_COL[n] for n in use]

    # Only the practised section takes part (frames outside it would compare against nothing).
    mapped_sec = np.where(in_section, mapped, np.nan)

    # --- timing
    delay = estimate_delay(l_ang, ref_t, ref_ang, mapped_sec, l_fps, s, cols)
    offset = delay["offset_s"] if delay["confident"] else 0.0
    timing = {
        "offset_s": delay["offset_s"],
        "confident": delay["confident"],
        "direction": (None if not delay["confident"] else
                      "on_time" if abs(delay["offset_s"]) < s.on_time_s else
                      "late" if delay["offset_s"] > 0 else "early"),
        "message": _timing_message(delay["offset_s"], delay["confident"], speed, s, delay.get("reason")),
        "curve": delay["curve"],
        "contrast_deg": delay.get("contrast_deg"),
    }
    # Drift: the same search on overlapping windows, around the global delay.
    windows = []
    if delay["confident"]:
        win, hop = int(round(3.0 * l_fps)), int(round(1.5 * l_fps))
        for start in range(int(frames[0]), int(frames[-1]) - win + 2, max(1, hop)):
            sub = np.full_like(mapped_sec, np.nan)
            sub[start:start + win] = mapped_sec[start:start + win]
            if np.isfinite(sub).sum() < 0.6 * win:
                continue
            d = estimate_delay(l_ang, ref_t, ref_ang, sub, l_fps, s, cols,
                               lag_range=(delay["offset_s"] - 0.5, delay["offset_s"] + 0.5))
            if d["confident"]:
                windows.append({"reference_time_s": _r(np.nanmedian(sub), 2), "offset_s": d["offset_s"]})
        if len(windows) >= 2:
            first, last = windows[0]["offset_s"], windows[-1]["offset_s"]
            if abs(last - first) >= 0.3:
                trend = "later" if last > first else "earlier"
                timing["drift_message"] = (
                    f"Your timing changed during the section: {first:+.2f} s at "
                    f"{_fmt_time(windows[0]['reference_time_s'])}, {last:+.2f} s by "
                    f"{_fmt_time(windows[-1]['reference_time_s'])} (you got {trend}).")
    timing["windows"] = windows
    result["timing"] = timing

    # --- key moments (from the reference only)
    ref_ok = ref_vis_mask[:, list(pu.TORSO_JOINTS)].all(axis=1) & \
        (np.isfinite(ref_ang[:, cols]).sum(axis=1) >= max(s.min_joints_per_moment, len(cols) // 2))
    moments_t = select_key_moments(ref_t, ref_ang, ref_ok, (a, b), s)

    shift = int(round(offset * l_fps))
    idx = np.arange(len(mapped_sec)) - shift
    ok = (idx >= 0) & (idx < len(mapped_sec))
    corrected = np.full(len(mapped_sec), np.nan)
    corrected[ok] = mapped_sec[idx[ok]]                 # what the learner was "really" dancing

    moments = []
    all_diffs = []
    side_name = _swap_side if flip else (lambda n: n)
    for k, tau in enumerate(moments_t):
        ref_row = _sample(ref_t, ref_ang, np.array([tau]))[0]
        ref_row2 = _sample(ref_t, ref_ang2, np.array([tau]))[0]
        ref_frame = int(np.argmin(np.abs(ref_t - tau)))
        cand = np.flatnonzero(np.isfinite(corrected) & (np.abs(corrected - tau) <= s.moment_window_s))
        moment = {"index": k, "reference_time_s": round(tau, 3), "label": _fmt_time(tau),
                  "learner_time_s": None, "status": "not_visible", "joints": [], "issues": [], "facing": None}
        covered = np.isfinite(corrected)
        if not (covered.any() and corrected[covered].min() <= tau <= corrected[covered].max()):
            # The recording stopped (or started) before this moment: nothing to compare it with.
            moment["status"] = "not_covered"
            moment["issues"].append("The recording does not cover this moment (it started late or stopped early).")
            moments.append(moment)
            continue
        best, best_cost = None, np.inf
        for i in cand:
            d = l_ang[i, cols] - ref_row[cols]
            valid = np.isfinite(d)
            if valid.sum() < s.min_joints_per_moment:
                continue
            d2 = np.abs(l_ang2[i, cols] - ref_row2[cols])
            c = float(np.abs(d[valid]).mean()) + (float(np.nanmean(d2)) if np.isfinite(d2).any() else 0.0)
            if c < best_cost:
                best, best_cost = int(i), c
        if best is None:
            moment["issues"].append("You were not visible enough at this moment to compare.")
            moments.append(moment)
            continue
        moment["learner_time_s"] = round(float(lt[best]), 3)
        worst = 0.0
        for name, col in zip(use, cols):
            d = float(l_ang[best, col] - ref_row[col])
            if not np.isfinite(d):
                moment["joints"].append({"joint": name, "your_joint": side_name(name),
                                         "reference_deg": _r(ref_row[col], 1), "yours_deg": None,
                                         "diff_deg": None, "status": "not_visible"})
                continue
            d2 = float(l_ang2[best, col] - ref_row2[col])
            status = joint_status(d, d2, s)
            worst = max(worst, abs(d))
            all_diffs.append(abs(d))
            moment["joints"].append({"joint": name, "your_joint": side_name(name),
                                     "reference_deg": _r(ref_row[col], 1), "yours_deg": _r(l_ang[best, col], 1),
                                     "diff_deg": _r(d, 1), "picture_diff_deg": _r(d2, 1), "status": status,
                                     "depth_uncertain": bool(abs(d) > s.review_deg and status == "minor")})
        moment["max_diff_deg"] = round(worst, 1)
        statuses = {j["status"] for j in moment["joints"]}
        moment["status"] = "review" if "review" in statuses else "minor" if "minor" in statuses else "close"

        flagged = sorted((j for j in moment["joints"] if j["status"] in ("review", "minor")),
                         key=lambda j: -abs(j["diff_deg"]))
        flagged.sort(key=lambda j: (j["status"] != "review", -abs(j["diff_deg"])))
        for j in flagged[:2]:
            note = " (possibly a depth-estimation error: it does not show in the 2-D picture)" if j["depth_uncertain"] else ""
            moment["issues"].append(f"Your {_pretty(j['your_joint'])} was "
                                    f"{describe_difference(j['joint'], j['diff_deg'])}{note}.")

        ry = _yaw_deg(np.asarray(ref.landmarks[ref_frame], dtype=np.float64), ref_vis_mask[ref_frame])
        ly = _yaw_deg(l_world[best], l_vis_mask[best])
        if np.isfinite(ry) and np.isfinite(ly):
            fd = (ly - ry + 180.0) % 360.0 - 180.0
            moment["facing"] = {"reference_deg": round(ry, 0), "yours_deg": round(ly, 0), "diff_deg": round(fd, 0)}
            if abs(fd) > s.facing_review_deg:
                moment["status"] = "review"
                if abs(ry) > abs(ly) + 30:
                    msg = f"The reference is turned about {abs(ry):.0f}° from the camera here; you were facing it more directly."
                elif abs(ly) > abs(ry) + 30:
                    msg = f"You were turned about {abs(ly):.0f}° from the camera; the reference faces it more directly here."
                else:
                    msg = "You seem to be facing a different direction from the reference here."
                moment["issues"].insert(0, msg + " (Facing direction is estimated from one camera: check the replay.)")
        moments.append(moment)

    scored = [m for m in moments if m["status"] not in ("not_visible", "not_covered")]
    counts = {st: sum(1 for m in scored if m["status"] == st) for st in ("close", "minor", "review")}
    joint_counts: dict[str, int] = {}
    for m in scored:
        for j in m["joints"]:
            if j["status"] == "review":
                joint_counts[j["your_joint"]] = joint_counts.get(j["your_joint"], 0) + 1
    recurring = max(joint_counts.items(), key=lambda kv: kv[1]) if joint_counts else None

    posture = {
        "moments_total": len(scored),
        "moments_not_visible": len(moments) - len(scored),
        "close": counts["close"], "minor": counts["minor"], "review": counts["review"],
        "median_abs_diff_deg": _r(np.median(all_diffs), 1) if all_diffs else None,
        "recurring": (f"Your {_pretty(recurring[0])} differed most often "
                      f"({recurring[1]} of {len(scored)} moments)." if recurring and recurring[1] >= 2 else None),
    }
    result["posture"] = posture
    result["moments"] = moments

    parts = []
    if scored:
        if counts["review"]:
            parts.append(f"{counts['review']} of {len(scored)} moments worth reviewing")
        else:
            parts.append(f"shape close at {counts['close']} of {len(scored)} moments"
                         + (f" ({counts['minor']} small differences)" if counts["minor"] else ""))
    if timing["direction"] == "late":
        parts.append(f"about {timing['offset_s']:.2f} s late")
    elif timing["direction"] == "early":
        parts.append(f"about {abs(timing['offset_s']):.2f} s early")
    elif timing["direction"] == "on_time":
        parts.append("on time")
    headline = " · ".join(parts) or "No moments could be compared"
    result["headline"] = headline[0].upper() + headline[1:]
    if excluded:
        result["partial_note"] = ("Not compared: " + ", ".join(_pretty(side_name(n)) for n in excluded)
                                  + " (not visible enough in this recording).")
    return result


def compare_attempts(current: dict, previous: dict) -> list[dict]:
    """Moment-by-moment change between two evaluations of the same section."""
    if not current or not previous or current.get("section") != previous.get("section"):
        return []
    before = {round(m["reference_time_s"], 2): m for m in previous.get("moments", [])}
    rank = {"close": 0, "minor": 1, "review": 2}
    out = []
    for m in current.get("moments", []):
        p = before.get(round(m["reference_time_s"], 2))
        if not p:
            continue
        now, then = m["status"], p["status"]
        change = ("same" if now == then else "unknown" if now not in rank or then not in rank
                  else "better" if rank[now] < rank[then] else "worse")
        out.append({"label": m["label"], "reference_time_s": m["reference_time_s"],
                    "before": then, "now": now, "change": change,
                    "before_max_diff_deg": p.get("max_diff_deg"), "now_max_diff_deg": m.get("max_diff_deg")})
    return out
