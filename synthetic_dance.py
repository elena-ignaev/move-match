"""Synthetic dance data with a KNOWN time warp, for testing and demonstrating DTW.

A choreography is a smooth, non-repeating pose trajectory (a few slow sinusoids driving random
joint displacement fields, torso kept fixed so normalization works). A "performance" samples it at
chosen choreography times, so the true frame-to-frame correspondence is known exactly.
"""

from __future__ import annotations

import numpy as np

from pose_utils import (L_HIP, L_SHOULDER, NUM_LANDMARKS, R_HIP, R_SHOULDER, TORSO_JOINTS,
                        mask_low_visibility, normalize_sequence)

FPS = 30


class Choreography:
    """pose(t) for any array of times t in seconds -> (T, 33, 3) world-like coordinates (meters)."""

    def __init__(self, seed: int = 0, n_channels: int = 8, motion: float = 1.0,
                 holds: tuple[tuple[float, float], ...] = ()) -> None:
        """motion scales joint displacement (smaller = poses look more alike).

        holds: (start_s, duration_s) pairs. The dancer freezes on the pose at start_s for the
        duration (like a held pose): every frame inside matches every other, which is exactly
        where unconstrained DTW can spread the stretch arbitrarily.
        """
        self.holds = tuple(holds)
        rng = np.random.default_rng(seed)
        base = rng.normal(0.0, 0.15, (NUM_LANDMARKS, 3))
        base[L_HIP], base[R_HIP] = [0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]
        base[L_SHOULDER], base[R_SHOULDER] = [0.2, -0.5, 0.0], [-0.2, -0.5, 0.0]   # torso 0.5 m, y down
        basis = rng.normal(0.0, 0.18, (n_channels, NUM_LANDMARKS, 3))
        basis[:, list(TORSO_JOINTS), :] = 0.0
        basis *= motion
        self.base, self.basis = base, basis
        self.freq = rng.uniform(0.08, 0.45, (n_channels, 2))      # Hz, incommensurate: never repeats
        self.phase = rng.uniform(0.0, 2 * np.pi, (n_channels, 2))
        self.amp = rng.uniform(0.6, 1.0, (n_channels, 2))

    def pose(self, t: np.ndarray, amplitude: float = 1.0) -> np.ndarray:
        """amplitude < 1 dances 'smaller' (moves shrink toward the rest pose)."""
        t = np.asarray(t, dtype=np.float64)
        for start, dur in self.holds:                  # dance clock stands still during a hold
            t = t - np.clip(t - start, 0.0, dur)
        t = t[:, None, None]
        s = (self.amp * np.sin(2 * np.pi * self.freq * t + self.phase)).sum(axis=-1)   # (T, K)
        return self.base + amplitude * np.einsum("tk,kjc->tjc", s, self.basis)


def tau_from_knots(chor_knots, learner_knots, fps: int = FPS, tail_s: float = 0.0) -> np.ndarray:
    """Choreography time (s) of each learner frame, for a piecewise-linear learner timeline.

    learner_knots[k] is the learner time at which choreography time chor_knots[k] is performed.
    Before the first knot the learner idles on the first pose; after the last, on the last pose
    (plus tail_s seconds of idling).
    """
    end = learner_knots[-1] + tail_s
    u = np.arange(int(round(end * fps))) / fps
    return np.interp(u, learner_knots, chor_knots)


def add_tracking_artifacts(xyz: np.ndarray, noise: float = 0.01, dropout: float = 0.02,
                           seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame landmark noise (meters) and random low-visibility joints. Returns (xyz, vis)."""
    rng = np.random.default_rng(seed)
    out = xyz + rng.normal(0.0, noise, xyz.shape)
    vis = np.ones(xyz.shape[:2])
    vis[rng.random(vis.shape) < dropout] = 0.2
    return out, vis


def make_performance(choreo: Choreography, tau: np.ndarray, noise: float = 0.01,
                     dropout: float = 0.02, seed: int = 1, amplitude: float = 1.0
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Sample the choreography at times tau, add landmark noise and random low-visibility joints.

    Returns (xyz (T, 33, 3), vis (T, 33)). noise is in meters (torso is 0.5 m).
    """
    return add_tracking_artifacts(choreo.pose(tau, amplitude), noise, dropout, seed)


def to_normalized(xyz: np.ndarray, vis: np.ndarray) -> np.ndarray:
    """Standard pipeline from pose_utils: mask low visibility, center, scale (rolling body scale)."""
    return normalize_sequence(mask_low_visibility(xyz, vis), vis=vis)


def mean_path_error(path: np.ndarray, tau_frames: np.ndarray, ref_offset: float = 0.0) -> float:
    """Mean |matched reference frame - true reference frame| over path cells, in frames.

    tau_frames[j] = true reference frame of query frame j (choreography time * fps minus the
    reference's own start, via ref_offset when the reference is only a segment).
    """
    truth = tau_frames[path[:, 1]] - ref_offset
    return float(np.abs(path[:, 0] - truth).mean())


# --------------------------------------------------------------------------
# Execution-quality knobs (what makes one dancer's version differ from another's)
# --------------------------------------------------------------------------
def add_execution_error(xyz: np.ndarray, sigma: float, seed: int = 2, corr_s: float = 0.5,
                        fps: int = FPS) -> np.ndarray:
    """Add temporally SMOOTH per-joint offsets (std sigma meters, correlation time corr_s).

    Unlike per-frame tracking noise this does not average out: it models a limb that is
    consistently a few centimetres off for half a second, i.e. a genuine execution error.
    Torso joints are left alone so normalization is unaffected.
    """
    if sigma <= 0:
        return xyz
    rng = np.random.default_rng(seed)
    T = len(xyz)
    half = max(1, int(round(corr_s * fps)))
    kernel = np.exp(-0.5 * (np.arange(-3 * half, 3 * half + 1) / half) ** 2)
    kernel /= np.sqrt((kernel ** 2).sum())                      # keeps unit variance after smoothing
    white = rng.normal(0.0, 1.0, (T + len(kernel) - 1, NUM_LANDMARKS, 3))
    smooth = np.stack([np.convolve(white[:, j, c], kernel, mode="valid")
                       for j in range(NUM_LANDMARKS) for c in range(3)], axis=1).reshape(T, NUM_LANDMARKS, 3)
    smooth[:, list(TORSO_JOINTS), :] = 0.0
    return xyz + sigma * smooth


def scale_limbs(xyz: np.ndarray, factor: float) -> np.ndarray:
    """A dancer with different limb proportions: non-torso joints scaled about the hip centre."""
    if factor == 1.0:
        return xyz
    centre = (xyz[:, L_HIP] + xyz[:, R_HIP])[:, None, :] / 2.0
    out = centre + (xyz - centre) * factor
    out[:, list(TORSO_JOINTS), :] = xyz[:, list(TORSO_JOINTS), :]
    return out


# --------------------------------------------------------------------------
# Learner timelines with KNOWN timing (for validating and calibrating the timing score)
# --------------------------------------------------------------------------
def random_timeline(rng: np.random.Generator, duration_s: float, tempo_sigma: float = 0.0,
                    global_tempo: float = 1.0, segment_s: float = 2.0,
                    pauses: tuple[tuple[float, float], ...] = (), idle_start: float = 0.5,
                    idle_end: float = 0.5, fps: int = FPS) -> np.ndarray:
    """Choreography time (s) of each learner frame, for a learner whose tempo wanders.

    global_tempo   > 1 means slower (1.2 = every second of choreography takes 1.2 s).
    tempo_sigma    log-normal wobble of the tempo per segment_s of choreography (local rushing/dragging).
    pauses         (choreography_time_s, length_s): the learner freezes there before continuing.
    idle_start/end seconds standing on the first/last pose before/after the dance.
    """
    edges = np.arange(0.0, duration_s - 1e-9, segment_s)
    edges = np.append(edges, duration_s)
    factors = global_tempo * np.exp(rng.normal(0.0, tempo_sigma, len(edges) - 1))
    bounds = sorted(set(edges.tolist()) | {p for p, _ in pauses})
    pause_at = dict(pauses)
    chor, learner = [0.0, 0.0], [0.0, idle_start]
    clock = idle_start
    for a, b in zip(bounds[:-1], bounds[1:]):
        if a in pause_at:
            clock += pause_at[a]
            chor.append(a)
            learner.append(clock)
        seg = min(int(np.searchsorted(edges, a, side="right")) - 1, len(factors) - 1)
        clock += (b - a) * factors[seg]
        chor.append(b)
        learner.append(clock)
    chor.append(duration_s)
    learner.append(clock + idle_end)
    return tau_from_knots(chor, learner, fps)


def true_learner_times(tau: np.ndarray, event_times_s: np.ndarray, fps: int = FPS) -> np.ndarray:
    """Learner time (s) at which the learner reaches each choreography time. Ground truth for events.

    Only well defined while the choreography clock is moving: inside a pause or hold the answer is
    ambiguous, so callers should not trust truth there.
    """
    u = np.arange(len(tau)) / fps
    strictly = tau + np.arange(len(tau)) * 1e-9
    return np.interp(event_times_s, strictly, u)