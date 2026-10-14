"""Pose score: turn a DTW alignment into a number a learner can act on.

Pure NumPy. Builds on pose_dtw (alignment) and pose_utils (normalization).

What is kept from every alignment (nothing is thrown away before scoring)
------------------------------------------------------------------------
* mean_pose_distance  average pose mismatch along the alignment, in torso lengths
* alignment_path      the matched (reference frame, learner frame) pairs
* summary             how much temporal stretching occurred (step counts, runs, rates, flags)

Which distance?
---------------
* mean_pose_distance averages PER REFERENCE FRAME: every reference frame counts once, however
  many learner frames it was matched to. A plain mean over path cells would count a reference
  frame once per matched learner frame, so a learner who pauses would silently reweight the score.
  The plain path mean is kept as path_mean_distance for comparison.
* Pairs where tracking was blind (no usable joint) are EXCLUDED, not counted as errors: a
  dropout is not a pose mismatch. How much was excluded is reported as tracking_coverage, and low
  coverage raises a warning instead of quietly lowering the score.

Score
-----
    S_pose = 100 * exp(-k * mean_pose_distance)

k is a calibration constant (see calibrate_k and calibrate_pose_k.py). DEFAULT_K is PROVISIONAL.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pose_dtw import (DTWResult, StretchReport, band_edge_fraction, band_radius_from_seconds,
                      check_stretching, dtw, pairwise_pose_cost, subsequence_dtw)

# PROVISIONAL. Best compromise on synthetic performances of four dances (see calibrate_pose_k.py and
# pose_k_calibration.png): the requirements "skilled dancer >= 90, poor <= 30, adjacent levels 8
# points apart" cannot all hold for this score shape on that data, and k = 2.0 minimizes the worst
# violation. Recalibrate on real labelled recordings before showing scores to learners.
DEFAULT_K = 2.0


# --------------------------------------------------------------------------
# Distance -> score
# --------------------------------------------------------------------------
def distance_to_score(distance, k: float = DEFAULT_K):
    """100 * exp(-k * distance). Works on scalars or arrays; NaN stays NaN. 0 -> 100, decreasing."""
    if not k > 0:
        raise ValueError("k must be positive")
    d = np.asarray(distance, dtype=np.float64)
    if (d[np.isfinite(d)] < 0).any():
        raise ValueError("distance must be non-negative")
    out = 100.0 * np.exp(-k * d)
    return float(out) if out.ndim == 0 else out


def score_to_distance(score, k: float = DEFAULT_K):
    """Inverse of distance_to_score: the mean pose distance that earns `score` (0 < score <= 100)."""
    s = np.asarray(score, dtype=np.float64)
    if ((s <= 0) | (s > 100)).any():
        raise ValueError("score must be in (0, 100]")
    out = -np.log(s / 100.0) / k
    return float(out) if out.ndim == 0 else out


# --------------------------------------------------------------------------
# Per-frame distances along the path
# --------------------------------------------------------------------------
def per_reference_distance(cost: np.ndarray, path: np.ndarray, usable: np.ndarray | None = None) -> np.ndarray:
    """Mean pose distance of each reference frame over the learner frames matched to it. (n_ref,)

    usable: optional (n_ref, n_query) boolean matrix; pairs marked False are ignored. A reference
    frame with no usable pair is NaN (tracking never saw it).
    """
    n = cost.shape[0]
    c = cost[path[:, 0], path[:, 1]]
    w = np.ones(len(path)) if usable is None else usable[path[:, 0], path[:, 1]].astype(float)
    total = np.bincount(path[:, 0], weights=c * w, minlength=n)
    count = np.bincount(path[:, 0], weights=w, minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / count, np.nan)


# --------------------------------------------------------------------------
# How much does this dance move? (a property of the reference alone)
# --------------------------------------------------------------------------
def reference_motion_scale(ref: np.ndarray, ref_vis: np.ndarray | None = None) -> float:
    """Mean distance between the reference's frames and its own median pose, in torso lengths.

    This is roughly the mean_pose_distance of a dancer who stands still in the dance's average
    posture. A dance with little motion has a small value, so even a dancer who does nothing stays
    close to the reference and scores deceptively well. Computed from the reference only.
    """
    ref = np.asarray(ref, dtype=np.float64)
    median_pose = np.nanmedian(ref, axis=0)[None]
    cost, _ = pairwise_pose_cost(ref, median_pose, ref_vis, None, nan_cost=None)
    return float(np.nanmean(cost))


# --------------------------------------------------------------------------
# Alignment statistics
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class AlignmentSummary:
    """How the learner's timeline was bent to match the reference."""
    n_ref: int
    n_query: int
    path_length: int
    diagonal_steps: int
    horizontal_steps: int            # reference frame repeated (learner slower / paused)
    vertical_steps: int              # learner frame repeated (learner faster / skipped)
    stretch_fraction: float          # (horizontal + vertical) / all steps; 0 = pure 1:1 timing
    query_start: int
    query_end: int
    stretch: StretchReport           # runs, local rates, flags (see pose_dtw.check_stretching)
    band_edge_fraction: float | None # share of path pinned to the band edge (None if no band)


def summarize_alignment(result: DTWResult, fps: float = 30.0, band: float | None = None,
                        **stretch_kwargs) -> AlignmentSummary:
    """Step counts plus the stretch diagnostics of pose_dtw, in one record."""
    path = result.path
    steps = np.diff(path, axis=0)
    h = int(((steps[:, 0] == 0) & (steps[:, 1] == 1)).sum())
    v = int(((steps[:, 0] == 1) & (steps[:, 1] == 0)).sum())
    d = int(((steps[:, 0] == 1) & (steps[:, 1] == 1)).sum())
    n_steps = max(1, h + v + d)
    rep = check_stretching(path, n_ref=result.n_ref, fps=fps, **stretch_kwargs)
    edge = None if band is None else band_edge_fraction(path, result.n_ref, result.n_query, band)
    return AlignmentSummary(result.n_ref, result.n_query, len(path), d, h, v, (h + v) / n_steps,
                            result.query_start, result.query_end, rep, edge)


# --------------------------------------------------------------------------
# Pose score
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class PoseScore:
    score: float                     # 0..100, NaN if nothing could be scored
    k: float
    mean_pose_distance: float        # per-reference-frame mean over usable matches, torso lengths
    path_mean_distance: float        # plain mean over usable path cells (for comparison)
    p90_distance: float              # 90th percentile of per-reference-frame distances: worst moments
    alignment_path: np.ndarray       # (L, 2) matched (reference frame, learner frame) pairs
    summary: AlignmentSummary
    tracking_coverage: float         # share of reference frames with at least one usable match
    per_frame_distance: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))
    warnings: tuple[str, ...] = ()
    still_baseline_score: float = float("nan")   # score a dancer who stands still would earn on this reference


def score_alignment(
    result: DTWResult,
    cost: np.ndarray,
    confidence: np.ndarray | None = None,
    k: float = DEFAULT_K,
    fps: float = 30.0,
    band: float | None = None,
    min_coverage: float = 0.7,
    max_band_edge: float = 0.1,
    reference_motion: float | None = None,
    max_still_baseline: float = 40.0,
) -> PoseScore:
    """Assemble the pose score and its supporting numbers from a DTW result.

    cost / confidence come from pairwise_pose_cost (confidence 0 marks blind pairs; without it every
    pair is treated as usable). reference_motion (see reference_motion_scale) adds the
    still_baseline_score and a warning when standing still would already earn more than
    max_still_baseline points. Warnings (never silent): stretching flags, band pinned to its edge,
    low tracking coverage, low-motion reference.
    """
    usable = None if confidence is None else confidence > 1e-9
    per_frame = per_reference_distance(cost, result.path, usable)
    seen = np.isfinite(per_frame)
    coverage = float(seen.mean())
    cells = cost[result.path[:, 0], result.path[:, 1]]
    if usable is not None:
        cells = cells[usable[result.path[:, 0], result.path[:, 1]]]
    summary = summarize_alignment(result, fps=fps, band=band)

    warnings: list[str] = list(summary.stretch.flags)
    if summary.band_edge_fraction is not None and summary.band_edge_fraction > max_band_edge:
        warnings.append(f"{summary.band_edge_fraction:.0%} of the path is pinned to the band edge: "
                        "timing offset may exceed the band")
    if coverage < min_coverage:
        warnings.append(f"only {coverage:.0%} of reference frames had usable tracking")

    still = float("nan")
    if reference_motion is not None and np.isfinite(reference_motion):
        still = distance_to_score(reference_motion, k)
        if still > max_still_baseline:
            warnings.append(f"standing still would earn {still:.0f} points on this reference: "
                            "little motion, so the pose score separates good from poor weakly")

    if not seen.any():
        return PoseScore(float("nan"), k, float("nan"), float("nan"), float("nan"), result.path, summary,
                         coverage, per_frame, tuple(warnings) + ("no usable frames: nothing to score",), still)
    mean_d = float(per_frame[seen].mean())
    return PoseScore(distance_to_score(mean_d, k), k, mean_d, float(cells.mean()),
                     float(np.percentile(per_frame[seen], 90)), result.path, summary, coverage,
                     per_frame, tuple(warnings), still)


def align_performance(
    ref: np.ndarray,
    query: np.ndarray,
    ref_vis: np.ndarray | None = None,
    query_vis: np.ndarray | None = None,
    fps: float = 30.0,
    band_s: float | None = 3.0,
    max_run: int | None = 8,
    subsequence: bool = False,
) -> tuple[DTWResult, np.ndarray, np.ndarray, float | None]:
    """The alignment used by every sub-score: (DTWResult, cost, confidence, band radius in frames).

    Defaults follow the DTW findings: a band of band_s seconds plus a slope limit for a complete
    performance; subsequence=True (no band) for a segment inside a longer recording. Pose and timing
    scores both read this ONE alignment, so they always describe the same matching.
    """
    cost, conf = pairwise_pose_cost(ref, query, ref_vis, query_vis)
    if subsequence:
        return subsequence_dtw(cost, max_run=max_run), cost, conf, None
    band = None if band_s is None else band_radius_from_seconds(band_s, fps)
    return dtw(cost, band=band, max_run=max_run), cost, conf, band


def score_performance(
    ref: np.ndarray,
    query: np.ndarray,
    ref_vis: np.ndarray | None = None,
    query_vis: np.ndarray | None = None,
    k: float = DEFAULT_K,
    fps: float = 30.0,
    band_s: float | None = 3.0,
    max_run: int | None = 8,
    subsequence: bool = False,
) -> PoseScore:
    """End to end: normalized reference and learner poses -> PoseScore (see align_performance)."""
    result, cost, conf, band = align_performance(ref, query, ref_vis, query_vis, fps, band_s, max_run, subsequence)
    return score_alignment(result, cost, conf, k=k, fps=fps, band=band,
                           reference_motion=reference_motion_scale(ref, ref_vis))


# --------------------------------------------------------------------------
# Calibrating k
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class KCalibration:
    k: float                         # chosen value (geometric middle of the feasible range)
    k_min: float                     # smallest k meeting every requirement (nan if infeasible)
    k_max: float                     # largest k meeting every requirement
    feasible: bool
    medians: dict[str, float]        # median distance per level
    scores: dict[str, float]         # median score per level at the chosen k
    notes: tuple[str, ...]


def calibrate_k(
    distances: dict[str, np.ndarray],
    order: list[str],
    floor_level: str,
    bad_level: str,
    floor_min_score: float = 90.0,
    bad_max_score: float = 30.0,
    min_gap: float = 8.0,
) -> KCalibration:
    """Pick k from labelled example performances, using requirements stated up front.

    distances  mean_pose_distance values of example performances, grouped by quality level.
    order      levels from best (smallest distance) to worst.
    Requirements (all judged on level MEDIANS):
      1. the floor level (e.g. a faithful repeat of the choreography) scores >= floor_min_score
         - the scale must not saturate: good dancers need headroom, tracking noise is not a fault;
      2. the bad level (clearly poor execution) scores <= bad_max_score
         - the scale must actually fall for poor dancing;
      3. each adjacent pair of levels in `order` differs by >= min_gap points
         - neighbouring quality levels must be distinguishable.
    Returns the feasible k range and its geometric middle. If no k meets all three the result has
    feasible=False and the k that minimizes the worst violation, with notes saying which failed.
    These thresholds are a design decision, not a measurement: change them deliberately.
    """
    med = {name: float(np.nanmedian(distances[name])) for name in order}
    ks = np.geomspace(0.02, 50.0, 5000)

    def margins(k: float) -> np.ndarray:
        s = {name: 100 * np.exp(-k * med[name]) for name in order}
        m = [s[floor_level] - floor_min_score, bad_max_score - s[bad_level]]
        m += [s[a] - s[b] - min_gap for a, b in zip(order[:-1], order[1:])]
        return np.array(m)

    M = np.array([margins(k) for k in ks])
    ok = (M >= 0).all(axis=1)
    notes: list[str] = []
    if ok.any():
        k_min, k_max = float(ks[ok].min()), float(ks[ok].max())
        k = float(np.sqrt(k_min * k_max))
        if not ok[np.argmin(np.abs(ks - k))]:
            # feasible set is not contiguous: fall back to the feasible k with the largest worst-margin
            idx = np.flatnonzero(ok)
            k = float(ks[idx[np.argmax(M[idx].min(axis=1))]])
            notes.append("feasible range is not contiguous; chose the k with the largest worst-case margin")
        feasible = True
    else:
        k = float(ks[np.argmax(M.min(axis=1))])
        k_min = k_max = float("nan")
        feasible = False
        names = ["floor level too low", "bad level too high"] + [f"{a} vs {b} too close" for a, b in zip(order[:-1], order[1:])]
        worst = M[np.argmax(M.min(axis=1))]
        notes += [f"no k meets every requirement; at the best compromise: {n} (margin {m:+.1f})"
                  for n, m in zip(names, worst) if m < 0]
    scores = {name: float(100 * np.exp(-k * med[name])) for name in order}
    return KCalibration(k, k_min, k_max, feasible, med, scores, tuple(notes))