"""Constrained dynamic time warping (DTW) for the dance app.

Pure NumPy, like pose_utils: no plotting, MediaPipe or Flask imports. Plots live in dtw_plot.py.

Conventions
-----------
* Rows of the cost matrix are the REFERENCE frames (n), columns are the learner's QUERY
  frames (m): cost[i, j] = how different reference frame i is from query frame j.
* A path is an (L, 2) int array of (i, j) cells from start to end, moving one step at a time:
    diagonal    (i+1, j+1)  both advance
    horizontal  (i,   j+1)  the same reference frame matches another query frame (learner slower)
    vertical    (i+1, j  )  the same query frame matches another reference frame (learner faster)
* Objective = sum of cost over the path cells + step_penalty for every non-diagonal step.
  The reported `cost` leaves the penalty out; `normalized_cost` = cost / path length, so
  alignments of different lengths are comparable. With pose costs it is in torso lengths.

Constraints (all optional, combinable except band + open ends)
--------------------------------------------------------------
* band      Sakoe-Chiba radius in query frames around the straight line from (0, 0) to
            (n-1, m-1). Use band_radius_from_seconds(). Caps how far the learner's timing may
            drift from "the same moment in both videos". Needs both ends fixed.
* max_run   Slope constraint: at most this many consecutive horizontal (or vertical) steps, so
            no frame is matched to more than max_run + 1 frames of the other sequence in a row.
            Stops the path from collapsing a long stretch of one video onto a single frame.
* open_begin / open_end   The path may start / end at any query frame, but still covers the
            whole reference. subsequence_dtw() sets both: align the reference against a
            portion of a longer recording (early start, extra tail, or one segment of a dance).

Cheap defaults to reason about: a complete performance -> band; one segment inside a longer
recording -> subsequence_dtw; always look at check_stretching() on the result.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # project root: pose_utils, landmark_store

from dataclasses import dataclass, field

import numpy as np

from pose_utils import DEFAULT_JOINT_WEIGHTS, NUM_LANDMARKS


# --------------------------------------------------------------------------
# Frame-to-frame cost
# --------------------------------------------------------------------------
def pairwise_pose_cost(
    ref: np.ndarray,
    query: np.ndarray,
    ref_vis: np.ndarray | None = None,
    query_vis: np.ndarray | None = None,
    joint_weights: np.ndarray | None = None,
    nan_cost: float | None = 1.0,
    chunk: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """All-pairs pose_distance between two normalized sequences, without a Python loop.

    ref (N, 33, 3), query (M, 33, 3), optional visibility (N, 33) and (M, 33). Same weighting as
    pose_utils.pose_distance (joint weight * min visibility, NaN joints ignored).

    Returns (cost (N, M), confidence (N, M)). Pairs with no usable joints are NaN in `cost`
    unless nan_cost is given, which replaces them (default 1.0 = a one-torso-length error, a
    clearly bad match). Pass nan_cost=None to keep the NaNs.
    """
    a = _as_sequence(ref, "ref")
    b = _as_sequence(query, "query")
    jw = DEFAULT_JOINT_WEIGHTS if joint_weights is None else np.asarray(joint_weights, float)
    va = np.ones(a.shape[:2]) if ref_vis is None else np.clip(np.asarray(ref_vis, float), 0, 1)
    vb = np.ones(b.shape[:2]) if query_vis is None else np.clip(np.asarray(query_vis, float), 0, 1)
    if va.shape != a.shape[:2] or vb.shape != b.shape[:2]:
        raise ValueError("visibility must have shape (T, 33) matching its sequence")

    n, m = a.shape[0], b.shape[0]
    cost = np.empty((n, m))
    conf = np.empty((n, m))
    full = jw.sum()
    for s in range(0, n, chunk):
        sl = slice(s, s + chunk)
        d = np.linalg.norm(a[sl, None] - b[None], axis=-1)                 # (c, M, 33)
        w = jw * np.minimum(va[sl, None], vb[None])
        valid = np.isfinite(d)
        w = np.where(valid, w, 0.0)
        d = np.where(valid, d, 0.0)
        total = w.sum(axis=-1)
        with np.errstate(invalid="ignore", divide="ignore"):
            c = (w * d).sum(axis=-1) / total
        cost[sl] = np.where(total > 1e-9, c, np.nan)
        conf[sl] = total / full if full > 0 else 0.0
    if nan_cost is not None:
        cost = np.where(np.isfinite(cost), cost, float(nan_cost))
    return cost, conf


def _as_sequence(x, name: str) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 3 or x.shape[1:] != (NUM_LANDMARKS, 3):
        raise ValueError(f"{name} must have shape (T, 33, 3), got {x.shape}")
    return x


# --------------------------------------------------------------------------
# Constraints
# --------------------------------------------------------------------------
def band_radius_from_seconds(seconds: float, fps: float) -> int:
    """Sakoe-Chiba radius in frames for a timing tolerance in seconds."""
    if seconds < 0 or fps <= 0:
        raise ValueError("seconds must be >= 0 and fps positive")
    return int(np.ceil(seconds * fps))


def sakoe_chiba_mask(n: int, m: int, radius: float) -> np.ndarray:
    """Boolean (n, m) mask of cells within `radius` query frames of the line (0,0) -> (n-1,m-1).

    The line is slanted when the sequences differ in length, so the band still ends in the
    corner. A radius too small to connect the corners makes dtw() raise (see "feasible").
    """
    if radius < 0:
        raise ValueError("radius must be >= 0")
    slope = (m - 1) / (n - 1) if n > 1 else 0.0
    i = np.arange(n, dtype=np.float64)[:, None]
    j = np.arange(m, dtype=np.float64)[None, :]
    return np.abs(j - i * slope) <= radius + 1e-9


# --------------------------------------------------------------------------
# DTW
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class DTWResult:
    path: np.ndarray            # (L, 2) int (i = reference frame, j = query frame)
    cost: float                 # sum of cost over the path (no step penalty)
    normalized_cost: float      # cost / L
    objective: float            # what the DP minimized: cost + step penalties
    n_ref: int
    n_query: int

    @property
    def query_start(self) -> int:
        return int(self.path[0, 1])

    @property
    def query_end(self) -> int:
        return int(self.path[-1, 1])


def dtw(
    cost: np.ndarray,
    band: float | None = None,
    max_run: int | None = None,
    open_begin: bool = False,
    open_end: bool = False,
    step_penalty: float = 0.0,
) -> DTWResult:
    """Constrained DTW over a (n_ref, n_query) cost matrix. See the module docstring.

    band          Sakoe-Chiba radius in query frames, or None. Not combinable with open ends.
    max_run       max consecutive horizontal (or vertical) steps, or None for unlimited.
    open_begin    path may start at any query frame (start of the reference is still row 0).
    open_end      path may end at any query frame (end of the reference is still the last row).
    step_penalty  extra cost (same units as `cost`) per non-diagonal step. A soft version of
                  max_run: discourages stretching without forbidding it.

    Raises ValueError for a non-finite cost matrix (fill NaNs first, e.g. pairwise_pose_cost's
    nan_cost), bad arguments, or when the constraints leave no path (widen the band or max_run).
    """
    C = np.asarray(cost, dtype=np.float64)
    if C.ndim != 2 or C.size == 0:
        raise ValueError(f"cost must be a non-empty 2-D matrix, got shape {C.shape}")
    if not np.isfinite(C).all():
        raise ValueError("cost contains NaN or inf; replace them first (see pairwise_pose_cost nan_cost)")
    if (C < 0).any():
        raise ValueError("cost must be non-negative")
    if step_penalty < 0:
        raise ValueError("step_penalty must be >= 0")
    if max_run is not None and (int(max_run) != max_run or max_run < 1):
        raise ValueError("max_run must be a positive integer or None")
    n, m = C.shape
    if band is not None and (open_begin or open_end):
        raise ValueError("band assumes both ends are fixed; it cannot be combined with open_begin/open_end")
    allowed = np.ones((n, m), dtype=bool) if band is None else sakoe_chiba_mask(n, m, band)

    if max_run is None:
        path, objective = _dp_plain(C, allowed, open_begin, open_end, float(step_penalty))
    else:
        path, objective = _dp_runs(C, allowed, int(max_run), open_begin, open_end, float(step_penalty))
    total = float(C[path[:, 0], path[:, 1]].sum())
    return DTWResult(path, total, total / len(path), float(objective), n, m)


def subsequence_dtw(cost: np.ndarray, max_run: int | None = None, step_penalty: float = 0.0) -> DTWResult:
    """Align the whole reference (rows) against the best-matching portion of the query (columns).

    Free start and end in the query. The reference must be the shorter sequence for this to be
    meaningful. Note the objective is a plain sum, which slightly favors shorter query spans:
    use max_run and/or step_penalty to keep the matched span from being squeezed or stretched.
    """
    return dtw(cost, band=None, max_run=max_run, open_begin=True, open_end=True, step_penalty=step_penalty)


_NO_PATH = ("No feasible path under these constraints; widen the band, raise max_run, or allow open ends "
            "(the sequences may differ too much in length for the limits chosen)")


def _dp_plain(C, allowed, open_begin, open_end, pen):
    """Unlimited runs. One value per cell; back-pointer codes: -1 start, 0 diag, 1 horiz, 2 vert."""
    n, m = C.shape
    D = np.full((n, m), np.inf)
    bp = np.full((n, m), -1, dtype=np.int8)
    for i in range(n):
        for j in range(m):
            if not allowed[i, j]:
                continue
            best, code = np.inf, -1
            if i == 0 and (j == 0 or open_begin):
                best = 0.0
            if i > 0 and j > 0 and D[i - 1, j - 1] < best:
                best, code = D[i - 1, j - 1], 0
            if j > 0 and D[i, j - 1] + pen < best:
                best, code = D[i, j - 1] + pen, 1
            if i > 0 and D[i - 1, j] + pen < best:
                best, code = D[i - 1, j] + pen, 2
            if np.isfinite(best):
                D[i, j] = best + C[i, j]
                bp[i, j] = code
    if open_end:
        j = int(np.argmin(D[n - 1]))
    else:
        j = m - 1
    objective = D[n - 1, j]
    if not np.isfinite(objective):
        raise ValueError(_NO_PATH)
    i = n - 1
    cells = []
    while True:
        cells.append((i, j))
        code = bp[i, j]
        if code < 0:
            break
        if code == 0:
            i, j = i - 1, j - 1
        elif code == 1:
            j -= 1
        else:
            i -= 1
    return np.array(cells[::-1], dtype=np.int64), objective


def _dp_runs(C, allowed, R, open_begin, open_end, pen):
    """Run-limited DP. Each cell keeps S = 2R + 1 states:

        state 0          reached by a diagonal step (or the start)
        state r (1..R)   reached by a horizontal step, the r-th in a row
        state R + r      reached by a vertical step, the r-th in a row

    A run of R may not be extended. Every row is computed with vectorized operations over
    the columns, so cost is O(n * m * R) numpy work with only n Python iterations.
    """
    n, m = C.shape
    S = 2 * R + 1
    D = np.full((n, m, S), np.inf)
    BP = np.full((n, m, S), -1, dtype=np.int16)    # predecessor state at the predecessor cell
    non_h = np.concatenate(([0], np.arange(R + 1, S)))   # states a horizontal run may start from
    non_v = np.arange(0, R + 1)                          # states a vertical run may start from
    cols = np.arange(m)
    for i in range(n):
        cur, bp, c, ok = D[i], BP[i], C[i], allowed[i]
        if i == 0:
            start = cols if open_begin else np.array([0])
            cur[start, 0] = c[start]
        else:
            prev = D[i - 1]
            # diagonal: from any state of (i-1, j-1)
            pm, pa = prev.min(axis=1), prev.argmin(axis=1)
            cur[1:, 0] = pm[:-1] + c[1:]
            bp[1:, 0] = pa[:-1]
            # vertical run 1: from the diagonal or any horizontal state of (i-1, j)
            sub = prev[:, non_v]
            k = sub.argmin(axis=1)
            cur[:, R + 1] = sub[cols, k] + pen + c
            bp[:, R + 1] = non_v[k]
            # vertical runs 2..R extend the previous run
            if R > 1:
                cur[:, R + 2:] = prev[:, R + 1:S - 1] + pen + c[:, None]
                bp[:, R + 2:] = np.arange(R + 1, S - 1)[None, :]
        cur[~ok] = np.inf
        # horizontal runs depend on this row's diagonal/vertical states, which are final here
        sub = cur[:-1][:, non_h]
        k = sub.argmin(axis=1)
        cur[1:, 1] = sub[np.arange(m - 1), k] + pen + c[1:]
        bp[1:, 1] = non_h[k]
        for r in range(2, R + 1):
            cur[1:, r] = cur[:-1, r - 1] + pen + c[1:]
            bp[1:, r] = r - 1
        cur[~ok] = np.inf

    last = D[n - 1]
    if open_end:
        j, s = divmod(int(np.argmin(last)), S)
    else:
        j, s = m - 1, int(np.argmin(last[m - 1]))
    objective = last[j, s]
    if not np.isfinite(objective):
        raise ValueError(_NO_PATH)
    i = n - 1
    cells = []
    while True:
        cells.append((i, j))
        p = int(BP[i, j, s])
        if p < 0:
            break
        if s == 0:
            i, j = i - 1, j - 1
        elif s <= R:
            j -= 1
        else:
            i -= 1
        s = p
    return np.array(cells[::-1], dtype=np.int64), objective


def align_sequences(
    ref: np.ndarray,
    query: np.ndarray,
    ref_vis: np.ndarray | None = None,
    query_vis: np.ndarray | None = None,
    nan_cost: float = 1.0,
    subsequence: bool = False,
    **dtw_kwargs,
) -> tuple[DTWResult, np.ndarray]:
    """Normalized poses -> (DTWResult, cost matrix). Convenience wrapper around the two steps.

    subsequence=True calls subsequence_dtw (accepts max_run, step_penalty); otherwise dtw
    (accepts band, max_run, open_begin, open_end, step_penalty).
    """
    cost, _ = pairwise_pose_cost(ref, query, ref_vis, query_vis, nan_cost=nan_cost)
    result = subsequence_dtw(cost, **dtw_kwargs) if subsequence else dtw(cost, **dtw_kwargs)
    return result, cost


# --------------------------------------------------------------------------
# Is the alignment stretching too much?
# --------------------------------------------------------------------------
def band_edge_fraction(path: np.ndarray, n: int, m: int, radius: float, margin: float = 2.0) -> float:
    """Share of path cells within `margin` query frames of the Sakoe-Chiba band edge.

    A path pinned to the edge means the band, not the data, decided the timing: the true offset
    is probably larger than the band allows (e.g. the recording started early). check_stretching
    cannot see this because a clamped path can be perfectly smooth. Values above roughly 0.1
    deserve a look; the alignment's normalized cost will usually be poor too.
    """
    path = np.asarray(path)
    slope = (m - 1) / (n - 1) if n > 1 else 0.0
    dist = np.abs(path[:, 1] - path[:, 0] * slope)
    return float((dist >= radius - margin).mean())


def warp_curve(path: np.ndarray, n_ref: int) -> np.ndarray:
    """For each reference frame, the mean query frame it is matched to. Shape (n_ref,)."""
    path = np.asarray(path)
    count = np.bincount(path[:, 0], minlength=n_ref)
    total = np.bincount(path[:, 0], weights=path[:, 1], minlength=n_ref)
    return total / np.maximum(count, 1)


def _longest_run(flags: np.ndarray) -> int:
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return best


@dataclass(frozen=True)
class StretchReport:
    longest_horizontal_run: int     # most consecutive steps where one reference frame repeats
    longest_vertical_run: int       # most consecutive steps where one query frame repeats
    max_query_per_ref: int          # most query frames matched to a single reference frame
    max_ref_per_query: int          # most reference frames matched to a single query frame
    overall_rate: float             # query frames consumed per reference frame, whole path
    local_rate_min: float           # slowest/fastest windowed rate (nan if the path is too short)
    local_rate_max: float
    frac_outside_rate_range: float  # share of windows outside rate_range
    rate_range: tuple[float, float]
    window: int                     # window length in frames
    max_run_limit: int              # run length above which a run is flagged, in frames
    flags: list[str] = field(default_factory=list)

    @property
    def excessive(self) -> bool:
        return bool(self.flags)


def check_stretching(
    path: np.ndarray,
    n_ref: int | None = None,
    fps: float = 30.0,
    window_s: float = 1.0,
    max_run_s: float = 1.0,
    rate_range: tuple[float, float] = (0.5, 2.0),
    max_outside_frac: float = 0.05,
) -> StretchReport:
    """Measure and flag excessive temporal stretching of a DTW path.

    Two views of the same problem:
    * runs: the longest stretch of horizontal or vertical steps. A long run means a stretch of
      one video was squashed onto a few frames of the other (learner paused, skipped a part, or
      the match is wrong). Flagged above max_run_s.
    * rate: over every window of window_s seconds of reference, how many query frames were
      consumed per reference frame. 1.0 = same speed, 2.0 = learner twice as slow. Flagged when
      more than max_outside_frac of windows leave rate_range.

    The defaults are starting points; tune them on real recordings. fps converts seconds to frames.
    """
    path = np.asarray(path)
    if path.ndim != 2 or path.shape[1] != 2 or len(path) < 1:
        raise ValueError("path must have shape (L, 2)")
    n = int(path[:, 0].max()) + 1 if n_ref is None else int(n_ref)
    steps = np.diff(path, axis=0)
    horiz = (steps[:, 0] == 0) & (steps[:, 1] == 1)
    vert = (steps[:, 0] == 1) & (steps[:, 1] == 0)
    h_run, v_run = _longest_run(horiz), _longest_run(vert)
    per_ref = np.bincount(path[:, 0])
    per_query = np.bincount(path[:, 1])

    qm = warp_curve(path, n)
    span_q = float(path[-1, 1] - path[0, 1] + 1)
    overall = span_q / n
    window = max(1, int(round(window_s * fps)))
    limit = max(1, int(round(max_run_s * fps)))
    if n > window:
        rates = (qm[window:] - qm[:-window]) / window
        outside = (rates < rate_range[0]) | (rates > rate_range[1])
        r_min, r_max, frac = float(rates.min()), float(rates.max()), float(outside.mean())
    else:
        r_min = r_max = float("nan")
        frac = 0.0

    flags = []
    if h_run > limit:
        flags.append(f"horizontal run {h_run} frames (> {limit}): reference frames held while the learner moves on")
    if v_run > limit:
        flags.append(f"vertical run {v_run} frames (> {limit}): reference frames squashed onto one learner frame")
    if frac > max_outside_frac:
        flags.append(f"{frac:.0%} of {window_s:g} s windows outside rate {rate_range[0]:g} to {rate_range[1]:g}")
    return StretchReport(h_run, v_run, int(per_ref.max()), int(per_query.max()), overall,
                         r_min, r_max, frac, tuple(rate_range), window, limit, flags)