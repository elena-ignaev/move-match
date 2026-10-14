"""Tests for pose_dtw. Run with pytest, or simply: python test_pose_dtw.py"""

from __future__ import annotations

import itertools
import sys

import numpy as np

from pose_dtw import (band_edge_fraction, band_radius_from_seconds, check_stretching, dtw,
                      pairwise_pose_cost, sakoe_chiba_mask, subsequence_dtw, warp_curve)
from pose_utils import NUM_LANDMARKS, pose_distance
from synthetic_dance import (FPS, Choreography, make_performance, mean_path_error, tau_from_knots,
                             to_normalized)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def longest_runs(path):
    steps = np.diff(path, axis=0)
    out = []
    for kind in ((0, 1), (1, 0)):
        best = cur = 0
        for s in steps:
            cur = cur + 1 if tuple(s) == kind else 0
            best = max(best, cur)
        out.append(best)
    return tuple(out)


def assert_valid_path(path, n, m, open_begin=False, open_end=False):
    assert path[0, 0] == 0 and path[-1, 0] == n - 1
    if not open_begin:
        assert path[0, 1] == 0
    if not open_end:
        assert path[-1, 1] == m - 1
    steps = {tuple(s) for s in np.diff(path, axis=0)}
    assert steps <= {(1, 1), (0, 1), (1, 0)}, steps


def brute_force(C, band=None, max_run=None, open_begin=False, open_end=False, pen=0.0):
    """Minimum objective over ALL valid paths (exponential: tiny matrices only)."""
    n, m = C.shape
    allowed = np.ones((n, m), bool) if band is None else sakoe_chiba_mask(n, m, band)
    best = [np.inf]

    def rec(i, j, total, last, run):
        if i == n - 1 and (j == m - 1 or open_end):
            best[0] = min(best[0], total)
        for di, dj, kind in ((1, 1, 0), (0, 1, 1), (1, 0, 2)):
            ni, nj = i + di, j + dj
            if ni >= n or nj >= m or not allowed[ni, nj]:
                continue
            nrun = (run + 1 if kind == last else 1) if kind else 0
            if max_run is not None and kind and nrun > max_run:
                continue
            rec(ni, nj, total + C[ni, nj] + (pen if kind else 0.0), kind, nrun)

    for j0 in (range(m) if open_begin else [0]):
        if allowed[0, j0]:
            rec(0, j0, C[0, j0], 0, 0)
    return best[0]


def random_poses(rng, t, nan_frames=()):
    p = rng.normal(0, 1, (t, NUM_LANDMARKS, 3))
    for f in nan_frames:
        p[f] = np.nan
    return p


# --------------------------------------------------------------------------
# frame cost
# --------------------------------------------------------------------------
def test_pairwise_cost_matches_pose_distance():
    rng = np.random.default_rng(0)
    a, b = random_poses(rng, 5), random_poses(rng, 7)
    a[1, 13] = np.nan                                   # a missing joint
    va, vb = rng.random((5, 33)), rng.random((7, 33))
    cost, conf = pairwise_pose_cost(a, b, va, vb, nan_cost=None, chunk=2)
    for i, j in itertools.product(range(5), range(7)):
        d, c = pose_distance(a[i], b[j], va[i], vb[j])
        assert np.isclose(cost[i, j], d) and np.isclose(conf[i, j], c)


def test_pairwise_cost_nan_handling():
    rng = np.random.default_rng(1)
    a, b = random_poses(rng, 3, nan_frames=[1]), random_poses(rng, 4)
    raw, _ = pairwise_pose_cost(a, b, nan_cost=None)
    assert np.isnan(raw[1]).all() and np.isfinite(raw[[0, 2]]).all()
    filled, _ = pairwise_pose_cost(a, b, nan_cost=1.0)
    assert (filled[1] == 1.0).all() and np.isfinite(filled).all()


# --------------------------------------------------------------------------
# constraints
# --------------------------------------------------------------------------
def test_identical_sequences_follow_the_diagonal():
    x = np.cumsum(np.random.default_rng(2).normal(size=(40, 1)), axis=0)
    C = np.abs(x - x.T)
    for kw in ({}, dict(band=3), dict(max_run=2), dict(band=3, max_run=2)):
        r = dtw(C, **kw)
        assert r.cost == 0.0 and (r.path[:, 0] == r.path[:, 1]).all() and len(r.path) == 40


def test_sakoe_chiba_mask_properties():
    mask = sakoe_chiba_mask(10, 10, 0)
    assert (mask == np.eye(10, dtype=bool)).all()
    mask = sakoe_chiba_mask(10, 25, 3)                  # slanted diagonal reaches both corners
    assert mask[0, 0] and mask[9, 24]
    assert not mask[0, 10] and not mask[9, 0]
    assert sakoe_chiba_mask(20, 20, 4).sum() > sakoe_chiba_mask(20, 20, 2).sum()
    assert band_radius_from_seconds(2.0, 30) == 60


def test_path_stays_inside_the_band():
    rng = np.random.default_rng(3)
    C = rng.random((30, 36))
    r = dtw(C, band=5)
    mask = sakoe_chiba_mask(30, 36, 5)
    assert mask[r.path[:, 0], r.path[:, 1]].all()
    assert_valid_path(r.path, 30, 36)


def two_valley_cost(n=80, offset=30, shallow=0.0, deep=-0.0):
    """Cost with the true (diagonal) valley and a slightly cheaper decoy valley `offset` frames away."""
    i = np.arange(n)[:, None]
    j = np.arange(n + offset)[None, :]
    C = np.ones((n, n + offset))
    C[np.abs(j - i) <= 1] = 0.30                        # honest timing
    C[np.abs(j - i - offset) <= 1] = 0.20               # decoy: repeated phrase, better match, wrong time
    return C


def test_band_rejects_the_decoy_that_unconstrained_dtw_picks():
    C = two_valley_cost()
    plain = dtw(C, open_end=True, open_begin=True)      # free to pick either valley
    assert abs(plain.query_start - 30) <= 1             # takes the decoy
    banded = dtw(two_valley_cost(offset=0), band=5)     # same shape without the length mismatch
    assert (banded.path[:, 0] == banded.path[:, 1]).all()
    # square matrix with both valleys: the band keeps the path on the honest diagonal
    n = 80
    i = np.arange(n)[:, None]
    j = np.arange(n)[None, :]
    D = np.ones((n, n))
    D[np.abs(j - i) <= 1] = 0.30
    D[np.abs(j - i + 40) <= 1] = 0.20                   # decoy valley 40 frames off the diagonal
    free, banded = dtw(D), dtw(D, band=10)
    assert np.abs(banded.path[:, 1] - banded.path[:, 0]).max() <= 10
    assert free.cost <= banded.cost


def test_infeasible_constraints_raise():
    C = np.random.default_rng(4).random((5, 40))
    for kw in (dict(band=1), dict(max_run=2)):          # cannot cover an 8x length ratio
        try:
            dtw(C, **kw)
        except ValueError as e:
            assert "feasible" in str(e)
        else:
            raise AssertionError(f"expected ValueError for {kw}")


def test_run_limit_is_respected_and_binds():
    n, m = 60, 21
    C = np.ones((n, m))
    C[:, m - 1] = 0.0                                   # only the last query frame is a good match, for every reference frame
    free = dtw(C)
    assert longest_runs(free.path)[1] >= 30             # unconstrained: one long vertical run down that column
    for limit in (2, 5, 8):                             # (1 is infeasible here: 39+ vertical steps, room for ~21)
        r = dtw(C, max_run=limit)
        assert max(longest_runs(r.path)) <= limit
        assert_valid_path(r.path, n, m)
        assert r.cost >= free.cost                      # a constraint can only make the fit worse or equal
    assert dtw(C, max_run=2).cost > free.cost           # and here it strictly does


def test_dp_matches_brute_force_for_every_constraint():
    rng = np.random.default_rng(5)
    configs = [dict(), dict(max_run=1), dict(max_run=2), dict(band=1), dict(band=2, max_run=2),
               dict(open_begin=True), dict(open_end=True), dict(open_begin=True, open_end=True, max_run=2),
               dict(step_penalty=0.3), dict(max_run=2, step_penalty=0.2, open_end=True)]
    checked = 0
    for seed in range(12):
        n, m = int(rng.integers(3, 6)), int(rng.integers(3, 7))
        C = rng.random((n, m))
        for cfg in configs:
            truth = brute_force(C, cfg.get("band"), cfg.get("max_run"), cfg.get("open_begin", False),
                                cfg.get("open_end", False), cfg.get("step_penalty", 0.0))
            if not np.isfinite(truth):
                try:
                    dtw(C, **cfg)
                except ValueError:
                    continue
                raise AssertionError(f"expected infeasible: {cfg} {C.shape}")
            r = dtw(C, **cfg)
            assert np.isclose(r.objective, truth), (cfg, C.shape, r.objective, truth)
            assert_valid_path(r.path, n, m, cfg.get("open_begin", False), cfg.get("open_end", False))
            if "max_run" in cfg:
                assert max(longest_runs(r.path)) <= cfg["max_run"]
            checked += 1
    assert checked > 80


def test_loose_run_limit_equals_unlimited():
    rng = np.random.default_rng(6)
    C = rng.random((14, 17))
    base = dtw(C)
    loose = dtw(C, max_run=40)
    assert np.isclose(base.objective, loose.objective)
    for kw in (dict(open_begin=True, open_end=True), dict(band=6)):
        assert np.isclose(dtw(C, **kw).objective, dtw(C, max_run=40, **kw).objective)


def test_step_penalty_reduces_non_diagonal_steps():
    rng = np.random.default_rng(7)
    C = rng.random((40, 50))
    def n_off(r):
        return int((np.abs(np.diff(r.path, axis=0)).sum(axis=1) == 1).sum())
    assert n_off(dtw(C, step_penalty=1.0)) <= n_off(dtw(C))


def test_argument_validation():
    C = np.random.default_rng(8).random((6, 6))
    bad = [dict(band=2, open_end=True), dict(max_run=0), dict(max_run=1.5), dict(step_penalty=-1.0)]
    for kw in bad:
        try:
            dtw(C, **kw)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {kw}")
    for matrix in (np.full((3, 3), np.nan), -np.ones((3, 3)), np.ones(5), np.ones((0, 3))):
        try:
            dtw(matrix)
        except ValueError:
            continue
        raise AssertionError("expected ValueError for bad cost matrix")


# --------------------------------------------------------------------------
# subsequence
# --------------------------------------------------------------------------
def test_subsequence_finds_an_embedded_pattern():
    rng = np.random.default_rng(9)
    pattern = np.cumsum(rng.normal(size=(50, 2)), axis=0)
    recording = np.concatenate([rng.normal(10, 3, (70, 2)), pattern, rng.normal(-10, 3, (45, 2))])
    C = np.linalg.norm(pattern[:, None] - recording[None], axis=-1)
    r = subsequence_dtw(C)
    assert (r.query_start, r.query_end) == (70, 119) and r.cost < 1e-9
    assert_valid_path(r.path, 50, len(recording), open_begin=True, open_end=True)


# --------------------------------------------------------------------------
# stretch diagnostics
# --------------------------------------------------------------------------
def test_warp_curve_and_stretch_report_on_handmade_paths():
    n = 200
    diag = np.stack([np.arange(n), np.arange(n)], axis=1)
    rep = check_stretching(diag, fps=30)
    assert not rep.excessive and rep.longest_horizontal_run == rep.longest_vertical_run == 0
    assert np.isclose(rep.overall_rate, 1.0) and np.isclose(rep.local_rate_min, 1.0)

    # squash 60 reference frames onto one query frame (the first of them is entered diagonally: run of 59)
    squash = np.concatenate([diag[:70], [(i, 70) for i in range(70, 130)], [(i, i - 59) for i in range(130, n)]])
    squash = np.asarray(squash)
    rep = check_stretching(squash, n_ref=n, fps=30)
    assert rep.longest_vertical_run == 59 and rep.max_ref_per_query == 60
    assert rep.excessive and any("vertical run" in f for f in rep.flags)

    # hold one reference frame for 50 query frames (run of 49)
    hold = np.concatenate([diag[:50], [(50, j) for j in range(50, 100)], [(i, i + 50) for i in range(51, n)]])
    rep = check_stretching(hold, n_ref=n, fps=30)
    assert rep.longest_horizontal_run == 49 and rep.max_query_per_ref == 50 and rep.excessive

    # uniform half speed: no runs, but every window is outside rate (0.5, 2) once slower than 2x
    slow = np.stack([np.arange(n), np.minimum(3 * np.arange(n), 3 * n)], axis=1)  # not a legal path, rate only
    qm = warp_curve(slow, n)
    assert np.allclose(qm, slow[:, 1])


def test_band_edge_fraction():
    n = 100
    diag = np.stack([np.arange(n), np.arange(n)], axis=1)
    assert band_edge_fraction(diag, n, n, 10) == 0.0
    pinned = np.stack([np.arange(n), np.minimum(np.arange(n) + 10, n - 1)], axis=1)
    assert band_edge_fraction(pinned, n, n, 10) > 0.8


# --------------------------------------------------------------------------
# end to end on synthetic dances with a known warp
# --------------------------------------------------------------------------
def _world(seed=0, holds=((8.0, 4.0),)):
    choreo = Choreography(seed=seed, motion=0.5, holds=holds)
    ref_xyz, ref_vis = make_performance(choreo, np.arange(600) / FPS, noise=0.005, dropout=0.0, seed=10)
    return choreo, to_normalized(ref_xyz, ref_vis)


def _learner(choreo, tau, seed):
    xyz, vis = make_performance(choreo, tau, noise=0.03, dropout=0.03, seed=seed)
    return to_normalized(xyz, vis), tau * FPS


def test_full_performance_run_limit_spreads_the_stretch_of_a_hold():
    choreo, ref = _world()
    tau = tau_from_knots([0, 0, 5, 8, 12, 16, 20, 20], [0, 0.5, 5.6, 9.0, 11.5, 15.9, 20.6, 21.1])
    q, tf = _learner(choreo, tau, 11)
    cost, _ = pairwise_pose_cost(ref, q)
    radius = band_radius_from_seconds(2.0, FPS)
    free = dtw(cost, band=radius)
    held = dtw(cost, band=radius, max_run=8)
    rf, rh = check_stretching(free.path, 600, FPS), check_stretching(held.path, 600, FPS)
    assert rf.excessive and rf.longest_vertical_run > 30           # the hold collapsed onto a few frames
    assert not rh.excessive and max(longest_runs(held.path)) <= 8
    assert mean_path_error(held.path, tf) < mean_path_error(free.path, tf)
    assert mean_path_error(held.path, tf) < 3.0                    # recovered the warp within 0.1 s
    assert held.normalized_cost <= free.normalized_cost * 1.05     # constraint barely costs fit
    assert band_edge_fraction(held.path, *cost.shape, radius) < 0.05


def test_skipped_segment_is_flagged_and_band_cannot_follow():
    choreo, ref = _world()
    tau = np.concatenate([np.zeros(15), np.arange(0, 3, 1 / FPS), np.arange(7, 20, 1 / FPS), np.full(15, 20.0)])
    q, tf = _learner(choreo, tau, 12)
    cost, _ = pairwise_pose_cost(ref, q)
    free = dtw(cost)
    rf = check_stretching(free.path, 600, FPS)
    assert rf.excessive and rf.longest_vertical_run > 60           # a 4 s skip squashed onto a few frames
    tight = dtw(cost, band=band_radius_from_seconds(2.0, FPS), max_run=8)
    assert tight.normalized_cost > free.normalized_cost            # the constraint visibly costs fit
    assert check_stretching(tight.path, 600, FPS).excessive        # still flagged (rate), so it is not hidden


def test_subsequence_locates_a_segment_in_a_longer_recording():
    choreo, ref = _world()
    tau = tau_from_knots([0, 0, 5, 8, 12, 16, 20, 20], [0, 0.5, 5.6, 9.0, 11.5, 15.9, 20.6, 21.1])
    q, tf = _learner(choreo, tau, 11)
    s0, s1 = 6 * FPS, 14 * FPS
    cost, _ = pairwise_pose_cost(ref[s0:s1], q)
    r = subsequence_dtw(cost, max_run=8)
    true_start, true_end = int(np.argmin(np.abs(tf - s0))), int(np.argmin(np.abs(tf - s1)))
    assert abs(r.query_start - true_start) <= 10 and abs(r.query_end - true_end) <= 15
    assert mean_path_error(r.path, tf, ref_offset=s0) < 5.0


def test_early_start_needs_subsequence_not_a_fixed_start_band():
    choreo, ref = _world()
    tau = np.concatenate([np.zeros(6 * FPS),
                          tau_from_knots([0, 0, 5, 8, 12, 16, 20, 20], [0, 0.5, 5.6, 9.0, 11.5, 15.9, 20.6, 21.1])])
    q, tf = _learner(choreo, tau, 13)
    cost, _ = pairwise_pose_cost(ref, q)
    radius = band_radius_from_seconds(2.0, FPS)
    sub = subsequence_dtw(cost, max_run=8)
    fixed = dtw(cost, band=radius, max_run=8)
    assert abs(sub.query_start - 6 * FPS) <= 25                    # idle frames are ambiguous, so a loose bound
    assert mean_path_error(sub.path, tf) < 3.0
    assert fixed.normalized_cost > 2 * sub.normalized_cost         # the wrong tool fits far worse
    assert band_edge_fraction(fixed.path, *cost.shape, radius) > 0.1   # and is visibly pinned to the band edge


# --------------------------------------------------------------------------
if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:                          # noqa: BLE001 - report every failure, then exit non-zero
            failed += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)