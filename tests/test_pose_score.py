"""Tests for pose_score. Run with pytest, or: python test_pose_score.py"""

from __future__ import annotations

import sys

import numpy as np

from calibrate_pose_k import simulate
from pose_dtw import dtw
from pose_score import (calibrate_k, distance_to_score, per_reference_distance, reference_motion_scale,
                        score_alignment, score_performance, score_to_distance, summarize_alignment)


def raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False


# --------------------------------------------------------------------------
# distance <-> score
# --------------------------------------------------------------------------
def test_distance_to_score_basic_properties():
    assert distance_to_score(0.0, 2.0) == 100.0
    d = np.linspace(0, 3, 50)
    s = distance_to_score(d, 1.5)
    assert (np.diff(s) < 0).all() and s.min() > 0
    assert np.isnan(distance_to_score(np.nan, 1.0))
    assert np.allclose(distance_to_score(d, 1.0)[1:], 100 * np.exp(-d[1:]))
    assert np.isclose(distance_to_score(0.5, 3.0), 100 * np.exp(-1.5))


def test_score_to_distance_inverts():
    for k in (0.5, 2.0, 7.0):
        d = np.array([0.0, 0.1, 0.4, 1.3])
        assert np.allclose(score_to_distance(distance_to_score(d, k), k), d)
    assert raises(lambda: score_to_distance(0.0))
    assert raises(lambda: score_to_distance(101.0))


def test_score_validation():
    assert raises(lambda: distance_to_score(0.1, 0.0))
    assert raises(lambda: distance_to_score(0.1, -1.0))
    assert raises(lambda: distance_to_score(-0.1, 1.0))


# --------------------------------------------------------------------------
# averaging along the path
# --------------------------------------------------------------------------
def test_per_reference_distance_counts_each_reference_frame_once():
    # reference frame 1 is matched to three learner frames (a pause); its cost is 0.9 everywhere
    cost = np.array([[0.1, 0.1, 0.1, 0.1, 0.1],
                     [0.9, 0.9, 0.9, 0.9, 0.9],
                     [0.1, 0.1, 0.1, 0.1, 0.1]])
    path = np.array([[0, 0], [1, 1], [1, 2], [1, 3], [2, 4]])
    per = per_reference_distance(cost, path)
    assert np.allclose(per, [0.1, 0.9, 0.1])
    assert np.isclose(per.mean(), 1.1 / 3)                       # each reference frame counts once
    assert np.isclose(cost[path[:, 0], path[:, 1]].mean(), 0.58)  # the plain path mean overweights the pause (0.58 vs 0.37)


def test_per_reference_distance_ignores_blind_pairs():
    cost = np.full((3, 4), 0.2)
    cost[1, 1] = 1.0                                             # filler value for a blind pair
    usable = np.ones((3, 4), bool)
    usable[1, 1] = False
    path = np.array([[0, 0], [1, 1], [1, 2], [2, 3]])
    per = per_reference_distance(cost, path, usable)
    assert np.allclose(per, [0.2, 0.2, 0.2])                     # the 1.0 never enters the average
    usable[1, 2] = False                                         # now reference frame 1 has no usable pair
    assert np.isnan(per_reference_distance(cost, path, usable)[1])


# --------------------------------------------------------------------------
# alignment statistics
# --------------------------------------------------------------------------
def test_summary_step_counts_add_up():
    rng = np.random.default_rng(0)
    C = rng.random((40, 52))
    r = dtw(C, max_run=3)
    s = summarize_alignment(r)
    assert s.diagonal_steps + s.horizontal_steps + s.vertical_steps == s.path_length - 1
    assert s.horizontal_steps - s.vertical_steps == 52 - 40      # query frames minus reference frames
    assert 0 < s.stretch_fraction < 1 and s.stretch.longest_horizontal_run <= 3
    assert s.band_edge_fraction is None
    assert summarize_alignment(dtw(C, band=20), band=20).band_edge_fraction is not None


def test_identical_sequences_have_no_stretching():
    x = np.cumsum(np.random.default_rng(1).normal(size=(30, 1)), axis=0)
    s = summarize_alignment(dtw(np.abs(x - x.T)))
    assert s.stretch_fraction == 0 and s.path_length == 30 and not s.stretch.excessive


# --------------------------------------------------------------------------
# pose score assembly
# --------------------------------------------------------------------------
def test_score_alignment_outputs():
    x = np.cumsum(np.random.default_rng(2).normal(size=(30, 1)), axis=0)
    C = np.abs(x - x.T)
    res = dtw(C)
    ps = score_alignment(res, C, np.ones_like(C), k=2.0)
    assert ps.score == 100.0 and ps.mean_pose_distance == 0.0 and ps.tracking_coverage == 1.0
    assert (ps.alignment_path == res.path).all() and ps.summary.path_length == 30 and ps.warnings == ()

    noisy = C + 0.3
    ps = score_alignment(dtw(noisy), noisy, np.ones_like(noisy), k=2.0)
    assert np.isclose(ps.mean_pose_distance, 0.3) and np.isclose(ps.score, 100 * np.exp(-0.6))
    assert ps.p90_distance >= ps.mean_pose_distance - 1e-12


def test_blind_frames_lower_coverage_not_the_score():
    C = np.full((40, 40), 0.25)
    conf = np.ones_like(C)
    conf[:, 10:20] = 0.0                                         # learner frames 10..19 had no tracking
    C[:, 10:20] = 1.0                                            # the filler cost DTW saw there
    ps = score_alignment(dtw(C), C, conf, k=2.0)
    assert np.isclose(ps.mean_pose_distance, 0.25)               # filler excluded from the score
    assert 0.6 < ps.tracking_coverage < 1.0
    ps = score_alignment(dtw(C), C, np.where(conf > 0, 1.0, 0.0) * (np.arange(40) < 3), k=2.0)
    assert any("usable tracking" in w for w in ps.warnings)       # low coverage is reported


def test_nothing_usable_gives_nan_and_a_warning():
    C = np.full((10, 10), 0.5)
    ps = score_alignment(dtw(C), C, np.zeros_like(C), k=2.0)
    assert np.isnan(ps.score) and any("nothing to score" in w for w in ps.warnings)


def test_reference_motion_scale():
    rng = np.random.default_rng(3)
    base = rng.normal(size=(1, 33, 3))
    wobble = rng.normal(size=(60, 33, 3))
    still = np.repeat(base, 60, axis=0)
    assert reference_motion_scale(still) < 1e-9
    small, big = reference_motion_scale(base + 0.1 * wobble), reference_motion_scale(base + 0.4 * wobble)
    assert 0 < small < big


# --------------------------------------------------------------------------
# calibrate_k
# --------------------------------------------------------------------------
def toy_levels(scale=1.0):
    return {"good": np.array([0.02, 0.025, 0.03]) * scale, "ok": np.array([0.15, 0.17, 0.2]) * scale,
            "bad": np.array([0.6, 0.65, 0.7]) * scale}


def test_calibrate_k_meets_stated_requirements_when_feasible():
    cal = calibrate_k(toy_levels(), ["good", "ok", "bad"], "good", "bad")
    assert cal.feasible and cal.k_min <= cal.k <= cal.k_max
    assert cal.scores["good"] >= 90 and cal.scores["bad"] <= 30
    assert cal.scores["good"] - cal.scores["ok"] >= 8 and cal.scores["ok"] - cal.scores["bad"] >= 8
    # every k inside the reported range really satisfies the requirements
    for k in np.geomspace(cal.k_min, cal.k_max, 7):
        s = {n: 100 * np.exp(-k * m) for n, m in cal.medians.items()}
        assert s["good"] >= 90 - 1e-6 and s["bad"] <= 30 + 1e-6


def test_calibrate_k_reports_infeasibility_instead_of_hiding_it():
    # good dancers sit at 0.1 and bad ones at only 0.3: 90 and 30 cannot both hold
    lv = {"good": np.array([0.1]), "bad": np.array([0.3])}
    cal = calibrate_k(lv, ["good", "bad"], "good", "bad")
    assert not cal.feasible and np.isnan(cal.k_min) and cal.notes
    assert np.isfinite(cal.k) and cal.k > 0                      # still returns the least-bad compromise


def test_calibrate_k_scales_with_distance_units():
    a = calibrate_k(toy_levels(1.0), ["good", "ok", "bad"], "good", "bad")
    b = calibrate_k(toy_levels(2.0), ["good", "ok", "bad"], "good", "bad")
    assert np.isclose(a.k, 2 * b.k, rtol=0.02)                   # distances twice as large -> k half as large


# --------------------------------------------------------------------------
# end to end on synthetic performances
# --------------------------------------------------------------------------
def test_end_to_end_scores_rank_quality_and_keep_outputs():
    scores = {}
    for level in ("repeat", "decent", "poor", "unrelated"):
        ref, q, vis = simulate("A", level, 0)
        scores[level] = score_performance(ref, q, None, vis, k=2.0)
    s = [scores[l].score for l in ("repeat", "decent", "poor", "unrelated")]
    assert s == sorted(s, reverse=True) and s[0] > 85 and s[-1] < 25
    ps = scores["decent"]
    assert ps.alignment_path.shape[1] == 2 and ps.alignment_path[0, 0] == 0
    assert ps.summary.path_length == len(ps.alignment_path) and ps.tracking_coverage > 0.9
    assert ps.mean_pose_distance > 0 and ps.p90_distance > 0
    assert np.isclose(ps.score, distance_to_score(ps.mean_pose_distance, 2.0))


def test_low_motion_dance_warns_that_standing_still_scores_well():
    ref_a, q_a, vis_a = simulate("A", "repeat", 0)
    ref_b, q_b, vis_b = simulate("B", "repeat", 0)               # dance B barely moves
    a = score_performance(ref_a, q_a, None, vis_a, k=2.0)
    b = score_performance(ref_b, q_b, None, vis_b, k=2.0)
    assert a.still_baseline_score < b.still_baseline_score
    assert not any("standing still" in w for w in a.warnings)
    assert any("standing still" in w for w in b.warnings)


def test_pausing_does_not_reweight_the_score():
    ref, q, vis = simulate("A", "still", 0)                      # learner stands still for the whole dance
    ps = score_performance(ref, q, None, vis, k=2.0)
    assert abs(ps.path_mean_distance - ps.mean_pose_distance) > 0.02   # the two averages genuinely differ here


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:                          # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)