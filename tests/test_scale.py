"""Tests for step 2 (smoothed body scale). Run with: pytest tests/test_scale.py -v"""

import numpy as np
import pytest

import scale_benchmark as sb
from pose_utils import (
    L_HIP, R_HIP, L_SHOULDER, R_SHOULDER,
    StreamingBodyScale, center_pose, estimate_body_scale, normalize_pose,
    normalize_sequence, rotate_yaw, scale_pose, torso_length, window_from_seconds,
)

RIGHT_WRIST = 16
FPS = 30.0


@pytest.fixture
def base_pose():
    """Asymmetric synthetic pose: hips at origin, torso length 0.5, person's left at +x, y down."""
    rng = np.random.default_rng(0)
    p = rng.normal(0.0, 0.3, (33, 3))
    p[L_HIP], p[R_HIP] = (0.1, 0.0, 0.0), (-0.1, 0.0, 0.0)
    p[L_SHOULDER], p[R_SHOULDER] = (0.2, -0.5, 0.0), (-0.2, -0.5, 0.0)
    return p


def bent_sequence(pose, T=200, start=70, length=60, depth=0.30, ramp=5):
    """Static pose whose torso shortens (shoulders move toward the hips) during one long bend.

    Only the torso changes: every other joint, including the right wrist, never moves.
    """
    t = np.arange(T, dtype=float)
    d = depth * np.minimum(np.clip((t - start) / ramp, 0, 1), np.clip((start + length - t) / ramp, 0, 1))
    seq = np.repeat(pose[None], T, axis=0)
    hip_mid = (pose[L_HIP] + pose[R_HIP]) / 2
    for j in (L_SHOULDER, R_SHOULDER):
        seq[:, j] = hip_mid + (1 - d)[:, None] * (pose[j] - hip_mid)
    return seq, d


# ------------------------------------------------------------ torso length
def test_torso_length_known_value_and_invariances(base_pose):
    assert torso_length(base_pose) == pytest.approx(0.5)
    moved = rotate_yaw(base_pose, 0.9) + np.array([3.0, -1.0, 2.0])
    assert torso_length(moved) == pytest.approx(0.5)                        # translation and rotation
    assert torso_length(base_pose * 2.4) == pytest.approx(1.2)              # scales linearly
    assert torso_length(np.stack([base_pose, base_pose * 2])).shape == (2,)


def test_torso_length_missing_and_low_visibility(base_pose):
    q = base_pose.copy()
    q[R_SHOULDER] = np.nan
    assert np.isnan(torso_length(q))
    assert np.isnan(torso_length(np.zeros((33, 3))))                        # zero length

    vis = np.ones(33)
    vis[L_HIP] = 0.2
    assert np.isnan(torso_length(base_pose, vis=vis))
    vis[L_HIP] = 0.9
    assert torso_length(base_pose, vis=vis) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        torso_length(base_pose, vis=np.ones(10))


def test_window_from_seconds_is_odd_and_frame_rate_aware():
    assert window_from_seconds(3.0, 30) == 91
    assert window_from_seconds(3.0, 60) == 181                              # same duration, more frames
    assert window_from_seconds(0.01, 30) == 3                               # never degenerate
    assert all(window_from_seconds(s, f) % 2 == 1 for s in (0.5, 1, 2.5, 3) for f in (24, 25, 30, 60))
    with pytest.raises(ValueError):
        window_from_seconds(0, 30)


# ------------------------------------------------------- estimate_body_scale
@pytest.mark.parametrize("kwargs", [dict(), dict(causal=True), dict(window=None)])
def test_constant_lengths_give_that_constant(kwargs):
    est = estimate_body_scale(np.full(120, 0.5), **kwargs)
    assert est.shape == (120,)
    start = 4 if kwargs.get("causal") else 0                                # causal needs 5 frames first
    assert np.isnan(est[:start]).all() and np.allclose(est[start:], 0.5)


def test_long_window_ignores_a_bend_short_window_follows_it():
    L = np.full(300, 0.5)
    L[100:160] *= 0.7                                                       # 2-second bow, 30% shorter
    assert np.abs(estimate_body_scale(L, window=91) - 0.5).max() < 0.01
    assert np.abs(estimate_body_scale(L, window=31) - 0.5).max() > 0.1      # window shorter than the bend
    assert np.abs(L - 0.5).max() > 0.1                                      # whereas per-frame is fooled


def test_global_mode_is_one_constant_and_high_percentile_resists_bending():
    L = np.full(300, 0.5)
    L[50:230] *= 0.8                                                        # bent 60% of the time
    g = estimate_body_scale(L, window=None, percentile=75)
    assert np.all(g == g[0]) and g[0] == pytest.approx(0.5)
    assert estimate_body_scale(L, window=None, percentile=50)[0] < 0.45     # the median is dragged down


def test_gaps_centred_interpolates_causal_holds_and_starts_empty():
    L = np.full(100, 0.5)
    L[40:80] = np.nan
    assert np.isfinite(estimate_body_scale(L, window=9)).all()              # window fully inside the gap -> filled
    causal = estimate_body_scale(L, window=9, causal=True)
    assert np.allclose(causal[44:], 0.5, atol=0) and np.isfinite(causal[4:]).all()
    assert np.isnan(causal[:4]).all()                                       # needs min_count=5 valid frames


def test_too_little_data_raises_instead_of_returning_nans():
    with pytest.raises(ValueError, match="valid torso measurements"):
        estimate_body_scale(np.array([0.5, np.nan, 0.5]))
    scattered = np.full(300, np.nan)
    scattered[[0, 70, 140, 210, 280]] = 0.5                                 # 5 valid, but never within one window
    with pytest.raises(ValueError, match="No window"):
        estimate_body_scale(scattered, window=31)


def test_parameter_validation():
    L = np.full(50, 0.5)
    for bad in (dict(percentile=0), dict(percentile=101), dict(window=0), dict(min_count=0)):
        with pytest.raises(ValueError):
            estimate_body_scale(L, **bad)
    with pytest.raises(ValueError):
        estimate_body_scale(np.ones((3, 3)))


@pytest.mark.parametrize("scenario", sb.SCENARIOS)
def test_matches_the_benchmark_baselines(scenario):
    L, _ = sb.make_scenario(scenario, seed=1)
    assert np.allclose(estimate_body_scale(L, window=91, percentile=75),
                       sb.rolling_percentile(91, 75)(L))
    ours = estimate_body_scale(L, window=91, percentile=75, causal=True)
    theirs = sb.rolling_percentile(91, 75, causal=True)(L)
    assert np.allclose(ours, theirs, equal_nan=True)


def test_causal_mode_never_uses_future_frames():
    base, _ = sb.make_scenario("noise_only")
    outage = base.copy()
    outage[100:260] = np.nan                                                # longer than the window
    for L, cut in ((base, 200), (outage, 261), (base, 3)):
        changed = L.copy()
        changed[cut + 1:] += 1.0
        a = estimate_body_scale(L, window=91, causal=True, min_count=1 if cut == 3 else 5)
        b = estimate_body_scale(changed, window=91, causal=True, min_count=1 if cut == 3 else 5)
        assert np.allclose(a[:cut + 1], b[:cut + 1], equal_nan=True), cut


# --------------------------------------------------------- streaming (live)
@pytest.mark.parametrize("scenario", sb.SCENARIOS)
def test_streaming_matches_batch_causal(scenario):
    L, _ = sb.make_scenario(scenario, seed=2)
    scaler = StreamingBodyScale(window=91, percentile=75)
    streamed = np.array([scaler.update(v) for v in L])
    assert np.allclose(streamed, estimate_body_scale(L, window=91, causal=True), equal_nan=True)


def test_streaming_warmup_holds_through_dropouts_and_resets():
    s = StreamingBodyScale(window=11, min_count=3)
    assert np.isnan([s.update(0.5), s.update(0.5)]).all()                   # not enough data yet
    assert s.update(0.5) == pytest.approx(0.5)
    for _ in range(30):
        assert s.update(np.nan) == pytest.approx(0.5)                       # dropout: hold the last estimate
    s.reset()
    assert np.isnan(s.update(0.5))
    with pytest.raises(ValueError):
        StreamingBodyScale(window=0)


# ---------------------------------------------------------------- scale_pose
def test_scale_pose_scalar_per_frame_and_invalid(base_pose):
    c = center_pose(base_pose)
    assert np.allclose(scale_pose(c, 0.5), c / 0.5)
    seq = np.stack([c, c, c])
    out = scale_pose(seq, np.array([0.5, 1.0, np.nan]))
    assert np.allclose(out[0], c / 0.5) and np.allclose(out[1], c) and np.isnan(out[2]).all()
    assert np.isnan(scale_pose(c, 0.0)).all()
    with pytest.raises(ValueError):
        scale_pose(seq, np.ones(5))


def test_normalize_pose_with_explicit_scale(base_pose):
    c = center_pose(base_pose)
    assert np.allclose(normalize_pose(base_pose, scale=0.25), c / 0.25)     # ignores its own torso
    assert np.allclose(normalize_pose(base_pose), c / 0.5)                  # default unchanged
    assert np.isnan(normalize_pose(base_pose, scale=np.nan)).all()


# ------------------------------------------------------- normalize_sequence
WINDOW = window_from_seconds(3.0, FPS)                                      # 91 frames at 30 fps


def test_normalize_sequence_is_the_documented_composition(base_pose):
    seq, _ = bent_sequence(base_pose)
    expected = scale_pose(center_pose(seq), estimate_body_scale(torso_length(seq), window=WINDOW))
    assert np.allclose(normalize_sequence(seq, window=WINDOW), expected)


def test_window_none_gives_one_global_constant_and_shapes_are_validated(base_pose):
    seq, _ = bent_sequence(base_pose)
    out = normalize_sequence(seq, window=None)
    assert out.shape == (200, 33, 3)
    with pytest.raises(ValueError):
        normalize_sequence(seq[0])                                          # a single pose is not a sequence
    with pytest.raises(ValueError, match="valid torso measurements"):
        normalize_sequence(seq[:2])                                         # too little data to estimate a scale
    assert normalize_sequence(seq[:2], min_count=1).shape == (2, 33, 3)


def test_smoothed_scaling_is_invariant_to_body_size(base_pose):
    seq, _ = bent_sequence(base_pose)
    assert np.allclose(normalize_sequence(seq, window=WINDOW), normalize_sequence(seq * 1.7, window=WINDOW))


def test_bending_does_not_distort_joints_that_did_not_move(base_pose):
    seq, d = bent_sequence(base_pose)
    assert (d > 0.25).sum() >= 40                                           # a real, sustained bend

    def wrist_drift(norm):
        return np.abs(norm[:, RIGHT_WRIST] - norm[0, RIGHT_WRIST]).max()

    per_frame = normalize_sequence(seq, window=1, min_count=1)              # a 1-frame window IS per-frame scaling
    smoothed = normalize_sequence(seq, window=WINDOW)
    glob = normalize_sequence(seq, window=None)
    assert wrist_drift(per_frame) > 0.2                                     # the wrist "moves" only because of the scale
    assert wrist_drift(smoothed) < 0.02
    assert wrist_drift(glob) < 0.02


def test_window_is_in_frames_so_use_window_from_seconds(base_pose):
    """The default is 91 frames: 3 s at 30 fps but only 1.5 s at 60 fps, shorter than a 2 s bow."""
    L = np.full(400, 0.5)
    L[150:270] *= 0.7                                                       # a 2-second bow filmed at 60 fps
    default = estimate_body_scale(L)                                        # 91 frames = 1.5 s
    seconds = estimate_body_scale(L, window=window_from_seconds(3.0, 60.0)) # 181 frames = 3 s
    assert np.abs(default - 0.5).max() > 0.1
    assert np.abs(seconds - 0.5).max() < 0.01


def test_low_visibility_frames_are_left_out_of_the_scale_estimate(base_pose):
    T = 100
    seq = np.repeat(base_pose[None], T, axis=0)
    seq[30:90, L_SHOULDER] *= 3.0                                           # garbage torso in 60 frames
    seq[30:90, R_SHOULDER] *= 3.0
    vis = np.ones((T, 33))
    vis[30:90, [L_SHOULDER, R_SHOULDER]] = 0.1

    with_vis = normalize_sequence(seq, window=None, percentile=50, vis=vis)
    without = normalize_sequence(seq, window=None, percentile=50)
    assert np.allclose(with_vis[0], normalize_pose(base_pose))              # correct scale of 0.5
    assert not np.allclose(without[0], with_vis[0], atol=0.05)              # garbage frames dragged it off


def test_live_path_equals_offline_causal_path(base_pose):
    """Frame-by-frame streaming + normalize_pose must equal normalize_sequence(causal=True)."""
    seq, _ = bent_sequence(base_pose)
    seq = seq + np.random.default_rng(5).normal(0, 0.003, seq.shape)        # some tracking noise
    window = window_from_seconds(3.0, FPS)

    offline = normalize_sequence(seq, causal=True, window=window)
    scaler = StreamingBodyScale(window=window)
    live = np.stack([normalize_pose(f, scale=scaler.update(torso_length(f))) for f in seq])
    assert np.allclose(offline, live, equal_nan=True)
    assert np.isnan(live[:4]).all() and np.isfinite(live[4:]).all()         # warm-up, then usable