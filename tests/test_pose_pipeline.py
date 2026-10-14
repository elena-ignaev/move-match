"""Tests for the single reference/learner convention. Run with: pytest test_pose_pipeline.py -v"""

import json

import numpy as np
import pytest

import pose_pipeline as pp
import pose_utils as pu

FPS = 30.0
T = 120
LEFT_WRIST, RIGHT_WRIST = 15, 16
OFFSET = np.array([1.5, -0.4, 2.5])          # the learner stands somewhere else...
SIZE = 1.3                                   # ...and is a different size / distance from the camera


@pytest.fixture
def base_pose():
    """Asymmetric synthetic pose: hips at the origin, torso length 0.5, person's left at +x, y down."""
    rng = np.random.default_rng(0)
    p = rng.normal(0.0, 0.3, (33, 3))
    p[pu.L_HIP], p[pu.R_HIP] = (0.1, 0, 0), (-0.1, 0, 0)
    p[pu.L_SHOULDER], p[pu.R_SHOULDER] = (0.2, -0.5, 0), (-0.2, -0.5, 0)
    return p


@pytest.fixture
def reference(base_pose):
    """A dance where only the RIGHT arm moves, so a left/right mix-up cannot go unnoticed."""
    t = np.arange(T)
    s = 0.5 * (1 - np.cos(2 * np.pi * t / T))
    seq = np.repeat(base_pose[None], T, axis=0)
    seq[:, 14, 1] -= 0.3 * s
    seq[:, 16, 1] -= 0.6 * s
    seq[:, 16, 2] += 0.1 * s
    return seq


def mean_distance(a, b):
    return float(np.mean([pu.pose_distance(x, y)[0] for x, y in zip(a, b)]))


def what_the_extractor_sees(reference, capture_mirrored, follow_mode):
    """Simulate a CORRECT learner under a given setup, returning the raw landmark array."""
    physical = pu.mirror_pose(reference) if follow_mode == pp.FOLLOW_MIRROR else reference.copy()
    seen = pu.mirror_pose(physical) if capture_mirrored else physical      # a mirrored recording flips the array
    return seen * SIZE + OFFSET


# ------------------------------------------------------------ the truth table
@pytest.mark.parametrize("capture, follow, expected", [
    (False, "mirror", True), (False, "same_side", False),
    (True, "mirror", False), (True, "same_side", True),
])
def test_mirror_needed_truth_table(capture, follow, expected):
    assert pp.mirror_needed(capture, follow) is expected


def test_mirror_decisions_cannot_be_left_implicit():
    for bad in (None, "yes", 1, 0):
        with pytest.raises(ValueError, match="capture_mirrored"):
            pp.mirror_needed(bad, "mirror")
    with pytest.raises(ValueError, match="follow_mode"):
        pp.mirror_needed(False, "sideways")
    with pytest.raises(TypeError):                                           # keyword-only, no defaults
        pp.prepare_learner(np.zeros((T, 33, 3)), fps=FPS)


# ------------------------------------ the main guarantee, for all four setups
@pytest.mark.parametrize("capture_mirrored", [False, True])
@pytest.mark.parametrize("follow_mode", pp.FOLLOW_MODES)
def test_a_correct_follower_matches_the_reference_in_every_setup(reference, capture_mirrored, follow_mode):
    ref = pp.prepare_reference(reference, fps=FPS)
    seen = what_the_extractor_sees(reference, capture_mirrored, follow_mode)

    right = pp.prepare_learner(seen, fps=FPS, capture_mirrored=capture_mirrored, follow_mode=follow_mode)
    assert mean_distance(ref.pose, right.pose) < 1e-6                        # position and size removed, nothing else

    wrong = pp.prepare_learner(seen, fps=FPS, capture_mirrored=not capture_mirrored, follow_mode=follow_mode)
    assert mean_distance(ref.pose, wrong.pose) > 0.1                         # a wrong flag reverses left and right


def test_reference_is_never_mirrored(reference):
    ref = pp.prepare_reference(reference, fps=FPS)
    assert np.allclose(ref.pose, pu.normalize_sequence(reference, window=pu.window_from_seconds(3.0, FPS)))


# ------------------------------------- translation + scale in, orientation out
def test_standing_elsewhere_and_body_size_do_not_matter(reference):
    a = pp.prepare_reference(reference, fps=FPS)
    b = pp.prepare_reference(reference * 1.7 + OFFSET, fps=FPS)
    assert np.allclose(a.pose, b.pose)
    assert np.allclose(pu.hip_center(a.pose), 0.0, atol=1e-12)


def test_facing_direction_is_preserved(reference):
    turned = pu.rotate_yaw(reference, 0.5)                                   # same dance, filmed 28.6 deg off
    a = pp.prepare_reference(reference, fps=FPS)
    b = pp.prepare_reference(turned, fps=FPS)
    assert mean_distance(a.pose, b.pose) > 0.05                              # the rotation is still in the data
    assert pp.facing_difference_deg(a.pose, b.pose, n_frames=30) == pytest.approx(-28.648, abs=1e-3)


def test_mirroring_flips_the_direction_of_a_turn(reference):
    turned = pu.rotate_yaw(reference, 0.5)
    plain = pp.prepare_learner(turned, fps=FPS, capture_mirrored=False, follow_mode="same_side")
    flipped = pp.prepare_learner(turned, fps=FPS, capture_mirrored=False, follow_mode="mirror")
    assert pp.facing_angle_deg(flipped.pose) == pytest.approx(-pp.facing_angle_deg(plain.pose), abs=1e-6)


def test_mirroring_commutes_with_centring_and_scaling(base_pose):
    raw = base_pose * 1.4 + OFFSET
    assert np.allclose(pu.normalize_pose(pu.mirror_pose(raw)), pu.mirror_pose(pu.normalize_pose(raw)))


# ------------------------------------------------------------ visibility
def test_visibility_follows_the_mirror_and_low_visibility_joints_become_missing(reference):
    vis = np.ones((T, 33))
    vis[:, RIGHT_WRIST] = 0.2
    seen = reference * SIZE + OFFSET

    plain = pp.prepare_learner(seen, vis, fps=FPS, capture_mirrored=False, follow_mode="same_side")
    assert np.isnan(plain.pose[:, RIGHT_WRIST]).all() and np.isfinite(plain.pose[:, LEFT_WRIST]).all()

    flipped = pp.prepare_learner(seen, vis, fps=FPS, capture_mirrored=False, follow_mode="mirror")
    assert np.isnan(flipped.pose[:, LEFT_WRIST]).all()                    # the unreliable wrist moved sides...
    assert (flipped.visibility[:, LEFT_WRIST] == 0.2).all()               # ...and its visibility moved with it


# ------------------------------------------------------------- live path
def test_live_frames_match_the_offline_sequence(reference):
    seen = what_the_extractor_sees(reference, False, "mirror")
    settings = pp.NormalizationSettings(scale="global")
    offline = pp.prepare_learner(seen, fps=FPS, capture_mirrored=False, follow_mode="mirror", settings=settings)

    scale = pu.estimate_body_scale(pu.torso_length(seen), window=None)[0]    # what the live scaler converges to
    live = np.stack([
        pp.prepare_learner_frame(f, scale=scale, capture_mirrored=False, follow_mode="mirror", settings=settings).pose
        for f in seen
    ])
    assert np.allclose(offline.pose, live)


# ------------------------------------------------------------- settings
def test_smoothed_scale_needs_fps_but_global_does_not(reference):
    with pytest.raises(ValueError, match="fps"):
        pp.prepare_reference(reference)
    assert pp.prepare_reference(reference, settings=pp.NormalizationSettings(scale="global")).pose.shape == (T, 33, 3)


def test_window_is_defined_in_seconds_not_frames(reference):
    """The same 3 s window must be 91 frames at 30 fps and 181 at 60 fps."""
    a = pp.prepare_reference(reference, fps=30.0)
    b = pp.prepare_reference(reference, fps=60.0)
    assert a.pose.shape == b.pose.shape                                      # both valid; windows differ internally
    assert pu.window_from_seconds(pp.DEFAULT_SETTINGS.window_s, 30.0) == 91
    assert pu.window_from_seconds(pp.DEFAULT_SETTINGS.window_s, 60.0) == 181


def test_settings_are_validated_serialisable_and_frozen():
    s = pp.NormalizationSettings()
    assert json.loads(json.dumps(s.as_dict()))["convention_version"] == pp.CONVENTION_VERSION
    for bad in (dict(scale="median"), dict(scale="per_frame"), dict(window_s=0), dict(percentile=0), dict(min_visibility=1.5)):
        with pytest.raises(ValueError):
            pp.NormalizationSettings(**bad)
    with pytest.raises(Exception):
        s.percentile = 50.0


# --------------------------------------------------- mirror calibration
def make_images(left_up, right_up, T=60):
    y = np.full((T, 33, 3), 0.8)
    ramp = np.linspace(0.8, 0.3, T)
    if left_up:
        y[:, LEFT_WRIST, 1] = ramp
    if right_up:
        y[:, RIGHT_WRIST, 1] = ramp
    return y


def test_calibration_detects_mirrored_recordings():
    assert pp.detect_capture_mirroring(make_images(left_up=False, right_up=True)) is False   # labels correct
    assert pp.detect_capture_mirroring(make_images(left_up=True, right_up=False)) is True    # mirrored
    assert pp.detect_capture_mirroring(make_images(True, True)) is None                      # both raised: undecided
    assert pp.detect_capture_mirroring(make_images(False, False)) is None
    nan_frames = make_images(False, True)
    nan_frames[::3] = np.nan
    assert pp.detect_capture_mirroring(nan_frames) is False                                  # tolerates dropouts
    assert pp.detect_capture_mirroring(np.full((10, 33, 3), np.nan)) is None


# ------------------------------------------------------- facing helpers
def test_facing_difference_wraps_and_handles_missing(base_pose):
    front = pu.normalize_pose(base_pose)
    assert pp.facing_difference_deg(front, front) == pytest.approx(0.0, abs=1e-9)
    a, b = pu.rotate_yaw(front, np.deg2rad(170)), pu.rotate_yaw(front, np.deg2rad(-170))
    assert abs(pp.facing_difference_deg(a, b)) == pytest.approx(20.0, abs=1e-6)              # not 340
    assert np.isnan(pp.facing_angle_deg(np.full((5, 33, 3), np.nan)))