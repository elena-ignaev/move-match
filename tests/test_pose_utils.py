"""Run with: pytest tests/test_pose_utils.py -v"""

from types import SimpleNamespace

import numpy as np
import pytest

from pose_utils import (
    L_HIP, R_HIP, L_SHOULDER, R_SHOULDER,
    landmarks_to_array, mirror_pose, mirror_visibility,
    normalize_pose, normalize_sequence, pose_distance, rotate_yaw,
)

RIGHT_WRIST = 16


@pytest.fixture
def base_pose():
    """Deterministic, asymmetric synthetic pose in world-like coordinates (metres).

    Hips centred on the origin, torso length 0.5, person's left at +x, y down.
    """
    rng = np.random.default_rng(0)
    p = rng.normal(0.0, 0.3, (33, 3))
    p[L_HIP], p[R_HIP] = (0.1, 0.0, 0.0), (-0.1, 0.0, 0.0)
    p[L_SHOULDER], p[R_SHOULDER] = (0.2, -0.5, 0.0), (-0.2, -0.5, 0.0)
    return p


def dist(a, b, **kw):
    return pose_distance(normalize_pose(a, **kw.pop("norm", {})),
                         normalize_pose(b, **kw.pop("norm", {})), **kw)[0]


def test_identical_poses_have_zero_distance(base_pose):
    assert dist(base_pose, base_pose) == pytest.approx(0.0, abs=1e-12)


def test_translation_invariant(base_pose):
    assert dist(base_pose, base_pose + np.array([3.0, -2.0, 5.0])) == pytest.approx(0.0, abs=1e-9)


def test_scale_invariant(base_pose):
    assert dist(base_pose, base_pose * 1.7) == pytest.approx(0.0, abs=1e-9)


def test_yaw_alignment_removes_body_rotation(base_pose):
    turned = rotate_yaw(base_pose, np.deg2rad(40))
    assert dist(base_pose, turned) > 0.05                                   # off by default
    aligned = pose_distance(normalize_pose(base_pose, align_yaw=True),
                            normalize_pose(turned, align_yaw=True))[0]
    assert aligned == pytest.approx(0.0, abs=1e-9)


def test_different_pose_is_farther_and_grows_with_error(base_pose):
    prev = 0.0
    for shift in (0.1, 0.3, 0.6):
        other = base_pose.copy()
        other[RIGHT_WRIST, 1] -= shift                                      # raise one wrist
        d = dist(base_pose, other)
        assert d > prev
        prev = d


def test_mirror_is_involution_and_changes_asymmetric_pose(base_pose):
    n = normalize_pose(base_pose)
    assert np.allclose(mirror_pose(mirror_pose(n)), n)
    assert pose_distance(n, mirror_pose(n))[0] > 0.1
    vis = np.linspace(0, 1, 33)
    assert np.allclose(mirror_visibility(mirror_visibility(vis)), vis)


def test_mirrored_learner_matches_reference_after_mirroring(base_pose):
    """A learner physically mirroring the reference matches once we mirror them back."""
    ref = normalize_pose(base_pose)
    learner_seen_in_mirror = mirror_pose(ref)
    assert pose_distance(ref, mirror_pose(learner_seen_in_mirror))[0] == pytest.approx(0.0, abs=1e-12)


def test_low_visibility_joint_is_ignored_not_penalized(base_pose):
    ref = normalize_pose(base_pose)
    bad = base_pose.copy()
    bad[RIGHT_WRIST] += 5.0                                                 # wild tracking glitch
    learner = normalize_pose(bad)

    d_full, _ = pose_distance(ref, learner)
    vis = np.ones(33)
    vis[RIGHT_WRIST] = 0.0
    d_masked, conf = pose_distance(ref, learner, vis_b=vis)

    assert d_full > 0.1
    assert d_masked == pytest.approx(0.0, abs=1e-9)
    assert conf < 1.0                                                       # reported as lower confidence


def test_degenerate_torso_gives_nan_and_zero_confidence(base_pose):
    flat = np.zeros((33, 3))
    n = normalize_pose(flat)
    assert np.isnan(n).all()
    d, conf = pose_distance(normalize_pose(base_pose), n)
    assert np.isnan(d) and conf == 0.0


def test_normalize_sequence_shape(base_pose):
    # Two frames are fewer than the default min_count of 5 torso measurements, so use one global scale.
    seq = np.stack([base_pose, base_pose * 2])
    assert normalize_sequence(seq, window=None, min_count=1).shape == (2, 33, 3)
    with pytest.raises(ValueError):
        normalize_sequence(seq)                       # refuses to guess a scale from too little data


def test_landmarks_to_array_duck_typing(base_pose):
    lms = [SimpleNamespace(x=x, y=y, z=z, visibility=0.9) for x, y, z in base_pose]
    arr = landmarks_to_array(lms)
    assert arr.shape == (33, 4)
    assert np.allclose(arr[:, :3], base_pose) and np.allclose(arr[:, 3], 0.9)
    with pytest.raises(ValueError):
        landmarks_to_array(lms[:10])

# ---------------------------------------------------------------- angles, mirroring, missing data
from pose_utils import (JOINT_ANGLE_NAMES, compute_joint_angles, mask_low_visibility,  # noqa: E402
                        mirror_angles)


def test_identical_poses_have_identical_angles(base_pose):
    a = compute_joint_angles(base_pose)
    assert np.allclose(a, compute_joint_angles(base_pose.copy() * 1.4 + 3.0))      # scale/translation invariant


def test_mirrored_pose_swaps_left_and_right_angles(base_pose):
    angles = compute_joint_angles(base_pose)
    mirrored = compute_joint_angles(mirror_pose(normalize_pose(base_pose)))
    assert np.allclose(mirrored, mirror_angles(angles))
    left, right = JOINT_ANGLE_NAMES.index("left_elbow"), JOINT_ANGLE_NAMES.index("right_elbow")
    assert mirrored[left] == pytest.approx(angles[right])


def test_missing_joint_gives_nan_angle_not_zero(base_pose):
    pose = base_pose.copy()
    pose[RIGHT_WRIST] = np.nan
    angles = compute_joint_angles(pose)
    assert np.isnan(angles[JOINT_ANGLE_NAMES.index("right_elbow")])
    assert np.isfinite(angles[JOINT_ANGLE_NAMES.index("left_elbow")])


@pytest.mark.parametrize("bad", [np.nan, -0.5, np.inf])
def test_invalid_confidence_values_mask_the_joint(base_pose, bad):
    vis = np.ones(33)
    vis[RIGHT_WRIST] = bad
    masked = mask_low_visibility(base_pose, vis)
    assert np.isnan(masked[RIGHT_WRIST]).all()      # NaN, negative and infinite confidence are all "not seen"
    assert np.isfinite(masked[0]).all()


def test_confidence_above_one_is_clipped_in_distance(base_pose):
    n = normalize_pose(base_pose)
    vis = np.full(33, 5.0)                  # nonsense confidence must not weight a joint more than 1
    d, conf = pose_distance(n, n, vis_a=vis, vis_b=vis)
    assert d == pytest.approx(0.0) and conf == pytest.approx(1.0)


def test_mask_rejects_out_of_range_threshold(base_pose):
    with pytest.raises(ValueError):
        mask_low_visibility(base_pose, np.ones(33), min_visibility=1.5)
