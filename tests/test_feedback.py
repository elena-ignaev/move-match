"""The explainable feedback method, on synthetic cases with known ground truth."""

import numpy as np
import pytest

import feedback as fb
from cases import learner_case, scenario_cases
from run_eval_set import check


def evaluate(case, **kw):
    return fb.evaluate_attempt(case.ref, case.learner, case.mapped, section=case.section, speed=case.speed,
                               follow_mode=case.follow_mode, **kw)


@pytest.mark.parametrize("case", scenario_cases(), ids=lambda c: c.name)
def test_evaluation_set(case):
    """Every case behaves as expected, except documented limitations (which must still fail)."""
    problems = check(case, evaluate(case))
    if case.expect.get("known_limitation"):
        assert problems, f"{case.name} is documented as a limitation but now passes: update docs/evaluation.md"
    else:
        assert not problems, problems


def test_identical_movement_has_zero_angle_difference():
    case = learner_case("identical", noise_m=0.0)
    result = evaluate(case)
    assert result["posture"]["median_abs_diff_deg"] == pytest.approx(0.0, abs=0.5)
    assert result["timing"]["offset_s"] == pytest.approx(0.0, abs=0.02)


def test_timing_is_separate_from_posture():
    """Correct movement performed late: timing says late, posture stays close."""
    result = evaluate(learner_case("late", lag_s=0.5))
    assert result["timing"]["direction"] == "late"
    assert result["posture"]["review"] == 0


def test_key_moments_depend_only_on_reference_and_section():
    a = evaluate(learner_case("a"))
    b = evaluate(learner_case("b", lag_s=0.3, body_scale=0.8))
    assert [m["reference_time_s"] for m in a["moments"]] == [m["reference_time_s"] for m in b["moments"]]


def test_mirror_mode_names_the_learners_own_side():
    """In mirror-image mode the reference's right elbow is the learner's LEFT elbow."""
    from cases import RIGHT_ELBOW, bend

    def wrong(world, performed):
        frames = (performed >= 3.0) & (performed <= 6.0)
        return bend(world, RIGHT_ELBOW, 60.0, frames)
    # The learner dances the mirror image; their LEFT arm plays the reference's right arm.
    case = learner_case("mirror_wrong", dances_mirror=True, follow_mode="mirror", modify=wrong)
    result = evaluate(case)
    issues = " ".join(" ".join(m["issues"]) for m in result["moments"])
    assert "left elbow" in issues and "right elbow" not in issues


def test_withheld_feedback_explains_instead_of_scoring():
    result = evaluate(learner_case("gone", detected_fraction=0.2))
    assert result["withheld"] and "posture" not in result and "timing" not in result
    assert any("found a person" in r for r in result["withheld_reasons"])


def test_recording_that_misses_the_reference_is_withheld():
    case = learner_case("short")
    mapped = np.full_like(case.mapped, np.nan)
    mapped[:20] = case.mapped[30:50]
    result = fb.evaluate_attempt(case.ref, case.learner, mapped, section=case.section, speed=1.0)
    assert result["withheld"] and "overlapped" in result["withheld_reasons"][0]


def test_invalid_confidence_values_are_treated_as_not_seen():
    def broken(vis, lt, image):
        vis[:, 14] = np.nan                     # NaN confidence on the right elbow
        vis[:, 16] = -1.0                       # negative on the right wrist
        return vis
    result = evaluate(learner_case("bad_conf", vis_fn=broken))
    assert "right_elbow" in result["excluded_joints"]
    assert not result["withheld"]


def test_flags_need_agreement_of_3d_and_picture():
    s = fb.DEFAULT_SETTINGS
    assert fb.joint_status(50, 40, s) == "review"
    assert fb.joint_status(50, 5, s) == "minor"            # depth-only difference: not flagged
    assert fb.joint_status(50, -40, s) == "minor"          # contradicting directions
    assert fb.joint_status(10, 40, s) == "close"


def test_describe_difference_is_plain_language():
    assert "straighter" in fb.describe_difference("right_elbow", 30)
    assert "more bent" in fb.describe_difference("left_knee", -30)
    assert "raised higher" in fb.describe_difference("left_shoulder", 30)


def test_compare_attempts_reports_improvement():
    before = evaluate(learner_case("w", modify=lambda w, p: __import__("cases").bend(
        w, __import__("cases").RIGHT_ELBOW, 70.0, (p >= 3) & (p <= 6))))
    after = evaluate(learner_case("ok"))
    rows = fb.compare_attempts(after, before)
    assert rows and any(r["change"] == "better" for r in rows)
    assert fb.compare_attempts(after, {**before, "section": [0, 1]}) == []


def test_result_is_json_serializable():
    import json
    json.dumps(evaluate(learner_case("json")), allow_nan=False)


def test_moments_after_the_recording_stopped_are_not_judged():
    """Stopping early must not compare the last frame with a later reference moment."""
    case = learner_case("stopped_early")
    mapped = case.mapped.copy()
    cut = np.flatnonzero(np.isfinite(mapped) & (mapped > 5.0))[0]
    learner = case.learner
    keep = slice(0, cut)
    from landmark_store import LandmarkSequence
    short = LandmarkSequence(learner.timestamps[keep], learner.landmarks[keep], learner.visibility[keep],
                             learner.detected[keep], learner.image_landmarks[keep], learner.metadata,
                             learner.brightness[keep])
    result = fb.evaluate_attempt(case.ref, short, mapped[keep], section=case.section, speed=1.0)
    late = [m for m in result["moments"] if m["reference_time_s"] > 5.3]
    assert late and all(m["status"] == "not_covered" for m in late)
    assert result["posture"]["review"] == 0
