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


# ---------------------------------------------------------------- attempt score
def _score(case):
    return evaluate(case)["score"]


def test_score_is_100_for_identical_movement_and_has_breakdown():
    score = _score(learner_case("identical_score", noise_m=0.0))
    assert score["value"] == 100 and score["provisional"]
    assert set(score["components"]) == {"posture", "timing", "completion"}
    assert sum(c["effective_weight"] for c in score["components"].values()) == pytest.approx(1.0)


def test_withheld_feedback_has_no_score():
    score = _score(learner_case("gone_score", detected_fraction=0.2))
    assert score["value"] is None and "withheld" in score["reason"]


def test_lateness_lowers_only_the_timing_component():
    on_time, late = _score(learner_case("t0")), _score(learner_case("t1", lag_s=0.4))
    assert late["components"]["posture"]["value"] == on_time["components"]["posture"]["value"]
    assert late["components"]["timing"]["value"] < on_time["components"]["timing"]["value"]
    assert late["value"] < on_time["value"]


def test_score_orders_cases_like_a_teacher_would():
    by_name = {c.name: _score(c)["value"] for c in scenario_cases()}
    assert by_name["correct"] >= 95
    assert by_name["correct"] > by_name["subtle_right_elbow_25deg"] > by_name["wrong_right_elbow"]
    assert by_name["correct"] > by_name["late_0.4s"]
    assert by_name["mirror_image_unannounced"] < 50
    assert by_name["mirror_image_announced"] >= 95
    # body size, distance, depth noise and hidden limbs must not lower the score
    for name in ("smaller_dancer", "farther_from_camera", "depth_noise_6cm", "legs_out_of_frame", "right_arm_occluded"):
        assert by_name[name] >= 95, name


def test_unmeasurable_timing_is_left_out_not_scored_zero():
    result = evaluate(learner_case("t_static"))
    result["timing"] = {"confident": False, "offset_s": None}
    score = fb.attempt_score(result)
    assert "timing" not in score["components"] and score["notes"]
    assert score["components"]["posture"]["effective_weight"] == pytest.approx(0.7 / 0.8)


def test_score_can_be_derived_for_old_evaluations():
    result = evaluate(learner_case("old"))
    stored = {k: v for k, v in result.items() if k != "score"}
    assert fb.score_of(stored)["value"] == result["score"]["value"]
