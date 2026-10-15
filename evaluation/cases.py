"""Synthetic landmark sequences with KNOWN ground truth, for testing the feedback method.

A reference dancer and a learner are generated from the same choreography (research/synthetic_dance),
then the learner is changed in one controlled way: late, wrong arm, mirrored, legs out of frame,
smaller, farther from the camera, occluded, dark, missed turn, ... Because we know exactly what was
changed, we know what correct feedback must say.

What this can NOT tell us: whether the feedback agrees with what a dance teacher would say about a
real recording. MediaPipe errors on real video (depth mistakes, left/right swaps when facing away,
loose clothing) are only crudely imitated by noise here. See docs/evaluation.md.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "research"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pose_utils as pu                                   # noqa: E402
from landmark_store import LandmarkSequence, build_metadata  # noqa: E402
from synthetic_dance import Choreography                  # noqa: E402

FPS = 30.0

# Joint chains for bending a limb: (parent, vertex, child, points moved with the child)
RIGHT_ELBOW = (12, 14, 16, (16, 18, 20, 22))
LEFT_KNEE = (23, 25, 27, (27, 29, 31))


def standing_pose() -> np.ndarray:
    """Adult standing pose in MediaPipe world convention (metres, hips at origin, y down, left = +x).

    research/synthetic_dance places non-torso joints at random, which gives 10-20 cm "limbs"; joint
    angles of such short segments are far more sensitive to noise than a real body's.
    """
    p = np.zeros((33, 3))
    p[0] = (0.0, -0.66, -0.06)                                  # nose
    p[1:4] = (0.03, -0.69, -0.04); p[4:7] = (-0.03, -0.69, -0.04)   # eyes
    p[7], p[8] = (0.07, -0.67, 0.02), (-0.07, -0.67, 0.02)         # ears
    p[9], p[10] = (0.02, -0.61, -0.05), (-0.02, -0.61, -0.05)      # mouth
    p[11], p[12] = (0.18, -0.50, 0.0), (-0.18, -0.50, 0.0)         # shoulders
    p[13], p[14] = (0.24, -0.22, 0.02), (-0.24, -0.22, 0.02)       # elbows
    p[15], p[16] = (0.27, 0.04, -0.02), (-0.27, 0.04, -0.02)       # wrists
    for j, w in ((17, 15), (19, 15), (21, 15), (18, 16), (20, 16), (22, 16)):
        p[j] = p[w] + (0.0, 0.07, -0.01)                           # hands
    p[23], p[24] = (0.10, 0.0, 0.0), (-0.10, 0.0, 0.0)             # hips
    p[25], p[26] = (0.11, 0.42, -0.01), (-0.11, 0.42, -0.01)       # knees
    p[27], p[28] = (0.12, 0.82, 0.03), (-0.12, 0.82, 0.03)         # ankles
    p[29], p[30] = (0.12, 0.87, 0.07), (-0.12, 0.87, 0.07)         # heels
    p[31], p[32] = (0.13, 0.88, -0.10), (-0.13, 0.88, -0.10)       # toes
    return p


def make_choreography(seed: int = 0) -> Choreography:
    # Holds give the dance clear "hit" moments, like real choreography.
    choreo = Choreography(seed=seed, motion=0.55, holds=((2.0, 0.6), (4.5, 0.6), (6.5, 0.5)))
    choreo.base = standing_pose()
    return choreo


def project(world: np.ndarray, scale: float = 0.33, cx: float = 0.5, cy: float = 0.45,
            aspect: float = 16 / 9) -> np.ndarray:
    """Pseudo camera: world metres -> normalized image coords (orthographic, y down)."""
    img = np.empty_like(world)
    img[..., 0] = cx + world[..., 0] * scale / aspect
    img[..., 1] = cy + world[..., 1] * scale
    img[..., 2] = world[..., 2]
    return img


def to_sequence(world, times_s, *, image=None, vis=None, detected=None, brightness=None,
                fps: float = FPS) -> LandmarkSequence:
    world = np.asarray(world, dtype=np.float64)
    t = len(world)
    det = np.ones(t, dtype=bool) if detected is None else np.asarray(detected, dtype=bool)
    image = project(world) if image is None else image
    vis = np.full((t, 33), 0.95) if vis is None else vis
    world = world.copy()
    image = image.copy()
    world[~det] = np.nan
    image[~det] = np.nan
    vis = np.where(det[:, None], vis, 0.0)
    return LandmarkSequence(
        timestamps=np.asarray(times_s, dtype=np.float64) * 1000.0,
        landmarks=world, visibility=vis, detected=det, image_landmarks=image,
        metadata=build_metadata(fps=fps, width=1280, height=720, num_frames=t),
        brightness=np.full(t, 120.0) if brightness is None else np.asarray(brightness, dtype=np.float64),
    )


def _rotate_about(points, center, axis, angle_rad):
    """Rodrigues rotation of points (..., 3) about `axis` through `center`."""
    k = axis / np.linalg.norm(axis, axis=-1, keepdims=True)
    v = points - center
    cos, sin = np.cos(angle_rad), np.sin(angle_rad)
    return center + v * cos + np.cross(k, v) * sin + k * np.sum(k * v, axis=-1, keepdims=True) * (1 - cos)


def bend(world: np.ndarray, chain, delta_deg: float, frames=slice(None)) -> np.ndarray:
    """Open (+) or close (-) the joint angle at chain's vertex by delta_deg, in the limb's own plane."""
    a, b, c, moved = chain
    out = world.copy()
    w = out[frames]
    ba, bc = w[:, a] - w[:, b], w[:, c] - w[:, b]
    axis = np.cross(bc, ba)                                   # rotating c about this axis opens the angle
    for j in moved:
        w[:, j] = _rotate_about(w[:, j], w[:, b], axis, -np.radians(delta_deg))
    out[frames] = w
    return out


@dataclass
class Case:
    name: str
    description: str
    ref: LandmarkSequence
    learner: LandmarkSequence
    mapped: np.ndarray                     # reference time on screen at each learner frame
    section: tuple[float, float]
    speed: float = 1.0
    follow_mode: str = "same_side"
    expect: dict = field(default_factory=dict)


def build_reference(choreo: Choreography, duration: float = 8.0, turn_at: float | None = None):
    t = np.arange(int(duration * FPS)) / FPS
    world = choreo.pose(t)
    if turn_at is not None:                                   # the reference turns 90 degrees side-on
        ang = np.clip((t - turn_at) / 0.5, 0, 1) * np.pi / 2
        world = np.stack([pu.rotate_yaw(w, a) for w, a in zip(world, ang)])
    return t, world


def learner_case(name: str, *, seed: int = 0, duration: float = 8.0, speed: float = 1.0, lag_s: float = 0.0,
                 lead_in_s: float = 1.0, noise_m: float = 0.012, follow_mode: str = "same_side",
                 dances_mirror: bool = False,
                 modify=None, image_scale: float = 0.33, image_cy: float = 0.45, body_scale: float = 1.0,
                 detected_fraction: float = 1.0, vis_fn=None, brightness: float = 120.0, turn_at=None,
                 learner_turns: bool = True, description: str = "", expect=None) -> Case:
    """One synthetic attempt. The learner starts recording lead_in_s before the reference plays."""
    choreo = make_choreography(seed)
    ref_t, ref_world = build_reference(choreo, duration, turn_at)
    ref = to_sequence(ref_world, ref_t)

    rec_len = lead_in_s + duration / speed + 0.5
    lt = np.arange(int(rec_len * FPS)) / FPS
    mapped = (lt - lead_in_s) * speed                          # reference time on screen
    mapped = np.where((mapped >= 0) & (mapped <= duration - 1 / FPS), mapped, np.nan)
    performed = np.clip(((lt - lead_in_s - lag_s) * speed), 0, duration - 1 / FPS)
    world = choreo.pose(performed)
    if turn_at is not None and learner_turns:
        ang = np.clip((performed - turn_at) / 0.5, 0, 1) * np.pi / 2
        world = np.stack([pu.rotate_yaw(w, a) for w, a in zip(world, ang)])
    hip = (world[:, pu.L_HIP] + world[:, pu.R_HIP]) / 2
    world = hip[:, None] + (world - hip[:, None]) * body_scale
    rng = np.random.default_rng(seed + 100)
    world = world + rng.normal(0, noise_m, world.shape)
    if modify is not None:
        world = modify(world, performed)
    if dances_mirror:                                          # the learner dances the mirror image
        world = pu.mirror_pose(world)
    image = project(world, scale=image_scale, cy=image_cy)
    vis = np.full((len(lt), 33), 0.95)
    if vis_fn is not None:
        vis = vis_fn(vis, lt, image)
    detected = np.ones(len(lt), dtype=bool)
    if detected_fraction < 1.0:
        detected[rng.random(len(lt)) > detected_fraction] = False
    learner = to_sequence(world, lt, image=image, vis=vis, detected=detected,
                          brightness=np.full(len(lt), brightness))
    return Case(name, description, ref, learner, mapped, (0.0, duration), speed, follow_mode, expect or {})


# ---------------------------------------------------------------- named scenarios
def _wrong_elbow(world, performed):
    frames = (performed >= 3.0) & (performed <= 6.0)
    return bend(world, RIGHT_ELBOW, 60.0, frames)


def _legs_out(vis, lt, image):
    image[:, 25:33, 1] += 0.6                                  # knees and feet pushed below the frame
    vis[:, 25:33] = 0.2
    return vis


def _arm_occluded(vis, lt, image):
    mask = (lt % 2.0) < 1.2
    vis[np.ix_(mask, [14, 16, 18, 20, 22])] = 0.1          # elbow, wrist, hand (shoulder stays visible)
    return vis


def _subtle_elbow(world, performed):
    frames = (performed >= 3.0) & (performed <= 6.0)
    return bend(world, RIGHT_ELBOW, 25.0, frames)


def _depth_noise(world, performed, sigma=0.06, seed=5):
    """Slowly varying depth (z) errors, the typical single-camera failure: MediaPipe guesses depth."""
    rng = np.random.default_rng(seed)
    t = len(world)
    kernel = np.exp(-0.5 * (np.arange(-30, 31) / 10.0) ** 2)
    kernel /= np.sqrt((kernel ** 2).sum())
    white = rng.normal(0, 1, (t + len(kernel) - 1, 33))
    smooth = np.stack([np.convolve(white[:, j], kernel, mode="valid") for j in range(33)], axis=1)
    out = world.copy()
    out[..., 2] += sigma * smooth
    return out


def _arm_forward(world, performed):
    """Right arm swung 80 degrees toward the camera about the vertical axis through the shoulder.

    The elbow angle and the arm-to-torso angle do not change at all, so joint angles cannot see it.
    """
    frames = (performed >= 3.0) & (performed <= 6.0)
    out = world.copy()
    shoulder = out[frames][:, 12:13]
    arm = out[frames][:, [14, 16, 18, 20, 22]] - shoulder
    a = np.radians(80.0)
    rot = arm.copy()
    rot[..., 0] = arm[..., 0] * np.cos(a) + arm[..., 2] * np.sin(a)
    rot[..., 2] = -arm[..., 0] * np.sin(a) + arm[..., 2] * np.cos(a)
    sub = out[frames]
    sub[:, [14, 16, 18, 20, 22]] = shoulder + rot
    out[frames] = sub
    return out


def _dark_and_lost(vis, lt, image):
    rng = np.random.default_rng(7)
    vis[:] = rng.uniform(0.1, 0.6, vis.shape)
    return vis


def scenario_cases() -> list[Case]:
    """The evaluation set. `expect` holds what correct feedback must say."""
    return [
        learner_case("correct", description="Same choreography, small tracking noise.",
                     expect={"withheld": False, "timing": "on_time", "max_review": 0}),
        learner_case("correct_slow_speed", speed=0.5, description="Correct, practised at 0.5x speed.",
                     expect={"withheld": False, "timing": "on_time", "max_review": 0}),
        learner_case("late_0.4s", lag_s=0.4, description="Correct shapes, 0.4 s behind the music.",
                     expect={"withheld": False, "timing": "late", "offset": 0.4, "max_review": 0}),
        learner_case("early_0.3s", lag_s=-0.3, description="Correct shapes, 0.3 s ahead.",
                     expect={"withheld": False, "timing": "early", "offset": -0.3, "max_review": 0}),
        learner_case("late_0.4s_at_half_speed", lag_s=0.4, speed=0.5,
                     description="0.4 s late (wall clock) while practising at 0.5x.",
                     expect={"withheld": False, "timing": "late", "offset": 0.4, "max_review": 0}),
        learner_case("wrong_right_elbow", modify=_wrong_elbow,
                     description="Right elbow opened by 60 degrees from 3 s to 6 s.",
                     expect={"withheld": False, "min_review": 1, "review_joint": "right_elbow",
                             "review_window": (3.0, 6.0)}),
        learner_case("mirror_image_unannounced", dances_mirror=True,
                     description="Learner dances the mirror image but chose 'same side'.",
                     expect={"withheld": False, "min_review": 2}),
        learner_case("mirror_image_announced", dances_mirror=True, follow_mode="mirror",
                     description="Learner dances the mirror image and chose 'mirror image' mode.",
                     expect={"withheld": False, "max_review": 0}),
        learner_case("smaller_dancer", body_scale=0.75, description="A smaller body (75%).",
                     expect={"withheld": False, "max_review": 0}),
        learner_case("farther_from_camera", image_scale=0.2, description="Learner further away (smaller in the image).",
                     expect={"withheld": False, "max_review": 0}),
        learner_case("legs_out_of_frame", vis_fn=_legs_out, description="Knees and feet below the bottom edge.",
                     expect={"withheld": False, "excluded_contains": "right_knee", "tracking_message": "below the bottom"}),
        learner_case("right_arm_occluded", vis_fn=_arm_occluded,
                     description="Right arm hidden 60% of the time (e.g. behind the body).",
                     expect={"withheld": False, "excluded_contains": "right_elbow",
                             "not_excluded": "left_elbow"}),
        learner_case("subtle_right_elbow_25deg", modify=_subtle_elbow,
                     description="Right elbow opened by only 25 degrees from 3 s to 6 s (a teacher might mention it).",
                     expect={"withheld": False, "review_joint": "right_elbow", "review_window": (3.0, 6.0)}),
        learner_case("slightly_late_0.1s", lag_s=0.1, description="0.1 s late: within the on-time tolerance.",
                     expect={"withheld": False, "timing": "on_time", "max_review": 0}),
        learner_case("depth_noise_6cm", modify=_depth_noise,
                     description="Correct dancing, but slowly drifting depth errors of ~6 cm (single-camera depth guessing).",
                     expect={"withheld": False, "max_review": 0}),
        learner_case("depth_noise_12cm", modify=lambda w, p: _depth_noise(w, p, sigma=0.12),
                     description="Correct dancing with ~12 cm depth errors (bad but realistic MediaPipe depth).",
                     expect={"withheld": False, "max_review": 0}),
        learner_case("arm_forward_not_sideways", modify=_arm_forward,
                     description="Right arm reaches toward the camera instead of to the side (3-6 s). KNOWN LIMITATION: "
                                 "no joint angle changes, so this is expected to be missed.",
                     expect={"withheld": False, "min_review": 1, "known_limitation": True}),
        learner_case("no_person_most_frames", detected_fraction=0.3,
                     description="Person detected in only 30% of frames.",
                     expect={"withheld": True, "reason_contains": "found a person"}),
        learner_case("dark_poor_tracking", vis_fn=_dark_and_lost, brightness=25.0,
                     description="Dark room: low joint confidence everywhere.",
                     expect={"withheld": True, "reason_contains": "dark"}),
        learner_case("missed_turn", turn_at=3.5, learner_turns=False,
                     description="Reference turns 90 degrees side-on at 3.5 s; learner keeps facing the camera.",
                     expect={"withheld": False, "facing_review_after": 4.0}),
        learner_case("turn_performed", turn_at=3.5, learner_turns=True,
                     description="Both turn 90 degrees: orientation is kept, so no facing flag.",
                     expect={"withheld": False, "max_review": 0}),
    ]
