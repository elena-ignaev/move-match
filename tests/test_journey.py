"""One complete journey with a small fixture video:

choose a reference -> practise a section -> record -> replay -> feedback -> retry and compare.

The fake landmarker plays the role of the camera + MediaPipe, so we know what the learner did:
first take 0.4 s late with the right elbow wrong, second take on time and correct.
"""

from __future__ import annotations

import json

import numpy as np

from cases import RIGHT_ELBOW, bend
from conftest import needs_ffmpeg, sync_payload

pytestmark = needs_ffmpeg

SPEED, START, END = 0.5, 1.0, 5.0
RECORDER_START_MS, LEAD_IN_MS = 500.0, 1000.0


def performer(poses, *, lag_s: float = 0.0, wrong_elbow: bool = False):
    """Learner pose at recording time t (s): the reference time on screen, shifted by lag_s."""
    def pose(t):
        on_screen = START + (t - LEAD_IN_MS / 1000.0 - lag_s) * SPEED
        world = poses.choreo.pose(np.array([np.clip(on_screen, 0.0, 5.9)]))
        if wrong_elbow and 2.5 <= on_screen <= 4.5:
            world = bend(world, RIGHT_ELBOW, 70.0)
        return world[0]
    return pose


def record(client, media_dir, video_id, upload_id):
    sync = sync_payload(speed=SPEED, start=START, end=END, lead_in_ms=LEAD_IN_MS, recorder_start_ms=RECORDER_START_MS)
    with open(media_dir / "attempt.webm", "rb") as f:
        response = client.post(f"/videos/{video_id}/attempts", data={
            "recording": (f, "attempt.webm", "video/webm"), "upload_id": upload_id,
            "speed": str(SPEED), "section_start": str(START), "section_end": str(END),
            "view_mirrored": "1", "follow_mode": "same_side", "sync": json.dumps(sync),
        }, content_type="multipart/form-data")
    assert response.status_code == 201, response.get_json()
    return response.get_json()["attempt_id"]


def test_complete_journey(app, client, media_dir, poses):
    from models import Video

    # 1. choose a short reference
    with open(media_dir / "reference.mp4", "rb") as f:
        assert client.post("/upload", data={"video_file": (f, "routine.mp4")},
                           content_type="multipart/form-data").status_code == 303
    with app.app_context():
        video = Video.query.one()
        assert video.status == "completed", video.error_message
        video_id = video.id
    assert "routine.mp4" in client.get("/library").get_data(as_text=True)

    # 2. practise a section
    practice = client.get(f"/videos/{video_id}/practice").get_data(as_text=True)
    assert "Loop section" in practice and "Mirror view" in practice
    assert client.get(f"/videos/{video_id}/media").headers.get("Accept-Ranges") == "bytes"

    # 3. record an attempt (late, wrong elbow)
    assert "Record attempt" in client.get(f"/videos/{video_id}/ghost?start=1&end=5&speed=0.5").get_data(as_text=True)
    poses.attempt = performer(poses, lag_s=0.4, wrong_elbow=True)
    first = record(client, media_dir, video_id, "take-1")

    # 4. replay side by side: recording time -> reference time map follows the sync log
    data = client.get(f"/api/attempts/{first}").get_json()
    assert data["status"] == "ready", data
    mapping = {round(t, 1): r for t, r in data["replay_map"] if r is not None}
    assert abs(mapping[1.0] - START) < 0.05                 # reference started 1 s into the recording
    assert abs(mapping[3.0] - (START + 2.0 * SPEED)) < 0.05  # half speed
    assert client.get(f"/attempts/{first}").status_code == 200

    # 5. understandable feedback: late, and the right elbow flagged
    fb = data["feedback"]
    assert fb["timing"]["direction"] == "late" and abs(fb["timing"]["offset_s"] - 0.4) < 0.08
    assert "behind the reference" in fb["timing"]["message"]
    assert fb["posture"]["review"] >= 1
    issues = " ".join(" ".join(m["issues"]) for m in fb["moments"])
    assert "right elbow" in issues
    assert "%" not in fb["headline"]                         # no accuracy percentage

    # 6. retry and compare: on time and correct
    poses.attempt = performer(poses)
    second = record(client, media_dir, video_id, "take-2")
    data2 = client.get(f"/api/attempts/{second}").get_json()
    fb2 = data2["feedback"]
    assert fb2["timing"]["direction"] == "on_time"
    assert fb2["posture"]["review"] == 0, [(m["label"], m["issues"], m["learner_time_s"]) for m in fb2["moments"] if m["status"] == "review"]
    assert data2["previous_attempt"]["id"] == first
    assert any(row["change"] == "better" for row in data2["comparison"])
    # the same key moments are used for both takes, so they can be compared one by one
    assert [m["reference_time_s"] for m in fb["moments"]] == [m["reference_time_s"] for m in fb2["moments"]]
    page = client.get(f"/videos/{video_id}").get_data(as_text=True)
    assert "+0.4" in page                                    # attempts table shows the timing trend


def test_buffering_stall_during_recording_is_excluded(app, client, media_dir, poses):
    """The reference froze for 1.5 s; the learner kept dancing. Those frames must not be compared."""
    from models import Video
    with open(media_dir / "reference.mp4", "rb") as f:
        client.post("/upload", data={"video_file": (f, "r.mp4")}, content_type="multipart/form-data")
    with app.app_context():
        video_id = Video.query.one().id
    poses.attempt = performer(poses)
    sync = sync_payload(speed=SPEED, start=START, end=3.0, lead_in_ms=LEAD_IN_MS,
                        recorder_start_ms=RECORDER_START_MS, stall=(2.0, 1500))
    with open(media_dir / "attempt.webm", "rb") as f:
        attempt_id = client.post(f"/videos/{video_id}/attempts", data={
            "recording": (f, "a.webm", "video/webm"), "speed": str(SPEED), "section_start": "1", "section_end": "3",
            "sync": json.dumps(sync)}, content_type="multipart/form-data").get_json()["attempt_id"]
    data = client.get(f"/api/attempts/{attempt_id}").get_json()
    unmapped = [t for t, r in data["replay_map"] if r is None and 3.1 < t < 4.4]
    assert unmapped, "frames during the stall should have no reference time"
