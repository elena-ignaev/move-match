"""Route tests for the dev-only landmark blueprint. Run: pytest tests/test_dev_tools.py -v"""

from pathlib import Path

import numpy as np
import pytest
from flask import Flask

import dev_tools
import landmark_store as ls
import pose_utils as pu

T = 6
MISSED = 3


def make_sequence():
    rng = np.random.default_rng(0)
    world = rng.normal(0, 0.3, (T, 33, 3))
    world[:, pu.L_HIP], world[:, pu.R_HIP] = (0.1, 0, 0), (-0.1, 0, 0)
    world[:, pu.L_SHOULDER], world[:, pu.R_SHOULDER] = (0.2, -0.5, 0), (-0.2, -0.5, 0)
    image = rng.uniform(0.2, 0.8, (T, 33, 3))
    vis = np.full((T, 33), 0.9)
    det = np.ones(T, dtype=bool)
    world[MISSED], image[MISSED], vis[MISSED], det[MISSED] = np.nan, np.nan, 0.0, False
    return ls.LandmarkSequence(
        timestamps=np.arange(T) * 100.0, landmarks=world, visibility=vis, detected=det,
        image_landmarks=image,
        metadata=ls.build_metadata(fps=10.0, width=64, height=48, num_frames=T),
    )


@pytest.fixture
def storage(tmp_path):
    for sub in ("uploads", "playback", "landmarks"):
        (tmp_path / sub).mkdir()
    (tmp_path / "uploads" / "dance.mp4").write_bytes(b"x")
    (tmp_path / "playback" / "dance_browser.mp4").write_bytes(b"0123456789")
    ls.save_landmarks(tmp_path / "landmarks" / "dance.npz", make_sequence())
    return tmp_path


@pytest.fixture
def client(storage):
    # Resolve templates relative to dev_tools.py (the project root), not this test file,
    # so the test works whether it lives in the root or in tests/.
    templates = Path(dev_tools.__file__).resolve().parent / "templates"
    app = Flask(__name__, template_folder=str(templates))
    app.config.update(TESTING=True, DEV_STORAGE_DIR=str(storage))
    app.register_blueprint(dev_tools.bp)
    return app.test_client()


def test_index_lists_files(client):
    html = client.get("/dev/landmarks/").get_data(as_text=True)
    assert "dance.mp4" in html and "dance_browser.mp4" in html and "dance.npz" in html


def test_summary_reports_detection_and_angles(client):
    s = client.get("/dev/landmarks/summary/dance.npz").get_json()
    assert s["frames"] == T and s["detected_frames"] == T - 1
    assert s["detected_ratio"] == pytest.approx((T - 1) / T, abs=1e-3)
    assert s["longest_missing_run"] == 1 and s["fps"] == 10.0
    assert set(s["angle_stats"]) == set(pu.JOINT_ANGLE_NAMES)
    assert s["metadata"]["schema_version"] == ls.SCHEMA_VERSION


def test_pose_is_normalized_and_mirror_swaps_angles(client):
    p = client.get("/dev/landmarks/pose/dance.npz/0?min_vis=0.5").get_json()
    pose = np.array(p["pose"], dtype=float)
    hip_mid = (pose[pu.L_HIP] + pose[pu.R_HIP]) / 2
    sh_mid = (pose[pu.L_SHOULDER] + pose[pu.R_SHOULDER]) / 2
    assert np.allclose(hip_mid, 0, atol=1e-3)
    assert np.linalg.norm(sh_mid - hip_mid) == pytest.approx(1.0, abs=1e-3)   # torso length 1

    m = client.get("/dev/landmarks/pose/dance.npz/0?mirror=1").get_json()
    assert m["angles"]["left_elbow"] == pytest.approx(p["angles"]["right_elbow"], abs=1e-3)
    assert m["angles"]["right_knee"] == pytest.approx(p["angles"]["left_knee"], abs=1e-3)


def test_missed_frame_returns_nulls_not_errors(client):
    p = client.get(f"/dev/landmarks/pose/dance.npz/{MISSED}").get_json()
    assert p["detected"] is False
    assert all(row[0] is None for row in p["pose"])
    assert all(v is None for v in p["angles"].values())


def test_high_min_visibility_masks_everything(client):
    p = client.get("/dev/landmarks/pose/dance.npz/0?min_vis=0.95").get_json()
    assert all(row[0] is None for row in p["pose"])


def test_overlay_payload_shape(client):
    o = client.get("/dev/landmarks/overlay/dance.npz").get_json()
    assert o["fps"] == 10.0 and len(o["detected"]) == T
    assert np.array(o["image"], dtype=object).shape == (T, 33, 2)
    assert o["image"][MISSED][0] == [None, None]


def test_media_serves_playback_video_with_range(client):
    r = client.get("/dev/landmarks/media/dance_browser.mp4", headers={"Range": "bytes=0-3"})
    assert r.status_code == 206 and r.data == b"0123"


def test_compute_calls_precompute_and_reports(client, monkeypatch, storage):
    seen = {}

    def fake_precompute(src, out, force=False):
        seen.update(src=Path(src).name, out=Path(out).name, force=force)
        return {"skipped": False, "landmarks_path": str(out), "frame_count": 6}

    monkeypatch.setattr(dev_tools, "precompute_landmarks", fake_precompute)
    r = client.post("/dev/landmarks/compute/dance.mp4?force=1")
    assert r.status_code == 200
    assert seen == {"src": "dance.mp4", "out": "dance.npz", "force": True}
    assert r.get_json()["landmark_file"] == "dance.npz"


def test_compute_surfaces_errors_as_json(client, monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError("Pose Landmarker model is missing.")

    monkeypatch.setattr(dev_tools, "precompute_landmarks", boom)
    r = client.post("/dev/landmarks/compute/dance.mp4")
    assert r.status_code == 400 and "model is missing" in r.get_json()["error"]


@pytest.mark.parametrize("url", [
    "/dev/landmarks/summary/..%2Fsecret.npz",
    "/dev/landmarks/media/..%2F..%2Fapp.py",
    "/dev/landmarks/summary/.hidden.npz",
])
def test_path_traversal_is_rejected(client, url):
    assert client.get(url).status_code in (400, 404)


def test_missing_and_wrong_files(client):
    assert client.get("/dev/landmarks/summary/nope.npz").status_code == 404
    assert client.get("/dev/landmarks/summary/dance.mp4").status_code == 400
    assert client.get("/dev/landmarks/pose/dance.npz/99").status_code == 404
    assert client.post("/dev/landmarks/compute/missing.mp4").status_code == 404


def test_stale_schema_gives_409(client, storage):
    path = storage / "landmarks" / "old.npz"
    seq = make_sequence()
    seq.metadata = dict(seq.metadata)
    ls.save_landmarks(path, seq)
    with np.load(path, allow_pickle=False) as d:
        arrays = {k: d[k] for k in d.files}
    import json
    meta = json.loads(str(arrays["metadata_json"]))
    meta["schema_version"] = 0
    arrays["metadata_json"] = np.array(json.dumps(meta))
    np.savez_compressed(path, **arrays)
    r = client.get("/dev/landmarks/summary/old.npz")
    assert r.status_code == 409 and "Stale" in r.get_json()["error"]


def test_trailing_slash_url_redirects(client):
    r = client.get("/dev/landmarks")
    assert r.status_code in (301, 308) and r.headers["Location"].endswith("/dev/landmarks/")


@pytest.mark.parametrize("value, expected", [
    (None, False), ("", False), ("0", False), ("1", True), ("true", True), ("YES", True), (" on ", True),
])
def test_init_app_respects_env_var(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("ENABLE_DEV_TOOLS", raising=False)
    else:
        monkeypatch.setenv("ENABLE_DEV_TOOLS", value)
    app = Flask(__name__)
    assert dev_tools.init_app(app) is expected
    assert dev_tools.init_app(app) is expected                      # idempotent
    assert any(r.rule.startswith("/dev/landmarks") for r in app.url_map.iter_rules()) is expected


def test_init_app_force_overrides_env(monkeypatch):
    monkeypatch.delenv("ENABLE_DEV_TOOLS", raising=False)
    app = Flask(__name__)
    assert dev_tools.init_app(app, force=True) is True


def test_templates_can_show_dev_link_only_when_enabled(monkeypatch):
    from flask import render_template_string
    snippet = "{% if config.DEV_TOOLS_ENABLED %}<a href=\"{{ url_for('dev_landmarks.index') }}\">Dev tools</a>{% endif %}"

    monkeypatch.setenv("ENABLE_DEV_TOOLS", "1")
    on = Flask(__name__)
    dev_tools.init_app(on)
    with on.test_request_context():
        assert 'href="/dev/landmarks/"' in render_template_string(snippet)

    monkeypatch.delenv("ENABLE_DEV_TOOLS")
    off = Flask(__name__)
    dev_tools.init_app(off)
    with off.test_request_context():
        assert render_template_string(snippet) == ""


@pytest.mark.parametrize("value", ["1.5", "-0.1", "nan", "abc"])
def test_pose_rejects_out_of_range_min_visibility_cleanly(client, value):
    r = client.get(f"/dev/landmarks/pose/dance.npz/0?min_vis={value}")
    assert r.status_code == 400
    assert "min_vis" in r.get_json()["error"]