"""Developer-only Flask blueprint for exercising landmark_store / pose_utils in isolation.

Nothing in the normal app flow imports this. Register it only when you opt in:

    # app.py, after the app is created
    import os
    if os.environ.get("ENABLE_DEV_TOOLS") == "1":
        from dev_tools import bp as dev_landmarks_bp
        app.register_blueprint(dev_landmarks_bp)

Then run with ENABLE_DEV_TOOLS=1 and open /dev/landmarks/.

It works directly on files under storage/ and never touches SQLite:
    storage/uploads/     original videos   (input for extraction)
    storage/annotated/   browser videos    (played back for the overlay check)
    storage/landmarks/   .npz caches       (created by this tool)
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
from flask import Blueprint, abort, current_app, jsonify, render_template, request, send_from_directory
from werkzeug.exceptions import HTTPException

import landmark_store as ls
import pose_utils as pu
from process_video import BASE_DIR, POSE_CONNECTIONS, precompute_landmarks

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

bp = Blueprint("dev_landmarks", __name__, url_prefix="/dev/landmarks")

# ------------------------------------------------------------ registration
def dev_tools_enabled() -> bool:
    return os.environ.get("ENABLE_DEV_TOOLS", "").strip().lower() in {"1", "true", "yes", "on"}
 
 
def init_app(app, force: bool = False) -> bool:
    """Register the blueprint on `app` if dev tools are enabled. Returns True if registered.
 
    Safe to call more than once. Enabled when ENABLE_DEV_TOOLS is 1/true/yes/on,
    or when force=True.
    """
    if not (force or dev_tools_enabled()):
        app.logger.info("Dev tools not registered (set ENABLE_DEV_TOOLS=1 to enable).")
        return False
    if bp.name not in app.blueprints:
        app.register_blueprint(bp)
        app.config["DEV_TOOLS_ENABLED"] = True      # lets templates show a dev link
        app.logger.warning("Dev tools ENABLED at %s (development only).", bp.url_prefix + "/")
    return True

# ------------------------------------------------------------------ helpers
def _dir(kind: str) -> Path:
    root = Path(current_app.config.get("DEV_STORAGE_DIR", BASE_DIR / "storage"))
    return root / kind


def _safe(name: str) -> str:
    """Reject anything that is not a bare file name (blocks path traversal)."""
    if not name or name != Path(name).name or name.startswith("."):
        abort(400, "Invalid file name")
    return name


def _list(kind: str, exts: set[str]) -> list[str]:
    d = _dir(kind)
    if not d.is_dir():
        return []
    return sorted(
        p.name for p in d.iterdir()
        if p.is_file() and p.suffix.lower() in exts and not p.name.endswith(".tmp.npz")
    )


def _load(name: str) -> ls.LandmarkSequence:
    name = _safe(name)
    if not name.endswith(".npz"):
        abort(400, "Expected a .npz landmark file")
    path = _dir("landmarks") / name
    if not path.is_file():
        abort(404, "No such landmark file")
    try:
        return ls.load_landmarks(path)
    except ls.LandmarkSchemaError as exc:
        abort(409, f"Stale or invalid landmark file: {exc}")


def _clean(x):
    if isinstance(x, list):
        return [_clean(v) for v in x]
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def _jsonable(a) -> list:
    """Array -> nested lists, rounded, with NaN/inf turned into null."""
    return _clean(np.round(np.asarray(a, dtype=float), 4).tolist())


@bp.errorhandler(HTTPException)
def _json_error(exc):
    return jsonify(error=exc.description), exc.code


# ------------------------------------------------------------------- routes
@bp.get("/")
def index():
    return render_template(
        "dev_landmarks.html",
        uploads=_list("uploads", VIDEO_EXTS),
        annotated=_list("annotated", VIDEO_EXTS),
        landmark_files=_list("landmarks", {".npz"}),
        connections=POSE_CONNECTIONS,
        angle_names=pu.JOINT_ANGLE_NAMES,
    )


@bp.post("/compute/<name>")
def compute(name):
    """Run precompute_landmarks on an upload. ?force=1 ignores a current cache."""
    name = _safe(name)
    src = _dir("uploads") / name
    if not src.is_file():
        abort(404, "Upload not found")
    out = _dir("landmarks") / f"{src.stem}.npz"
    try:
        info = precompute_landmarks(src, out, force=request.args.get("force") == "1")
    except (ValueError, FileNotFoundError) as exc:
        abort(400, str(exc))
    except RuntimeError as exc:
        abort(500, str(exc))
    info["landmark_file"] = out.name
    return jsonify(info)


@bp.get("/summary/<name>")
def summary(name):
    seq = _load(name)
    det = seq.detected
    longest = run = 0
    for d in det:
        run = 0 if d else run + 1
        longest = max(longest, run)

    angles = pu.compute_joint_angles(seq.landmarks)
    angle_stats = {}
    for k, angle_name in enumerate(pu.JOINT_ANGLE_NAMES):
        col = angles[:, k]
        col = col[np.isfinite(col)]
        angle_stats[angle_name] = (
            {"min": round(float(col.min()), 1), "max": round(float(col.max()), 1),
             "mean": round(float(col.mean()), 1)} if col.size else None
        )

    return jsonify(
        frames=seq.num_frames,
        fps=seq.metadata.get("extraction_fps"),
        duration_s=round(float(seq.timestamps[-1]) / 1000.0, 3),
        detected_frames=int(det.sum()),
        detected_ratio=round(float(det.mean()), 4),
        longest_missing_run=int(longest),
        mean_visibility=round(float(seq.visibility[det].mean()), 3) if det.any() else None,
        angle_stats=angle_stats,
        metadata=seq.metadata,
    )


@bp.get("/pose/<name>/<int:i>")
def pose(name, i):
    """Normalized pose + joint angles for one frame.

    Query: mirror=1, yaw=1 (per-frame yaw alignment), min_vis=0.5 (visibility mask).
    """
    seq = _load(name)
    if not 0 <= i < seq.num_frames:
        abort(404, "Frame out of range")
    mirror = request.args.get("mirror") == "1"
    yaw = request.args.get("yaw") == "1"
    min_vis = request.args.get("min_vis", default=0.5, type=float)

    vis = seq.visibility[i]
    xyz = pu.mask_low_visibility(seq.landmarks[i], vis, min_visibility=min_vis)
    norm = pu.normalize_pose(xyz, align_yaw=yaw)
    angles = pu.compute_joint_angles(seq.landmarks[i], vis=vis, min_visibility=min_vis)
    if mirror:
        norm = pu.mirror_pose(norm)
        vis = pu.mirror_visibility(vis)
        angles = pu.mirror_angles(angles)

    return jsonify(
        frame=i,
        t_ms=round(float(seq.timestamps[i]), 2),
        detected=bool(seq.detected[i]),
        pose=_jsonable(norm),
        visibility=_jsonable(vis),
        angles=dict(zip(pu.JOINT_ANGLE_NAMES, _jsonable(angles))),
    )


@bp.get("/overlay/<name>")
def overlay(name):
    """All image-space landmarks, for drawing over the video in the browser."""
    seq = _load(name)
    return jsonify(
        fps=seq.metadata.get("extraction_fps"),
        timestamps=_jsonable(seq.timestamps),
        detected=seq.detected.tolist(),
        image=_jsonable(seq.image_landmarks[:, :, :2]),
    )


@bp.get("/media/<name>")
def media(name):
    """Serve a processed video from storage/annotated (supports range requests)."""
    return send_from_directory(_dir("annotated"), _safe(name))
