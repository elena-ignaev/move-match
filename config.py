"""Application configuration, read from environment variables (see .env.example).

Two deployment modes:

* local (default): one person on their own computer. Every video is visible, uploads are allowed.
* public demo (MOVEMATCH_PUBLIC_DEMO=1): visitors only see prepared sample videos and their own
  attempts, recordings are deleted after RETENTION_HOURS, and a real SECRET_KEY is required.
"""

from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DEV_SECRET = "dev-only-change-me"


def _bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return default if raw is None or raw.strip() == "" else int(raw)


def _float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return default if raw is None or raw.strip() == "" else float(raw)


class ConfigError(RuntimeError):
    """The configuration is unsafe or incomplete; the app refuses to start."""


def load_config(overrides: dict | None = None) -> dict:
    """Build the Flask config dict. `overrides` (used by tests) win over the environment."""
    env = os.environ.get("MOVEMATCH_ENV", "development").strip().lower()
    public_demo = _bool("MOVEMATCH_PUBLIC_DEMO", False)
    storage_dir = Path(os.environ.get("MOVEMATCH_STORAGE_DIR", BASE_DIR / "storage"))

    cfg = {
        "ENV_NAME": env,
        "PUBLIC_DEMO": public_demo,
        "DEBUG": _bool("MOVEMATCH_DEBUG", False),
        "SECRET_KEY": os.environ.get("SECRET_KEY", DEV_SECRET),
        "SQLALCHEMY_DATABASE_URI": os.environ.get(
            "DATABASE_URL", f"sqlite:///{(BASE_DIR / 'videos.sqlite3').as_posix()}"
        ),
        "SQLALCHEMY_TRACK_MODIFICATIONS": False,
        "STORAGE_DIR": storage_dir,
        "MODEL_PATH": Path(os.environ.get(
            "MOVEMATCH_MODEL_PATH", BASE_DIR / "pretrained_models" / "pose_landmarker_lite.task")),

        # Upload limits. MAX_CONTENT_LENGTH is the hard cap enforced by Werkzeug for any request.
        "MAX_REFERENCE_MB": _int("MOVEMATCH_MAX_REFERENCE_MB", 200),
        "MAX_ATTEMPT_MB": _int("MOVEMATCH_MAX_ATTEMPT_MB", 150),
        "MAX_REFERENCE_SECONDS": _float("MOVEMATCH_MAX_REFERENCE_SECONDS", 90.0),
        "MAX_ATTEMPT_SECONDS": _float("MOVEMATCH_MAX_ATTEMPT_SECONDS", 240.0),
        "MAX_SOURCE_DIMENSION": _int("MOVEMATCH_MAX_SOURCE_DIMENSION", 1920),
        "PLAYBACK_MAX_HEIGHT": _int("MOVEMATCH_PLAYBACK_MAX_HEIGHT", 720),
        "PLAYBACK_MAX_FPS": _int("MOVEMATCH_PLAYBACK_MAX_FPS", 30),

        # Abuse controls (per visitor, per hour; 0 disables a limit).
        "REFERENCE_UPLOADS_PER_HOUR": _int("MOVEMATCH_REFERENCE_UPLOADS_PER_HOUR", 0 if not public_demo else 3),
        "ATTEMPT_UPLOADS_PER_HOUR": _int("MOVEMATCH_ATTEMPT_UPLOADS_PER_HOUR", 0 if not public_demo else 20),
        "ALLOW_REFERENCE_UPLOADS": _bool("MOVEMATCH_ALLOW_REFERENCE_UPLOADS", not public_demo),
        "RETENTION_HOURS": _float("MOVEMATCH_RETENTION_HOURS", 0 if not public_demo else 24),

        # Background processing. JOBS_SYNC runs jobs inside the request (tests only).
        "JOBS_SYNC": False,
        "JOB_WORKERS": _int("MOVEMATCH_JOB_WORKERS", 1),

        "WTF_CSRF_TIME_LIMIT": None,          # token lives as long as the session
        "SESSION_COOKIE_HTTPONLY": True,
        "SESSION_COOKIE_SAMESITE": "Lax",
        "SESSION_COOKIE_SECURE": env == "production",
        "PERMANENT_SESSION_LIFETIME": 60 * 60 * 24 * 30,

        # Test hook: a callable(kind) -> pose landmarker. None means MediaPipe.
        "LANDMARKER_FACTORY": None,
    }
    if overrides:
        cfg.update(overrides)
    cfg["MAX_CONTENT_LENGTH"] = max(cfg["MAX_REFERENCE_MB"], cfg["MAX_ATTEMPT_MB"]) * 1024 * 1024
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict) -> None:
    """Refuse unsafe combinations instead of silently running with them."""
    exposed = cfg["ENV_NAME"] == "production" or cfg["PUBLIC_DEMO"]
    if exposed:
        if not cfg["SECRET_KEY"] or cfg["SECRET_KEY"] == DEV_SECRET or len(cfg["SECRET_KEY"]) < 32:
            raise ConfigError(
                "Set SECRET_KEY to a random value of at least 32 characters before running in production "
                "or as a public demo, e.g. python -c \"import secrets; print(secrets.token_hex(32))\"")
        if cfg["DEBUG"]:
            raise ConfigError("MOVEMATCH_DEBUG must be off in production or public-demo mode.")
