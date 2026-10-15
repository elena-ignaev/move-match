"""Visitor identity, ownership checks and simple upload abuse limits.

Local mode is single-user: everything stored belongs to the one person at the computer. In public
demo mode every browser gets a random, signed session id; videos and attempts record it, and a
visitor can only see their own items plus the prepared sample videos. There are no accounts, so
nobody can be impersonated by guessing a name, and clearing cookies simply starts a new visitor.
"""

from __future__ import annotations

import secrets
import threading
import time
from collections import defaultdict, deque

from flask import abort, current_app, request, session
from sqlalchemy import or_

from extensions import db
from models import Attempt, Video


def current_owner_id() -> str:
    """Random per-browser id kept in the signed session cookie."""
    owner = session.get("owner_id")
    if not owner:
        owner = secrets.token_urlsafe(18)
        session["owner_id"] = owner
        session.permanent = True
    return owner


def public_mode() -> bool:
    return bool(current_app.config.get("PUBLIC_DEMO"))


def visible_videos():
    query = Video.query
    if public_mode():
        query = query.filter(or_(Video.owner_id == current_owner_id(), Video.is_sample.is_(True)))
    return query


def visible_attempts():
    query = Attempt.query
    if public_mode():
        query = query.filter(Attempt.owner_id == current_owner_id())
    return query


def can_view_video(video: Video) -> bool:
    return not public_mode() or video.is_sample or video.owner_id == current_owner_id()


def can_modify_video(video: Video) -> bool:
    if not public_mode():
        return True
    return not video.is_sample and video.owner_id == current_owner_id()


def can_view_attempt(attempt: Attempt) -> bool:
    return not public_mode() or attempt.owner_id == current_owner_id()


def get_video_or_404(video_id: int, modify: bool = False) -> Video:
    """404 (not 403) for other visitors' items, so ids reveal nothing."""
    video = db.session.get(Video, video_id)
    if video is None or not can_view_video(video) or (modify and not can_modify_video(video)):
        abort(404)
    return video


def get_attempt_or_404(attempt_id: int) -> Attempt:
    attempt = db.session.get(Attempt, attempt_id)
    if attempt is None or not can_view_attempt(attempt):
        abort(404)
    return attempt


class RateLimiter:
    """In-memory sliding-window counter per (visitor, IP, action). Single process only.

    Good enough to stop accidental floods and casual abuse of a small demo; a multi-process
    deployment needs a shared store (e.g. Redis) instead.
    """

    def __init__(self) -> None:
        self._hits: dict[tuple, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, action: str, limit_per_hour: int) -> bool:
        if not limit_per_hour or limit_per_hour <= 0:
            return True
        key = (action, current_owner_id(), request.remote_addr or "?")
        now = time.monotonic()
        with self._lock:
            hits = self._hits[key]
            while hits and now - hits[0] > 3600:
                hits.popleft()
            if len(hits) >= limit_per_hour:
                return False
            hits.append(now)
            return True

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


rate_limiter = RateLimiter()


def security_headers(response):
    """Conservative headers. Scripts and styles come only from this site (no inline code)."""
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("Referrer-Policy", "same-origin")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Permissions-Policy", "camera=(self), microphone=(), geolocation=()")
    if not request.path.startswith("/dev/"):
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data: blob:; "
            "media-src 'self' blob:; connect-src 'self'; object-src 'none'; base-uri 'self'; "
            "frame-ancestors 'none'; form-action 'self'")
    return response
