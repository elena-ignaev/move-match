"""Map each moment of a learner recording to the reference time that was on screen.

Matching "learner frame N" to "reference frame N" is wrong as soon as anything is irregular: the
practice speed is not 1x, the reference buffers for half a second, the camera drops frames, or the
recording started a moment before the music. Instead the browser records a session timeline on one
monotonic clock (performance.now) while recording:

    recorder_start_ms   when MediaRecorder actually started (its "start" event)
    samples             [session_ms, reference_time_s] for every reference frame presented while
                        playing (requestVideoFrameCallback), so stalls and seeks are visible
    events              play / pause / waiting / playing / seeked / ended / stop, with reference time
                        and playback rate (the only data older attempts have)

A learner frame at recording time t (seconds in the CFR transcode) happened at session time
    s = recorder_start_ms + (t + recording_offset_s) * 1000
and is mapped to a reference time by interpolating between samples of the SAME continuous playing
stretch. Frames during a stall, a pause, before the reference started or after it stopped get NaN:
the learner was not being shown a moving reference, so they are not compared with one.

Pure NumPy, no Flask: unit-testable and reusable offline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SYNC_VERSION = 2
MAX_SAMPLE_GAP_MS = 600.0      # longer without a presented frame = a stall: break the stretch
MAX_EXTRAPOLATE_MS = 120.0     # allowed beyond the first/last sample of a stretch
_START_TYPES = {"reference_play", "reference_playing"}
_STOP_TYPES = {"reference_pause", "reference_waiting", "reference_ended", "reference_seeking",
               "manual_stop", "recording_stop", "section_end"}


class SyncError(ValueError):
    """The attempt has no usable synchronization data."""


@dataclass(frozen=True)
class Stretch:
    """One continuous stretch of reference playback: session ms -> reference seconds."""
    session_ms: np.ndarray
    ref_s: np.ndarray

    @property
    def rate(self) -> float:
        span = self.session_ms[-1] - self.session_ms[0]
        return float((self.ref_s[-1] - self.ref_s[0]) * 1000.0 / span) if span > 0 else 0.0


@dataclass(frozen=True)
class SyncTimeline:
    stretches: tuple[Stretch, ...]
    recorder_start_ms: float
    recording_offset_s: float = 0.0
    source: str = "samples"          # "samples" (dense) or "events" (legacy)

    def reference_times(self, recording_times_s) -> np.ndarray:
        """Reference time (s) for each recording time (s); NaN where no reference was playing."""
        t = np.asarray(recording_times_s, dtype=np.float64)
        session = self.recorder_start_ms + (t + self.recording_offset_s) * 1000.0
        out = np.full(t.shape, np.nan)
        for st in self.stretches:
            lo, hi = st.session_ms[0], st.session_ms[-1]
            inside = (session >= lo) & (session <= hi)
            if inside.any():
                out[inside] = np.interp(session[inside], st.session_ms, st.ref_s)
            rate = st.rate
            if rate > 0:
                before = (session < lo) & (session >= lo - MAX_EXTRAPOLATE_MS) & np.isnan(out)
                after = (session > hi) & (session <= hi + MAX_EXTRAPOLATE_MS) & np.isnan(out)
                out[before] = st.ref_s[0] - (lo - session[before]) / 1000.0 * rate
                out[after] = st.ref_s[-1] + (session[after] - hi) / 1000.0 * rate
        return out

    def replay_map(self, duration_s: float, step_s: float = 0.1) -> list[list[float | None]]:
        """[[recording_s, reference_s or None], ...] for the replay page."""
        t = np.arange(0.0, max(duration_s, 0.0) + step_s / 2, step_s)
        r = self.reference_times(t)
        return [[round(float(a), 3), None if not np.isfinite(b) else round(float(b), 4)] for a, b in zip(t, r)]

    def playing_seconds(self) -> float:
        return float(sum((st.session_ms[-1] - st.session_ms[0]) / 1000.0 for st in self.stretches))


def _split_samples(samples: np.ndarray, max_rate: float = 2.5) -> list[Stretch]:
    """Cut the sample list wherever playback was not continuous (gap, seek, or stall)."""
    samples = samples[np.argsort(samples[:, 0], kind="stable")]
    stretches: list[Stretch] = []
    current = [samples[0]]
    frozen_ms = 0.0               # how long the reference time has not advanced (rAF fallback sampling)
    for prev, cur in zip(samples[:-1], samples[1:]):
        ds = cur[0] - prev[0]
        dr = cur[1] - prev[1]
        frozen_ms = frozen_ms + ds if abs(dr) < 1e-4 else 0.0
        continuous = (0 < ds <= MAX_SAMPLE_GAP_MS and dr >= -1e-3
                      and dr <= ds / 1000.0 * max_rate + 0.05 and frozen_ms <= 300.0)
        if continuous:
            current.append(cur)
        else:
            stretches.append(current)
            current = [cur]
            frozen_ms = 0.0
    stretches.append(current)

    out = []
    for group in stretches:
        g = np.asarray(group)
        if len(g) < 2:
            continue
        # Drop a frozen tail/head (reference stalled while samples kept coming).
        moving = np.diff(g[:, 1]) > 1e-4
        if not moving.any():
            continue
        first = int(np.argmax(moving))
        last = len(moving) - int(np.argmax(moving[::-1]))
        g = g[first:last + 1]
        out.append(Stretch(g[:, 0].copy(), np.maximum.accumulate(g[:, 1])))
    return out


def _stretches_from_events(events: list[dict]) -> list[Stretch]:
    """Legacy attempts: playing segments from play/pause events, assuming constant rate inside."""
    evs = sorted((e for e in events if isinstance(e, dict) and _num(e.get("session_time_ms")) is not None),
                 key=lambda e: float(e["session_time_ms"]))
    out = []
    start = None
    for e in evs:
        kind = e.get("type")
        s = float(e["session_time_ms"])
        r = _num(e.get("reference_time"))
        rate = _num(e.get("playback_rate")) or 1.0
        if kind in _START_TYPES and r is not None:
            start = (s, r, rate)
        elif kind in _STOP_TYPES and start is not None:
            s0, r0, rate0 = start
            r1 = r if r is not None else r0 + (s - s0) / 1000.0 * rate0
            if s > s0 and r1 > r0:
                out.append(Stretch(np.array([s0, s]), np.array([r0, r1])))
            start = None
    return out


def _num(value) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def build_timeline(details: dict, recording_offset_s: float = 0.0) -> SyncTimeline:
    """SyncTimeline from an attempt's stored details (dense samples if present, else events)."""
    if not isinstance(details, dict):
        raise SyncError("No synchronization data was saved with this attempt.")
    sync = details.get("sync") if isinstance(details.get("sync"), dict) else {}
    events = sync.get("events") or details.get("sync_events") or []

    raw = sync.get("samples") or []
    samples = np.array([[float(a), float(b)] for a, b in
                        (s[:2] for s in raw if isinstance(s, (list, tuple)) and len(s) >= 2)
                        if _num(a) is not None and _num(b) is not None], dtype=np.float64).reshape(-1, 2)

    recorder_start = _num(sync.get("recorder_start_ms"))
    if recorder_start is None:
        starts = [e for e in events if isinstance(e, dict) and e.get("type") in ("recorder_started", "recording_start")]
        recorder_start = _num(starts[0].get("session_time_ms")) if starts else None
    if recorder_start is None:
        raise SyncError("The recording start time was not saved, so it cannot be lined up with the reference.")

    if len(samples) >= 2:
        stretches, source = _split_samples(samples), "samples"
    else:
        stretches, source = _stretches_from_events(events), "events"
    if not stretches:
        raise SyncError("The reference never played while recording.")
    return SyncTimeline(tuple(stretches), recorder_start, recording_offset_s, source)


def sanitize_payload(raw: dict, max_samples: int = 20000, max_events: int = 500) -> dict:
    """Keep only the numeric fields we use from client-supplied sync data (bounded size)."""
    if not isinstance(raw, dict):
        return {}
    samples = []
    for s in (raw.get("samples") or [])[:max_samples]:
        if isinstance(s, (list, tuple)) and len(s) >= 2:
            a, b = _num(s[0]), _num(s[1])
            if a is not None and b is not None:
                samples.append([round(a, 2), round(b, 4)])
    events = []
    for e in (raw.get("events") or [])[:max_events]:
        if isinstance(e, dict):
            events.append({
                "type": str(e.get("type", ""))[:40],
                "session_time_ms": _num(e.get("session_time_ms")),
                "reference_time": _num(e.get("reference_time")),
                "playback_rate": _num(e.get("playback_rate")),
            })
    return {
        "version": SYNC_VERSION,
        "recorder_start_ms": _num(raw.get("recorder_start_ms")),
        "samples": samples,
        "events": events,
        "clock": "performance.now() ms since the practice session started",
    }
