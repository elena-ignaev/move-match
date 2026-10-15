"""Mapping recording time to the reference time that was on screen."""

import numpy as np
import pytest

from conftest import sync_payload
from sync_timeline import SyncError, build_timeline, sanitize_payload


def timeline(**kw):
    return build_timeline({"sync": sanitize_payload(sync_payload(**kw))})


def test_half_speed_mapping_with_lead_in():
    tl = timeline(speed=0.5, start=2.0, end=6.0, lead_in_ms=1000, recorder_start_ms=200)
    ref = tl.reference_times(np.array([0.5, 1.0, 3.0, 5.0]))
    assert np.isnan(ref[0])                                    # before the reference started
    assert ref[1:] == pytest.approx([2.0, 3.0, 4.0], abs=0.02)


def test_matching_frame_numbers_would_be_wrong():
    """At 0.5x, recording frame N is NOT reference frame N: the map must say so."""
    tl = timeline(speed=0.5, start=0.0, end=4.0, lead_in_ms=0, recorder_start_ms=0)
    assert tl.reference_times(np.array([4.0]))[0] == pytest.approx(2.0, abs=0.02)


def test_stall_breaks_the_mapping():
    tl = timeline(speed=1.0, start=0.0, end=4.0, lead_in_ms=0, recorder_start_ms=0, stall=(2.0, 1500))
    assert len(tl.stretches) == 2
    during = tl.reference_times(np.array([2.6, 3.0]))
    assert np.isnan(during).all()
    after = tl.reference_times(np.array([4.0]))[0]
    assert after == pytest.approx(2.5, abs=0.05)               # playback resumed where it froze


def test_seek_back_for_a_loop_starts_a_new_stretch():
    payload = sync_payload(speed=1.0, start=1.0, end=2.0, lead_in_ms=0, recorder_start_ms=0)
    last_session = payload["samples"][-1][0]
    payload["samples"] += [[last_session + 33 + i * 33.3, 1.0 + i * 0.0333] for i in range(20)]
    tl = build_timeline({"sync": sanitize_payload(payload)})
    assert len(tl.stretches) == 2


def test_legacy_attempts_use_events():
    details = {"sync_events": [
        {"type": "recording_start", "session_time_ms": 0, "reference_time": 0, "playback_rate": 0.5},
        {"type": "reference_play", "session_time_ms": 100, "reference_time": 0.0, "playback_rate": 0.5},
        {"type": "reference_ended", "session_time_ms": 8100, "reference_time": 4.0, "playback_rate": 0.5},
    ]}
    tl = build_timeline(details)
    assert tl.source == "events"
    assert tl.reference_times(np.array([4.1]))[0] == pytest.approx(2.0, abs=0.01)


def test_recording_offset_is_applied():
    tl = build_timeline({"sync": sanitize_payload(sync_payload(speed=1, start=0, end=3, lead_in_ms=0,
                                                               recorder_start_ms=0))}, recording_offset_s=0.5)
    assert tl.reference_times(np.array([1.0]))[0] == pytest.approx(1.5, abs=0.02)


def test_missing_data_raises():
    with pytest.raises(SyncError):
        build_timeline({})
    with pytest.raises(SyncError):
        build_timeline({"sync": {"recorder_start_ms": 0, "samples": [], "events": []}})


def test_sanitize_bounds_and_drops_junk():
    raw = {"recorder_start_ms": "12", "samples": [[1, 2], ["x", 3], [float("nan"), 1]] + [[i, i] for i in range(30000)],
           "events": [{"type": "a" * 100, "session_time_ms": 1}], "evil": "<script>"}
    clean = sanitize_payload(raw)
    assert clean["recorder_start_ms"] == 12.0
    assert len(clean["samples"]) <= 20000 and clean["samples"][0] == [1.0, 2.0]
    assert len(clean["events"][0]["type"]) == 40 and "evil" not in clean
