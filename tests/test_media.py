"""FFmpeg/ffprobe layer: content validation, constant-frame-rate transcodes with audio."""

import pytest

import media
from conftest import needs_ffmpeg

pytestmark = needs_ffmpeg


def test_probe_reads_real_streams(media_dir):
    info = media.probe(media_dir / "reference.mp4")
    assert (info.width, info.height) == (320, 240)
    assert info.has_audio and info.duration_s == pytest.approx(6.0, abs=0.1)
    assert not info.variable_frame_rate


def test_variable_frame_rate_is_measured(media_dir):
    assert media.probe(media_dir / "vfr.mkv").variable_frame_rate


@pytest.mark.parametrize("name,message", [("corrupt.mp4", "could not be read"), ("audio_only.m4a", "no video track")])
def test_probe_rejects_non_videos(media_dir, name, message):
    with pytest.raises(media.MediaError, match=message):
        media.probe(media_dir / name)


def test_probe_rejects_empty_file(tmp_path):
    (tmp_path / "empty.mp4").write_bytes(b"")
    with pytest.raises(media.MediaError, match="empty"):
        media.probe(tmp_path / "empty.mp4")


def test_validate_limits(media_dir):
    info = media.probe(media_dir / "reference.mp4")
    media.validate(info, max_seconds=10, max_dimension=1920)
    with pytest.raises(media.MediaError, match="short clips"):
        media.validate(info, max_seconds=3, max_dimension=1920)
    with pytest.raises(media.MediaError, match="pixels"):
        media.validate(info, max_seconds=10, max_dimension=200)


def test_reference_transcode_keeps_audio_and_is_cfr(media_dir, tmp_path):
    src = media_dir / "vfr.mkv"
    info = media.probe(src)
    progress = []
    out = media.transcode_reference(src, tmp_path / "out.mp4", info=info, fps=30, max_height=720,
                                    max_seconds=90, progress=progress.append)
    result = media.probe(out)
    assert not result.variable_frame_rate and result.fps == pytest.approx(30, abs=0.01)
    assert progress and max(progress) <= 1.0

    with_audio = media.transcode_reference(media_dir / "reference.mp4", tmp_path / "a.mp4",
                                           info=media.probe(media_dir / "reference.mp4"), fps=30,
                                           max_height=720, max_seconds=90)
    assert media.probe(with_audio).has_audio
    assert not list(tmp_path.glob("*.part.mp4"))


def test_recording_transcode_is_seekable_mp4(media_dir, tmp_path):
    src = media_dir / "attempt.webm"
    out = media.transcode_recording(src, tmp_path / "r.mp4", info=media.probe(src))
    info = media.probe(out)
    assert info.format_name.startswith("mov") and not info.has_audio
    assert info.duration_s == pytest.approx(8.0, abs=0.1)


def test_target_fps_is_capped(media_dir):
    info = media.probe(media_dir / "reference.mp4")
    assert media.target_fps(info, 30) == 30
    assert media.target_fps(info, 24) == 24
