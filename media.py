"""FFmpeg/ffprobe helpers: inspect real media contents and produce browser-ready files.

Why every video is transcoded to a CONSTANT frame rate
-----------------------------------------------------
Phone videos and browser (MediaRecorder) recordings usually have a variable frame rate: frame N is
not at N / fps seconds. The landmark files store one timestamp per frame, and the browser reports
playback position in seconds. If both are computed from a nominal FPS, they disagree by up to
several frames on a VFR file. Transcoding once to CFR makes "frame index / fps" exactly the
time the browser shows for that frame, for the file the learner actually watches.

MediaError messages are written for the person using the app and are safe to display. Anything
else raised here is an internal problem that belongs in the server log.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Callable


class MediaError(ValueError):
    """The file is not usable, for a reason the user can act on."""


class ToolMissingError(RuntimeError):
    """FFmpeg or ffprobe is not installed."""


def require_tools() -> None:
    missing = [tool for tool in ("ffmpeg", "ffprobe") if shutil.which(tool) is None]
    if missing:
        raise ToolMissingError(
            f"{' and '.join(missing)} not found on PATH. Install FFmpeg (see docs/setup.md).")


@dataclass(frozen=True)
class MediaInfo:
    duration_s: float
    width: int
    height: int
    fps: float                 # average frame rate (frames / duration)
    nominal_fps: float         # r_frame_rate: the container's declared rate
    has_audio: bool
    video_codec: str
    format_name: str
    start_time_s: float        # first video timestamp (usually 0)
    frame_interval_spread: float = 0.0   # (max - min) / median gap between frame timestamps, first 20 s

    @property
    def variable_frame_rate(self) -> bool:
        """Header rates can lie (Matroska reports a nominal rate), so measured spacing decides."""
        if self.frame_interval_spread > 0.1:
            return True
        return self.nominal_fps > 0 and abs(self.fps - self.nominal_fps) / self.nominal_fps > 0.002


def _rate(text: str | None) -> float:
    try:
        value = Fraction(text or "0")
    except (ValueError, ZeroDivisionError):
        return 0.0
    return float(value) if value > 0 else 0.0


def _float(value, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if out == out else default       # NaN -> default


def probe(path: str | Path, timeout_s: float = 30.0) -> MediaInfo:
    """Inspect the file's actual streams. Raises MediaError if it is not a readable video."""
    require_tools()
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise MediaError("The file is empty.")
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
            capture_output=True, text=True, timeout=timeout_s, check=False)
    except subprocess.TimeoutExpired as exc:
        raise MediaError("The file took too long to read. Try a shorter MP4 video.") from exc
    if result.returncode != 0:
        raise MediaError("This file could not be read as a video. Try an MP4 (H.264) file.")
    try:
        data = json.loads(result.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise MediaError("This file could not be read as a video.") from exc

    streams = data.get("streams", [])
    videos = [s for s in streams if s.get("codec_type") == "video"
              and s.get("disposition", {}).get("attached_pic", 0) == 0]
    if not videos:
        raise MediaError("This file has no video track.")
    v = videos[0]
    fmt = data.get("format", {})
    width, height = int(v.get("width") or 0), int(v.get("height") or 0)
    if width <= 0 or height <= 0:
        raise MediaError("The video's picture size could not be read.")

    duration = _float(v.get("duration")) or _float(fmt.get("duration"))
    nb_frames = int(_float(v.get("nb_frames")))
    avg = _rate(v.get("avg_frame_rate"))
    nominal = _rate(v.get("r_frame_rate"))
    if not avg and nb_frames and duration:
        avg = nb_frames / duration
    # MediaRecorder WebM files often have no duration in the header: count packets instead.
    if duration <= 0:
        duration = _count_duration(path, timeout_s)
    if duration <= 0:
        raise MediaError("The video's length could not be read.")
    spread = _frame_interval_spread(path, timeout_s)
    return MediaInfo(
        duration_s=duration, width=width, height=height,
        fps=avg or nominal or 30.0, nominal_fps=nominal or avg or 30.0,
        has_audio=any(s.get("codec_type") == "audio" for s in streams),
        video_codec=str(v.get("codec_name", "")), format_name=str(fmt.get("format_name", "")),
        start_time_s=_float(v.get("start_time")),
        frame_interval_spread=spread,
    )


def _frame_interval_spread(path: Path, timeout_s: float) -> float:
    """How irregular the frame timestamps are: (max gap - min gap) / median gap. 0 = constant rate."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-read_intervals", "%+20",
             "-show_entries", "packet=pts_time", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=timeout_s, check=False)
    except subprocess.TimeoutExpired:
        return 0.0
    times = sorted(t for t in (_float(line.split(",")[0], -1.0) for line in result.stdout.splitlines()) if t >= 0)
    if len(times) < 3:
        return 0.0
    gaps = [b - a for a, b in zip(times[:-1], times[1:]) if b > a]
    if len(gaps) < 2:
        return 0.0
    gaps.sort()
    median = gaps[len(gaps) // 2]
    return (gaps[-1] - gaps[0]) / median if median > 0 else 0.0


def _count_duration(path: Path, timeout_s: float) -> float:
    """Duration from the last video packet timestamp (slow path for headerless WebM)."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts_time",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=timeout_s, check=False)
    except subprocess.TimeoutExpired:
        return 0.0
    times = [_float(line.split(",")[0], -1.0) for line in result.stdout.splitlines() if line.strip()]
    times = [t for t in times if t >= 0]
    return max(times) if times else 0.0


def validate(info: MediaInfo, *, max_seconds: float, max_dimension: int, min_seconds: float = 1.0) -> None:
    """Enforce duration and resolution limits (file size is enforced at upload)."""
    if info.duration_s < min_seconds:
        raise MediaError(f"The video is shorter than {min_seconds:g} second.")
    if info.duration_s > max_seconds + 0.5:
        raise MediaError(
            f"The video is {info.duration_s:.0f} s long. MoveMatch works with short clips of up to "
            f"{max_seconds:.0f} s: trim it to the part you want to learn.")
    if max(info.width, info.height) > max_dimension:
        raise MediaError(
            f"The video is {info.width}x{info.height}. Use a video of at most {max_dimension} pixels "
            "on its longest side (1080p or smaller).")


def target_fps(info: MediaInfo, max_fps: int) -> int:
    """Whole-number CFR rate for the transcode: the source's average rate, capped."""
    fps = int(round(info.fps)) if info.fps > 0 else max_fps
    return max(10, min(max_fps, fps))


def _run_ffmpeg(args: list[str], duration_s: float, progress: Callable[[float], None] | None,
                timeout_s: float) -> None:
    """Run ffmpeg, reporting progress 0..1 from its -progress output."""
    require_tools()
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
               "-progress", "pipe:1", *args]
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            if progress and line.startswith("out_time_us=") and duration_s > 0:
                us = _float(line.split("=", 1)[1], 0.0)
                progress(min(1.0, max(0.0, us / 1e6 / duration_s)))
        _, stderr = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        raise RuntimeError("FFmpeg timed out")
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg failed ({proc.returncode}): {stderr.strip()[-2000:]}")


def transcode_reference(src: str | Path, dst: str | Path, *, info: MediaInfo, fps: int,
                        max_height: int, max_seconds: float,
                        progress: Callable[[float], None] | None = None) -> Path:
    """Browser-friendly reference: H.264 video at a constant frame rate + the ORIGINAL audio (AAC).

    The music is kept: it is what the learner dances to, and because it lives in the same file as
    the picture it stays in sync at any playbackRate.
    """
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.stem + ".part.mp4")
    args = [
        "-i", str(src), "-t", f"{max_seconds:.3f}",
        "-map", "0:v:0", "-map", "0:a:0?",
        "-vf", f"scale=-2:'min({max_height},ih)':flags=bicubic,format=yuv420p",
        "-fps_mode", "cfr", "-r", str(fps),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k", "-ac", "2",
        "-movflags", "+faststart", str(tmp),
    ]
    try:
        _run_ffmpeg(args, min(info.duration_s, max_seconds), progress, timeout_s=max(120.0, info.duration_s * 10))
        if not tmp.is_file() or tmp.stat().st_size == 0:
            raise RuntimeError("FFmpeg produced no output")
        tmp.replace(dst)
    finally:
        tmp.unlink(missing_ok=True)
    return dst


def transcode_recording(src: str | Path, dst: str | Path, *, info: MediaInfo, fps: int = 30,
                        max_height: int = 720,
                        progress: Callable[[float], None] | None = None) -> Path:
    """A learner recording -> seekable CFR MP4 (no audio: the camera stream is recorded silently).

    MediaRecorder WebM files often cannot be seeked and have irregular frame timing; after this,
    frame i of the output is at i / fps seconds of the recording timeline. FFmpeg shifts the first
    frame to t=0, so callers add info.start_time_s to map back to the recorder clock.
    """
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.stem + ".part.mp4")
    args = [
        "-i", str(src), "-map", "0:v:0", "-an",
        "-vf", f"scale=-2:'min({max_height},ih)':flags=bicubic,format=yuv420p",
        "-fps_mode", "cfr", "-r", str(fps),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "24",
        "-movflags", "+faststart", str(tmp),
    ]
    try:
        _run_ffmpeg(args, info.duration_s, progress, timeout_s=max(120.0, info.duration_s * 10))
        if not tmp.is_file() or tmp.stat().st_size == 0:
            raise RuntimeError("FFmpeg produced no output")
        tmp.replace(dst)
    finally:
        tmp.unlink(missing_ok=True)
    return dst
