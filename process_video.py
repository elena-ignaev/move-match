import cv2
import mediapipe as mp
from pathlib import Path
import subprocess

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "pretrained_models" / "pose_landmarker_lite.task"

# MediaPipe Pose Landmarker landmark connections, indexed by the 33 pose landmarks.
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (27, 29), (29, 31),
    (27, 31), (24, 26), (26, 28), (28, 30), (30, 32), (28, 32)
]

def convert_for_browser(input_path, output_path):
    """Convert an OpenCV-generated video to browser-friendly H.264 MP4."""

    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()

    if not input_path.is_file():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    # Ensure the output directory exists.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "ffmpeg",
        "-y",
        "-i", str(input_path),
        "-map", "0:v:0",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-an",
        str(output_path),
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg conversion failed:\n{result.stderr}"
        )

    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError("Converted video was not created correctly.")

    return str(output_path)


def draw_pose(frame, pose_landmarks):
    """Draw normalized MediaPipe pose landmarks and connections onto a BGR frame."""
    height, width = frame.shape[:2]
    points = []
    for landmark in pose_landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        # Keep coordinates in the image bounds for safe drawing.
        points.append((max(0, min(width - 1, x)), max(0, min(height - 1, y))))

    for start, end in POSE_CONNECTIONS:
        if start < len(points) and end < len(points):
            cv2.line(frame, points[start], points[end], (0, 220, 80), 2, cv2.LINE_AA)
    for point in points:
        cv2.circle(frame, point, 3, (0, 100, 255), -1, cv2.LINE_AA)
    return frame


def analyze_video(input_path: Path, output_path: Path):
    """Run Pose Landmarker in VIDEO mode and write an annotated video."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            "Pose Landmarker model is missing. Download pose_landmarker_lite.task "
            "and place it next to app.py. See README.md."
        )

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError("OpenCV could not open this video. Try an MP4 encoded with H.264.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or fps > 240:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        raise ValueError("Could not read the video's dimensions.")

    # mp4v is commonly available with OpenCV. Browser playback depends on installed codecs.
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not create the annotated video output file.")

    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    frame_index = 0
    try:
        with PoseLandmarker.create_from_options(options) as landmarker:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                # VIDEO mode requires monotonically increasing timestamps in milliseconds.
                timestamp_ms = int(frame_index * 1000.0 / fps)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect_for_video(mp_image, timestamp_ms)
                if result.pose_landmarks:
                    frame = draw_pose(frame, result.pose_landmarks[0])
                writer.write(frame)
                frame_index += 1
    finally:
        cap.release()
        writer.release()

    if frame_index == 0:
        raise ValueError("No readable frames were found in this video.")

    browser_path = output_path.with_name(
        f"{output_path.stem}_browser.mp4"
    )
    convert_for_browser(output_path, browser_path)

    if not browser_path.is_file() or browser_path.stat().st_size == 0:
        raise RuntimeError(f"Browser-compatible video was not created: {browser_path}")

    return {"frame_count": frame_index, "fps": fps, "width": width, "height": height, "output_path": str(browser_path)}
