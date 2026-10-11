import mediapipe as mp

import cv2 as cv

import time
import math
from pathlib import Path

# -------------------- Configuration --------------------

VIDEO_FILENAME = "examples/dance.mp4"
MODEL_PATH = "pretrained_models/pose_landmarker_lite.task"

WINDOW_NAME = "MediaPipe Pose Landmarker"
PLAY_VIDEO_AT_SOURCE_FPS = True
MAX_DISPLAY_FPS = 60

# -------------------- MediaPipe setup --------------------

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


def create_pose_landmarker(model_path):
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=model_path
        ),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )
    return PoseLandmarker.create_from_options(options)

# VIsualization
import numpy as np
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles
from mediapipe.tasks.python import vision

def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  pose_landmark_style = drawing_styles.get_default_pose_landmarks_style()
  pose_connection_style = drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2)

  for pose_landmarks in pose_landmarks_list:
    drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=pose_landmarks,
        connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
        landmark_drawing_spec=pose_landmark_style,
        connection_drawing_spec=pose_connection_style)

  return annotated_image


def process_video(video_filename, model_path, output_path=None):
    video_path = Path(video_filename)
    model_path = Path(model_path)

    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if not model_path.is_file():
        raise FileNotFoundError(f"Pose model not found: {model_path}")

    # Default output: annotated/<original_filename>_annotated.mp4
    if output_path is None:
        output_dir = video_path.parent / "annotated"
        output_path = output_dir / f"{video_path.stem}_annotated.mp4"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv.VideoCapture(str(video_path))

    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open video: {video_path}")

    detector = None
    writer = None
    success = False

    try:
        fps = cap.get(cv.CAP_PROP_FPS)
        frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

        if not math.isfinite(fps) or fps <= 0:
            print("Warning: Invalid FPS; using 30 FPS.")
            fps = 30.0

        width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

        if width <= 0 or height <= 0:
            raise RuntimeError("Could not determine video dimensions.")

        print(f"Input: {video_path}")
        print(f"Output: {output_path}")
        print(f"FPS: {fps:.3f}; frames: {frame_count}")

        # MP4 container with MPEG-4 Part 2 encoding.
        # For broad browser compatibility, H.264 transcoding may
        # be needed after processing.
        fourcc = cv.VideoWriter_fourcc(*"mp4v")
        writer = cv.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            (width, height),
        )

        if not writer.isOpened():
            raise RuntimeError(
                f"Could not create output video: {output_path}"
            )

        detector = create_pose_landmarker(str(model_path))

        frame_index = 0
        previous_timestamp_ms = -1

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            # OpenCV BGR -> RGB for MediaPipe.
            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame,
            )

            # Use source timestamps, rounded to milliseconds.
            timestamp_ms = round(frame_index * 1000.0 / fps)
            timestamp_ms = max(
                timestamp_ms, previous_timestamp_ms + 1
            )
            previous_timestamp_ms = timestamp_ms

            result = detector.detect_for_video(
                mp_image, timestamp_ms
            )

            # Your helper returns an annotated RGB image.
            annotated_rgb = draw_landmarks_on_image(
                rgb_frame, result
            )

            # RGB -> BGR for OpenCV's VideoWriter.
            annotated_bgr = cv.cvtColor(
                annotated_rgb, cv.COLOR_RGB2BGR
            )

            if annotated_bgr.shape[:2] != (height, width):
                raise RuntimeError(
                    "Annotated frame dimensions changed."
                )

            writer.write(annotated_bgr)
            frame_index += 1

            if frame_index % 100 == 0:
                print(f"Processed {frame_index} frames...")

        if frame_index == 0:
            raise RuntimeError("The input video contains no readable frames.")

        success = True
        print(f"Finished processing {frame_index} frames.")

        return {
            "output_path": str(output_path.resolve()),
            "frames_processed": frame_index,
            "fps": fps,
            "width": width,
            "height": height,
        }

    finally:
        cap.release()

        if writer is not None:
            writer.release()

        if detector is not None:
            detector.close()

        # Remove an incomplete output if processing failed.
        if not success and output_path.exists():
            output_path.unlink()

if __name__ == "__main__":
    result = process_video(
        video_filename="static/uploads/dance.mp4",
        model_path="pretrained_models/pose_landmarker_lite.task",
        output_path="static/annotated/dance_annotated.mp4",
    )

    print(result["output_path"])
    print(result["frames_processed"])
