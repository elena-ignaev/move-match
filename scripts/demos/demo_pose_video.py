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

# -------------------- Video processing --------------------

def process_video(video_filename, model_path):
    video_path = Path(video_filename)
    model_path = Path(model_path)

    if not video_path.is_file():
        raise FileNotFoundError(
            f"Video file not found: {video_path}"
        )

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Pose model not found: {model_path}"
        )

    cap = cv.VideoCapture(str(video_path))

    if not cap.isOpened():
        cap.release()
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    detector = None

    try:
        fps = cap.get(cv.CAP_PROP_FPS)
        frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

        # Use a valid FPS for timestamp calculation.
        if not math.isfinite(fps) or fps <= 0:
            print(
                "Warning: Video FPS is unavailable. "
                "Using 30 FPS for timestamps and playback."
            )
            fps = 30.0

        print(f"Video FPS: {fps:.3f}")
        print(f"Reported frame count: {frame_count}")

        detector = create_pose_landmarker(str(model_path))

        frame_index = 0
        previous_timestamp_ms = -1
        frame_period = 1.0 / fps

        cv.namedWindow(WINDOW_NAME, cv.WINDOW_NORMAL)

        while True:
            ret, frame = cap.read()

            if not ret:
                if frame_count and frame_index < frame_count:
                    print(
                        "Warning: Video ended before the "
                        "reported frame count."
                    )
                else:
                    print("Reached end of video.")

                break

            # OpenCV BGR -> RGB for MediaPipe.
            rgb_frame = cv.cvtColor(
                frame, cv.COLOR_BGR2RGB
            )

            # MediaPipe Image uses SRGB for RGB input.
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame,
            )

            # Generate monotonically increasing timestamps.
            timestamp_ms = round(frame_index * 1000.0 / fps)
            timestamp_ms = max(
                timestamp_ms, previous_timestamp_ms + 1
            )
            previous_timestamp_ms = timestamp_ms

            # VIDEO mode: synchronous inference.
            result = detector.detect_for_video(
                mp_image, timestamp_ms
            )

            annotated_rgb = draw_landmarks_on_image(
                rgb_frame, result
            )

            # RGB -> BGR for OpenCV display.
            annotated_bgr = cv.cvtColor(
                annotated_rgb, cv.COLOR_RGB2BGR
            )

            cv.imshow(WINDOW_NAME, annotated_bgr)

            # Optionally limit display rate.
            if PLAY_VIDEO_AT_SOURCE_FPS:
                delay_ms = max(
                    1, round(1000.0 * frame_period)
                )
            else:
                delay_ms = max(
                    1, round(1000.0 / MAX_DISPLAY_FPS)
                )

            # Stop on q or if the window is closed.
            key = cv.waitKey(delay_ms) & 0xFF

            if key == ord("q"):
                print("Stopped by user.")
                break

            if cv.getWindowProperty(
                WINDOW_NAME, cv.WND_PROP_VISIBLE
            ) < 1:
                break

            frame_index += 1

    finally:
        cap.release()
        cv.destroyAllWindows()

        if detector is not None:
            detector.close()


if __name__ == "__main__":
    process_video(VIDEO_FILENAME, MODEL_PATH)