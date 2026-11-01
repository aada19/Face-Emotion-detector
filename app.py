import os

# Must be set before importing TensorFlow / DeepFace
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import time
import threading

import av
import cv2
import streamlit as st

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from deepface import DeepFace


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Face Emotion Detector",
    page_icon="😊",
    layout="centered"
)


# ============================================================
# TITLE
# ============================================================

st.title("😊 AI Face Emotion Detector")

st.write(
    "Allow camera access and the AI will estimate the facial "
    "expression visible on camera."
)


# ============================================================
# EMOTION DETECTOR
# ============================================================

class EmotionDetector(VideoProcessorBase):

    def __init__(self):
        self.emotion = "Waiting for face..."
        self.confidence = 0.0
        self.lock = threading.Lock()
        self.last_check = 0.0

    def recv(self, frame):

        # Convert WebRTC frame to OpenCV image
        image = frame.to_ndarray(format="bgr24")

        # Mirror webcam
        image = cv2.flip(image, 1)

        # Analyze approximately every 2 seconds
        now = time.time()

        if now - self.last_check >= 2:

            self.last_check = now

            try:

                result = DeepFace.analyze(
                    img_path=image,
                    actions=["emotion"],
                    enforce_detection=False,
                    detector_backend="opencv",
                    silent=True
                )

                # DeepFace normally returns a list
                if isinstance(result, list):
                    result = result[0]

                emotion_data = result.get("emotion", {})
                dominant = result.get(
                    "dominant_emotion",
                    "Unknown"
                )

                confidence = float(
                    emotion_data.get(dominant, 0.0)
                )

                with self.lock:
                    self.emotion = str(dominant).title()
                    self.confidence = confidence

            except Exception:

                # Keep camera running even if
                # no face is detected
                with self.lock:
                    self.emotion = "Looking for a face..."
                    self.confidence = 0.0

        # ====================================================
        # DISPLAY RESULT
        # ====================================================

        with self.lock:
            label = (
                f"{self.emotion} "
                f"({self.confidence:.1f}%)"
            )

        cv2.putText(
            image,
            label,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (50, 220, 150),
            2,
            cv2.LINE_AA
        )

        return av.VideoFrame.from_ndarray(
            image,
            format="bgr24"
        )


# ============================================================
# CAMERA
# ============================================================

st.subheader("📷 Live Camera")

ctx = webrtc_streamer(
    key="emotion-detector",
    video_processor_factory=EmotionDetector,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=True
)


# ============================================================
# INFORMATION
# ============================================================

st.info(
    "💡 For better results, use good lighting, face the camera "
    "directly, and keep your face clearly visible."
)

st.caption(
    "Expression estimates are not definitive measures "
    "of a person's actual emotions."
)
