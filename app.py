
import os

# ============================================================
# IMPORTANT: Configure TensorFlow BEFORE importing DeepFace
# ============================================================

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import time
import threading

import av
import cv2
import streamlit as st

from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase,
)

from deepface import DeepFace


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Face Emotion Detector",
    page_icon="😊",
    layout="centered",
)


# ============================================================
# TITLE
# ============================================================

st.title("😊 AI Face Emotion Detector")

st.write(
    "Allow camera access and the AI will estimate "
    "the facial expression visible on camera."
)


# ============================================================
# EMOTION DETECTOR
# ============================================================

class EmotionDetector(VideoProcessorBase):

    def __init__(self):

        # Current detected emotion
        self.emotion = "Starting camera..."

        # Confidence percentage
        self.confidence = 0.0

        # Thread lock prevents conflicts between
        # the video thread and DeepFace
        self.lock = threading.Lock()

        # Do NOT analyze immediately.
        # Give the camera time to start first.
        self.last_check = time.time()

        # Analyze every 2 seconds
        self.analysis_interval = 2.0

    def recv(self, frame):

        # ====================================================
        # CONVERT WEBRTC FRAME TO OPENCV
        # ====================================================

        image = frame.to_ndarray(format="bgr24")

        # Mirror camera like a normal webcam
        image = cv2.flip(image, 1)

        current_time = time.time()

        # ====================================================
        # RUN DEEPFACE PERIODICALLY
        # ====================================================

        if (
            current_time - self.last_check
            >= self.analysis_interval
        ):

            self.last_check = current_time

            try:

                result = DeepFace.analyze(
                    img_path=image,
                    actions=["emotion"],
                    detector_backend="opencv",
                    enforce_detection=False,
                    silent=True,
                )

                # DeepFace usually returns a list
                if isinstance(result, list):
                    result = result[0]

                # Get emotion information
                emotion_data = result.get(
                    "emotion",
                    {}
                )

                dominant_emotion = result.get(
                    "dominant_emotion",
                    "Unknown"
                )

                confidence = float(
                    emotion_data.get(
                        dominant_emotion,
                        0.0
                    )
                )

                # ====================================================
                # UPDATE RESULT SAFELY
                # ====================================================

                with self.lock:

                    self.emotion = (
                        str(dominant_emotion)
                        .title()
                    )

                    self.confidence = confidence

            except Exception:

                # If a face is not detected or DeepFace
                # temporarily fails, DO NOT kill the camera.

                with self.lock:

                    self.emotion = (
                        "Looking for a face..."
                    )

                    self.confidence = 0.0

        # ====================================================
        # GET CURRENT RESULT
        # ====================================================

        with self.lock:

            emotion_text = self.emotion
            confidence_text = self.confidence

        # ====================================================
        # DRAW RESULT ON CAMERA
        # ====================================================

        display_text = (
            f"{emotion_text} "
            f"({confidence_text:.1f}%)"
        )

        cv2.putText(
            image,
            display_text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (50, 220, 150),
            2,
            cv2.LINE_AA,
        )

        # ====================================================
        # RETURN FRAME
        # ====================================================

        return av.VideoFrame.from_ndarray(
            image,
            format="bgr24",
        )


# ============================================================
# LIVE CAMERA
# ============================================================

st.subheader("📷 Live Camera")

ctx = webrtc_streamer(

    # Unique key
    key="emotion-detector",

    # Our custom processor
    video_processor_factory=EmotionDetector,

    # Camera settings
    media_stream_constraints={
        "video": True,
        "audio": False,
    },

    # ========================================================
    # IMPORTANT FOR STREAMLIT CLOUD
    # ========================================================

    rtc_configuration={
        "iceServers": [
            {
                "urls": [
                    "stun:stun.l.google.com:19302"
                ]
            }
        ]
    },

    # Process frames asynchronously
    async_processing=True,
)


# ============================================================
# INSTRUCTIONS
# ============================================================

st.info(
    "💡 For better results, use good lighting, "
    "face the camera directly, and keep your face "
    "clearly visible."
)


# ============================================================
# EXPLANATION
# ============================================================

st.markdown(
    """
### 🧠 How it works

**Camera → Face Detection → Emotion Analysis → Result**

The webcam sends frames to the application.  
OpenCV processes the frames and DeepFace analyzes
the facial expression.

The detected expression is displayed directly
on the camera feed.
"""
)


# ============================================================
# DISCLAIMER
# ============================================================

st.caption(
    "⚠️ Expression estimates are not definitive "
    "measurements of a person's actual emotions."
)
