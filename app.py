
import streamlit as st
import av
import cv2
import threading
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from deepface import DeepFace

st.set_page_config(
    page_title="AI Face Emotion Detector",
    page_icon="😊"
)

st.title("😊 AI Face Emotion Detector")
st.write("Allow camera access to estimate your facial expression.")

class EmotionDetector(VideoProcessorBase):
    def __init__(self):
        self.emotion = "Waiting for face..."
        self.confidence = 0.0
        self.lock = threading.Lock()
        self.last_check = 0

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")

        # Mirror the webcam image
        image = cv2.flip(image, 1)

        # Analyze approximately every 2 seconds
        import time
        now = time.time()

        if now - self.last_check >= 2:
            self.last_check = now
            try:
                result = DeepFace.analyze(
                    image,
                    actions=["emotion"],
                    enforce_detection=False,
                    detector_backend="opencv",
                    silent=True
                )

                emotion_data = result[0]["emotion"]
                dominant = result[0]["dominant_emotion"]
                confidence = emotion_data[dominant]

                with self.lock:
                    self.emotion = dominant.title()
                    self.confidence = float(confidence)

            except Exception:
                with self.lock:
                    self.emotion = "Looking for a face..."
                    self.confidence = 0.0

        with self.lock:
            label = f"{self.emotion} ({self.confidence:.1f}%)"

        cv2.putText(
            image,
            label,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (50, 220, 150),
            2
        )

        return av.VideoFrame.from_ndarray(image, format="bgr24")


ctx = webrtc_streamer(
    key="emotion-detector",
    video_processor_factory=EmotionDetector,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=True
)

st.caption(
    "Expression estimates are not definitive measures of a person's actual emotions."
)
