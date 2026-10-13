
import streamlit as st
import cv2
import time
from deepface import DeepFace

st.set_page_config(page_title="AI Face Emotion Detector", page_icon="😊")

st.title("😊 AI Face Emotion Detector")
st.write("Allow webcam access to estimate facial expressions.")

run = st.checkbox("Start Webcam")

frame_placeholder = st.empty()
result_placeholder = st.empty()

if run:
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        st.error("Webcam not found. Check your camera permissions.")
    else:
        try:
            while run:
                success, frame = camera.read()

                if not success:
                    st.error("Could not read from webcam.")
                    break

                # Mirror the webcam view
                frame = cv2.flip(frame, 1)

                # Display the live webcam feed
                frame_placeholder.image(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    channels="RGB",
                    use_container_width=True
                )

                # Analyze expressions periodically
                try:
                    analysis = DeepFace.analyze(
                        frame,
                        actions=["emotion"],
                        enforce_detection=False,
                        detector_backend="opencv",
                        silent=True
                    )

                    emotion = analysis[0]["dominant_emotion"]
                    confidence = analysis[0]["emotion"][emotion]

                    result_placeholder.markdown(
                        f"### Expression estimate: **{emotion.title()}**\n\n"
                        f"Confidence: **{confidence:.1f}%**"
                    )

                except Exception:
                    result_placeholder.info(
                        "Looking for a face... Please face the camera."
                    )

                time.sleep(0.2)

        finally:
            camera.release()
else:
    st.info("Select **Start Webcam** to begin.")
