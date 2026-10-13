
import streamlit as st
import numpy as np
import cv2
from PIL import Image
from fer import FER

# =====================================================
# AI FACE EMOTION DETECTOR
# School AI Club Exhibition Project
# =====================================================

st.set_page_config(
    page_title="AI Face Emotion Detector",
    page_icon="😊",
    layout="wide"
)

# ------------------- DESIGN -------------------

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #081323, #132b43);
    color: #f1f5f9;
}
h1, h2, h3 {
    color: #48e5bd !important;
}
[data-testid="stMetric"] {
    background: #172d44;
    border: 1px solid #28617a;
    border-radius: 14px;
    padding: 15px;
}
</style>
""", unsafe_allow_html=True)

# ------------------- EMOTION DATA -------------------

EMOTION_INFO = {
    "happy": ("😊", "Happy", "A smile-like expression was detected."),
    "sad": ("😢", "Sad", "The expression resembles sadness."),
    "angry": ("😠", "Angry", "The expression resembles anger."),
    "surprise": ("😲", "Surprised", "An expression of surprise was detected."),
    "fear": ("😨", "Fearful", "The expression resembles fear."),
    "disgust": ("🤢", "Disgust", "The expression resembles disgust."),
    "neutral": ("😐", "Neutral", "A neutral facial expression was detected.")
}

# Load the emotion model once per app session.
@st.cache_resource
def load_detector():
    return FER(mtcnn=False)

# ------------------- HEADER -------------------

st.markdown("### COMPUTER VISION LAB // PROJECT 001")

st.title("😊 AI FACE EMOTION DETECTOR")
st.subheader("Your face. Your expressions. Artificial intelligence.")

st.write(
    "Capture a picture using your computer's built-in camera. "
    "The AI will analyse facial expressions and display its predictions."
)

st.info(
    "🔒 Camera access requires your permission. "
    "This demo processes the captured image in the app; "
    "it does not intentionally save the image."
)

# ------------------- CAMERA -------------------

st.markdown("---")
st.markdown("## 📸 CAMERA SCANNER")

st.write(
    "Allow camera access when your browser asks. "
    "On an HP laptop, select its built-in camera if prompted."
)

photo = st.camera_input(
    "Take a picture with your computer camera",
    key="emotion_camera"
)

# Optional alternative if the camera is unavailable.
with st.expander("Camera not working? Upload a picture instead"):
    uploaded_photo = st.file_uploader(
        "Upload a face image",
        type=["jpg", "jpeg", "png"]
    )

image_source = photo if photo is not None else uploaded_photo

# ------------------- ANALYSIS -------------------

if image_source is not None:
    st.markdown("---")
    st.markdown("## 🤖 AI ANALYSIS")

    try:
        image = Image.open(image_source).convert("RGB")
        image_array = np.array(image)

        left, right = st.columns(2)

        with left:
            st.image(
                image,
                caption="Captured image",
                use_container_width=True
            )

        with st.spinner(
            "AI is detecting the face and analysing expressions..."
        ):
            detector = load_detector()

            # FER expects an image in OpenCV's BGR format.
            image_bgr = cv2.cvtColor(
                image_array,
                cv2.COLOR_RGB2BGR
            )

            results = detector.detect_emotions(image_bgr)

        with right:
            if not results:
                st.warning(
                    "No face detected. Please try again with your "
                    "face clearly visible, good lighting, and the "
                    "camera positioned at eye level."
                )
            else:
                # Use the face with the largest detected area.
                face = max(
                    results,
                    key=lambda result:
                    result["box"][2] * result["box"][3]
                )

                emotions = face["emotions"]

                dominant_emotion = max(
                    emotions,
                    key=emotions.get
                )

                confidence = emotions[dominant_emotion] * 100

                emoji, label, description = EMOTION_INFO.get(
                    dominant_emotion,
                    ("🧠", dominant_emotion.title(),
                     "Expression analysis completed.")
                )

                st.markdown("### DETECTION RESULT")

                st.markdown(
                    f"<div style='background:#172d44;"
                    f"padding:24px;border-radius:18px;"
                    f"border:1px solid #48e5bd;text-align:center;'>"
                    f"<div style='font-size:64px'>{emoji}</div>"
                    f"<h2>{label.upper()}</h2>"
                    f"<p>{description}</p>"
                    f"</div>",
                    unsafe_allow_html=True
                )

                st.metric(
                    "Model score for leading expression",
                    f"{confidence:.1f}%"
                )

                st.caption(
                    "This percentage is the model's score, "
                    "not the probability that the person truly "
                    "feels this emotion."
                )

                # Draw a box around the detected face.
                x, y, w, h = face["box"]

                boxed_image = image_array.copy()

                cv2.rectangle(
                    boxed_image,
                    (max(0, x), max(0, y)),
                    (
                        min(boxed_image.shape[1] - 1, x + w),
                        min(boxed_image.shape[0] - 1, y + h)
                    ),
                    (30, 220, 160),
                    3
                )

                st.image(
                    boxed_image,
                    caption="Face detection",
                    use_container_width=True
                )

                st.markdown("### 📊 EXPRESSION SCORES")

                # Show every emotion score.
                sorted_emotions = sorted(
                    emotions.items(),
                    key=lambda item: item[1],
                    reverse=True
                )

                for emotion, score in sorted_emotions:
                    emoji, name, _ = EMOTION_INFO.get(
                        emotion,
                        ("🧠", emotion.title(), "")
                    )

                    st.write(
                        f"{emoji} **{name}** — {score * 100:.1f}%"
                    )

                    st.progress(
                        float(max(0, min(1, score)))
                    )

                st.success("Analysis complete!")

    except Exception as error:
        st.error(
            "The image could not be analysed. Check the "
            "installation logs if this problem continues."
        )
        st.caption(f"Technical detail: {type(error).__name__}")

else:
    st.markdown("---")
    st.markdown("### 🟢 SYSTEM READY")
    st.write(
        "Your scanner is waiting. Capture a picture to begin."
    )

# ------------------- FOOTER -------------------

st.markdown("---")
st.caption(
    "AI FACE EMOTION DETECTOR | PYTHON + STREAMLIT + FER"
)
st.caption(
    "Educational demonstration only. Facial-expression "
    "classifications are estimates, not reliable measurements "
    "of a person's internal emotional state."
)
