                    self.emotion = str(dominant).title()
                    self.confidence = confidence

            except Exception:
                # Keep the app running if a frame has no detectable face.
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
            0.75,
            (50, 220, 150),
            2,
            cv2.LINE_AA
        )

        return av.VideoFrame.from_ndarray(image, format="bgr24")


st.subheader("Live camera")
ctx = webrtc_streamer(
    key="emotion-detector",
    video_processor_factory=EmotionDetector,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)

st.caption(
    "Tip: Use good lighting, face the camera, and keep your face clearly visible. "
    "The first analysis may take longer while the model initializes."
)
