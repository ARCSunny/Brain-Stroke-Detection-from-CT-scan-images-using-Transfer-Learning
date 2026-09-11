import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(
    page_title="Brain Stroke CT Detection",
    page_icon="🧠",
    layout="centered"
)

MODEL_PATH = "brain_stroke_model.keras"
IMG_SIZE = 224

# IMPORTANT:
# flow_from_dataframe() normally assigns class indices alphabetically.
# For the training notebook, this is:
# Bleeding -> 0, Ischemia -> 1, Normal -> 2
CLASS_NAMES = ["Bleeding", "Ischemia", "Normal"]


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None

    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )


model = load_model()

st.title("🧠 Brain Stroke CT Detection")
st.write(
    "Upload a brain CT image to obtain a prediction from "
    "the trained transfer-learning model."
)

st.info("Model classes: Bleeding, Ischemia, and Normal")

if model is None:
    st.error(
        f"Model file '{MODEL_PATH}' was not found. "
        "Place the .keras model in the same folder as app.py."
    )
    st.stop()

uploaded_file = st.file_uploader(
    "Upload a brain CT image",
    type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Uploaded CT Image")
    st.image(
        image,
        caption="Uploaded image",
        use_container_width=True
    )

    # Resize only. The saved model contains the preprocessing layer
    # used during training.
    resized_image = image.resize((IMG_SIZE, IMG_SIZE))

    image_array = np.asarray(
        resized_image,
        dtype=np.float32
    )

    image_array = np.expand_dims(image_array, axis=0)

    probabilities = model.predict(
        image_array,
        verbose=0
    )[0]

    predicted_index = int(np.argmax(probabilities))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(probabilities[predicted_index])

    st.subheader("Prediction Result")

    if predicted_class == "Normal":
        st.success(f"Prediction: {predicted_class}")
    elif predicted_class == "Ischemia":
        st.warning(f"Prediction: {predicted_class}")
    else:
        st.error(f"Prediction: {predicted_class}")

    st.metric(
        "Confidence",
        f"{confidence * 100:.2f}%"
    )

    st.subheader("Class Probabilities")

    probability_data = {
        CLASS_NAMES[i]: float(probabilities[i])
        for i in range(len(CLASS_NAMES))
    }

    st.bar_chart(probability_data)

    for i, class_name in enumerate(CLASS_NAMES):
        st.write(
            f"**{class_name}:** "
            f"{probabilities[i] * 100:.2f}%"
        )

st.divider()

st.caption(
    "⚠️ Academic project only. This application is not a "
    "clinically validated medical diagnostic system and should "
    "not be used to make medical decisions."
)
