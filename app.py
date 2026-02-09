import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers

@st.cache_resource
def load_flower_model():
    base_model = keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights=None
    )

    model = keras.Sequential([
        layers.Input(shape=(224, 224, 3)),
        layers.Rescaling(1./255),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2),
        layers.Dense(128, activation='relu'),
        layers.Dense(4, activation='softmax')
    ])

    model.load_weights("flower_model.h5", by_name=True, skip_mismatch=True)
    return model


st.title("🌼 Flower Classification App")

model = load_flower_model()

class_names = ["Daisy", "Dandelion", "Rose", "Sunflower"]

uploaded_file = st.file_uploader("Upload flower image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, use_column_width=True)

    img = image.resize((224, 224))
    img_array = np.expand_dims(np.array(img), axis=0)

    preds = model.predict(img_array)
    st.success(f"🌸 Flower: {class_names[np.argmax(preds)]}")
    st.info(f"Confidence: {np.max(preds)*100:.2f}%")
