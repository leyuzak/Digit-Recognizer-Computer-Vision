import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from io import BytesIO

st.set_page_config(page_title="MNIST", page_icon="🧠")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mnist_cnn.h5", compile=False)

model = load_model()

st.title("🧠 MNIST Digit Recognizer")

file = st.file_uploader("Upload PNG/JPG", type=["png","jpg","jpeg"])

def preprocess(img):
    img = img.convert("L").resize((28,28))
    arr = np.array(img).astype("float32")
    if arr.mean() > 127:
        img = ImageOps.invert(img)
        arr = np.array(img).astype("float32")
    arr = (arr/255.0).reshape(1,28,28,1)
    return arr, img

if file:
    img = Image.open(BytesIO(file.getvalue()))
    x, pimg = preprocess(img)
    st.image(pimg, width=180)
    probs = model.predict(x, verbose=0)[0]
    pred = int(np.argmax(probs))
    st.subheader(f"Prediction: {pred}")
    st.bar_chart(probs)
