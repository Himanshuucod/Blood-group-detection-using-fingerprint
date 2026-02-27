import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from PIL import Image

st.title("Fingerprint Blood Group Detection")

# Load model
model = load_model("model_blood_group_detection_lenet.keras")

labels = {'A+': 0, 'A-': 1, 'AB+': 2, 'AB-': 3, 'B+': 4, 'B-': 5, 'O+': 6, 'O-': 7}
labels = dict((v, k) for k, v in labels.items())

uploaded_file = st.file_uploader("Upload Fingerprint Image", type=["jpg", "png", "bmp"])

if uploaded_file is not None:
    
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img = img.resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    result = model.predict(x)
    predicted_class = np.argmax(result)

    prediction = labels[predicted_class]
    confidence = result[0][predicted_class] * 100

    st.success(f"Prediction: {prediction}")
    st.info(f"Confidence: {confidence:.2f}%")