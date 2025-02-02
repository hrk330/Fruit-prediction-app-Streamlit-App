import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import streamlit as st

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)  # Ignore deprecation warnings

# Disable oneDNN (if desired for consistency across runs)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Load the trained model and label encoder
@st.cache_resource
def load_trained_model():
    model_path = "trained_cnn_model.h5"
    return load_model(model_path)

@st.cache_resource
def load_label_encoder():
    label_encoder_path = "label_encoder.pkl"
    return joblib.load(label_encoder_path)

# Load the trained model and label encoder
model = load_trained_model()
label_encoder = load_label_encoder()

# Define CSV file for storing incorrect predictions
feedback_file = "feedback_data.csv"

# Ensure CSV file exists
if not os.path.exists(feedback_file):
    df = pd.DataFrame(columns=["image_path", "incorrect_label", "correct_label"])
    df.to_csv(feedback_file, index=False)

# Function to preprocess the image
def preprocess_image(img):
    img = img.convert("RGB")  # Ensure image has 3 color channels
    img = img.resize((128, 128))  # Resize to match model input
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Function to predict the image class
def predict_image_class(img):
    img_array = preprocess_image(img)
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_label = label_encoder.classes_[predicted_class_index]
    return predicted_class_label

# Streamlit UI
st.title("Fruit Classification App 🍎🍌🍊")
st.write("Upload an image of a fruit, and the app will predict its class.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    predicted_label = predict_image_class(img)
    st.write(f"### **Predicted Class: {predicted_label}**")

    # Checkbox for incorrect prediction
    incorrect_prediction = st.checkbox("Is this prediction incorrect?")

    if incorrect_prediction:
        # Show dropdown with all possible fruit labels
        correct_label = st.selectbox(
            "Select the correct label:", options=label_encoder.classes_, index=0
        )

        if st.button("Submit Feedback"):
            if correct_label:
                # Save feedback
                df = pd.read_csv(feedback_file)
                new_feedback = pd.DataFrame({
                    "image_path": [uploaded_file.name], 
                    "incorrect_label": [predicted_label], 
                    "correct_label": [correct_label]
                })
                df = pd.concat([df, new_feedback], ignore_index=True)
                df.to_csv(feedback_file, index=False)
                
                st.success("✅ Feedback saved! The model will improve over time.")
            else:
                st.warning("⚠️ Please select the correct label before submitting.")
