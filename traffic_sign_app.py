import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import tensorflow as tf

# Load the pre-trained model
model = load_model('/content/traffic_sign_model.h5')  # Update the path to where the model is saved

# Define a function to preprocess the image
def preprocess_image(image):
    image = image.resize((64, 64))  # Resize to the input size of the model
    image = np.array(image) / 255.0  # Normalize the image (scaling to [0, 1])
    
    if len(image.shape) == 2:  # If image is grayscale, convert to RGB
        image = np.stack([image] * 3, axis=-1)  # Convert single channel to 3 channels
    
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Define class labels
class_labels = {
    0: "Speed Limit 20", 1: "Speed Limit 30", 2: "Speed Limit 50", 3: "Speed Limit 60",
    4: "Speed Limit 70", 5: "Speed Limit 80", 6: "End of Speed Limit 80", 7: "Speed Limit 100",
    8: "Speed Limit 120", 9: "No Passing", 10: "No Passing for Vehicles > 3.5T", 11: "Right of Way",
    12: "Priority Road", 13: "Yield", 14: "Stop", 15: "No Vehicles", 16: "Vehicles > 3.5T Prohibited",
    17: "No Entry", 18: "General Caution", 19: "Dangerous Curve Left", 20: "Dangerous Curve Right",
    21: "Double Curve", 22: "Bumpy Road", 23: "Slippery Road", 24: "Road Narrows on Right",
    25: "Road Work", 26: "Traffic Signals", 27: "Pedestrians", 28: "Children Crossing",
    29: "Bicycles Crossing", 30: "Beware of Ice/Snow", 31: "Wild Animals Crossing",
    32: "End of All Restrictions", 33: "Turn Right Ahead", 34: "Turn Left Ahead",
    35: "Ahead Only", 36: "Go Straight or Right", 37: "Go Straight or Left",
    38: "Keep Right", 39: "Keep Left", 40: "Roundabout Mandatory", 41: "End of No Passing",
    42: "End of No Passing for Vehicles > 3.5T"
}

# Streamlit interface
st.title("Traffic Sign Detection")
st.write("Upload an image of a traffic sign to detect it.")

# Upload image
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")

    # Preprocess the image
    processed_image = preprocess_image(image)

    # Make predictions
    predictions = model.predict(processed_image)
    class_idx = np.argmax(predictions, axis=1)

    # Display the prediction
    st.write(f"Predicted class: {class_labels[class_idx[0]]}")