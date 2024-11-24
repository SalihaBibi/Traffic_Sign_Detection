import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image

# Define the transformation for preprocessing
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# Streamlit app
st.title("Traffic Sign Detection")
st.write("Upload an image to classify the traffic sign!")

# Load the entire model (architecture + weights)
model = torch.load("traffic_sign_model.pth", map_location=torch.device("cpu"))
model.eval()  # Set the model to evaluation mode
st.write("Model successfully loaded!")


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

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Open and display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    # Preprocess the image
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Make prediction
    try:
        with torch.no_grad():
            outputs = model(image_tensor)
            predicted_class_index = torch.argmax(outputs, 1).item()  # Get predicted class index
            predicted_class_label = class_labels[predicted_class_index]  # Map index to label
        st.write(f"Predicted Traffic Sign: {predicted_class_label}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")