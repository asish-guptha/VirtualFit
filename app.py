import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Function to resize the image
def resize_image(image, width, height):
    return cv2.resize(image, (width, height))

# Load the image to overlay
uploaded_image = st.file_uploader("Upload image to overlay", type=["png", "jpg", "jpeg"])

# Streamlit widgets to control the overlay image size and position
st.sidebar.title("Control Panel")
width = st.sidebar.slider("Width of overlay", 50, 500, 200)
height = st.sidebar.slider("Height of overlay", 50, 500, 200)
x_pos = st.sidebar.slider("X position of overlay", 0, 640, 100)
y_pos = st.sidebar.slider("Y position of overlay", 0, 480, 100)

# Open the webcam
cap = cv2.VideoCapture(0)

if uploaded_image:
    overlay_img = Image.open(uploaded_image)
    
    # Convert the image to RGB (this will discard alpha if it's present)
    overlay_img = overlay_img.convert("RGB")
    overlay_img = np.array(overlay_img)

    # Streamlit image display
    stframe = st.empty()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Ensure the frame has the correct type and color format
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize the overlay image
        resized_overlay = resize_image(overlay_img, width, height)
        
        # Get the dimensions of the resized image
        h, w, _ = resized_overlay.shape
        
        # Ensure that the overlay image fits within the frame bounds
        y2, x2 = y_pos + h, x_pos + w
        if y2 <= frame.shape[0] and x2 <= frame.shape[1]:
            frame[y_pos:y2, x_pos:x2] = resized_overlay

        # Display the frame with the overlay
        stframe.image(frame, channels="RGB", use_container_width=True)

cap.release()
