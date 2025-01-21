import cv2
import mediapipe as mp
import numpy as np
import math
from tkinter import filedialog, Tk
from PIL import Image
import io
from rembg import remove

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Function to calculate distance between two points
def calculate_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# Function to remove background from an image
def remove_background(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
    image_bytes = cv2.imencode('.png', image_rgb)[1].tobytes()
    bg_removed = remove(image_bytes)
    image_with_transparency = Image.open(io.BytesIO(bg_removed)).convert("RGBA")
    return cv2.cvtColor(np.array(image_with_transparency), cv2.COLOR_RGBA2BGRA)

# Function to detect clothing keypoints
def detect_clothing_keypoints(image):
    image_no_bg = remove_background(image)

    # Convert to grayscale and apply threshold
    gray = cv2.cvtColor(image_no_bg, cv2.COLOR_BGRA2GRAY)
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No contours found.")
        return None

    # Get the largest contour (assumed to be the clothing)
    largest_contour = max(contours, key=cv2.contourArea)

    # Find the bounding box
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Approximate the contour to reduce unnecessary points
    epsilon = 0.01 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    keypoints = [(point[0][0], point[0][1]) for point in approx]
    return {
        "bounding_box": (x, y, w, h),
        "keypoints": keypoints,
        "processed_image": image_no_bg
    }

# Function to select a local file using a file dialog
def select_local_file():
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg")])
    return file_path

# Ask user to provide clothing image
print("Press 'l' to select a local file for the clothing image.")
choice = input("Enter your choice (l): ").strip().lower()

if choice == 'l':
    file_path = select_local_file()
    if file_path:
        clothing_image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    else:
        print("No file selected. Exiting.")
        exit()
else:
    print("Invalid choice. Exiting.")
    exit()

# Detect clothing keypoints
clothing_info = detect_clothing_keypoints(clothing_image)
if not clothing_info:
    print("Failed to process clothing image. Exiting.")
    exit()

clothing_no_bg = clothing_info["processed_image"]
clothing_keypoints = clothing_info["keypoints"]

# Open webcam for human pose detection
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb_frame)

    if result.pose_landmarks:
        mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        landmarks = result.pose_landmarks.landmark
        width, height = frame.shape[1], frame.shape[0]

        # Get body landmarks
        body_keypoints = {
            "left_shoulder": (int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * width),
                               int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * height)),
            "right_shoulder": (int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * width),
                                int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * height)),
            "left_hip": (int(landmarks[mp_pose.PoseLandmark.LEFT_HIP].x * width),
                          int(landmarks[mp_pose.PoseLandmark.LEFT_HIP].y * height)),
            "right_hip": (int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x * width),
                           int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y * height))
        }

        # Debug: Draw body keypoints
        for key, point in body_keypoints.items():
            cv2.circle(frame, point, 5, (255, 0, 0), -1)
            cv2.putText(frame, key, point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Map clothing keypoints to body landmarks
        if len(clothing_keypoints) >= 4:
            clothing_top_left = clothing_keypoints[0]
            clothing_top_right = clothing_keypoints[1]
            clothing_bottom_left = clothing_keypoints[2]
            clothing_bottom_right = clothing_keypoints[3]

            body_top_left = body_keypoints["left_shoulder"]
            body_top_right = body_keypoints["right_shoulder"]
            body_bottom_left = body_keypoints["left_hip"]
            body_bottom_right = body_keypoints["right_hip"]

            # Compute transformation matrix
            src_points = np.array([clothing_top_left, clothing_top_right, clothing_bottom_left, clothing_bottom_right], dtype=np.float32)
            dst_points = np.array([body_top_left, body_top_right, body_bottom_left, body_bottom_right], dtype=np.float32)

            transformation_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

            # Warp the clothing image to align with the body
            warped_clothing = cv2.warpPerspective(clothing_no_bg, transformation_matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

            # Debug: Show warped clothing
            cv2.imshow("Warped Clothing", warped_clothing)

            # Overlay the transformed clothing
            for c in range(3):
                alpha = warped_clothing[:, :, 3] / 255.0
                frame[:, :, c] = (1 - alpha) * frame[:, :, c] + alpha * warped_clothing[:, :, c]

    cv2.imshow("Virtual Fitting Room", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
