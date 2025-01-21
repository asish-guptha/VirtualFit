import cv2
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)

print("Press 'q' to quit.")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB (required by MediaPipe)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb_frame)

    # If pose landmarks are detected
    if result.pose_landmarks:
        # Draw the landmarks on the frame
        mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Get landmark coordinates
        landmarks = result.pose_landmarks.landmark
        height, width, _ = frame.shape

        print("\nDetected Body Coordinates:")
        for idx, landmark in enumerate(landmarks):
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            z = landmark.z
            print(f"Landmark {idx}: (x: {x}, y: {y}, z: {z})")

            # Draw circles on the detected landmarks
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    # Show the frame
    cv2.imshow("Body Detection", frame)

    # Break loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
