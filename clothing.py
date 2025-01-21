import cv2
import numpy as np
from rembg import remove
from PIL import Image
import io

def remove_background(image):
    """Remove background using rembg."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)  # Convert to RGBA
    image_bytes = cv2.imencode('.png', image_rgb)[1].tobytes()
    bg_removed = remove(image_bytes)  # Apply background removal
    image_with_transparency = Image.open(io.BytesIO(bg_removed)).convert("RGBA")
    return cv2.cvtColor(np.array(image_with_transparency), cv2.COLOR_RGBA2BGRA)

def detect_clothing_keypoints(image_path):
    """Detect keypoints of the clothing (e.g., shoulders, edges)."""
    # Load and preprocess the image
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        print("Image not found.")
        return None

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

    # Extract keypoints (corners, edges, etc.)
    keypoints = [(point[0][0], point[0][1]) for point in approx]

    # Highlight keypoints on the image
    result_image = image_no_bg.copy()
    for point in keypoints:
        cv2.circle(result_image, point, 5, (0, 255, 0, 255), -1)

    # Draw bounding box for visualization
    cv2.rectangle(result_image, (x, y), (x + w, y + h), (255, 0, 0, 255), 2)

    # Show the result
    cv2.imshow("Clothing Detection", cv2.cvtColor(result_image, cv2.COLOR_BGRA2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return {
        "bounding_box": (x, y, w, h),
        "keypoints": keypoints
    }

# Test the function
image_path = "tshirtsbgr/2.png"
clothing_info = detect_clothing_keypoints(image_path)
if clothing_info:
    print("Bounding Box:", clothing_info["bounding_box"])
    print("Keypoints:", clothing_info["keypoints"])
