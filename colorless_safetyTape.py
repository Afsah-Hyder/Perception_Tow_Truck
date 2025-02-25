import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov5n.pt")

# Open live camera feed (0 = default webcam, 6 = external camera)
video_path = "//home/arfah/Desktop/FYP/Framework/WhatsApp Video 2025-02-07 at 13.24.45.mp4"
cap = cv2.VideoCapture(video_path)
# cap = cv2.VideoCapture(6)

# Set camera resolution (optional, adjust as needed)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Exit if no frame is captured

    height, width, _ = frame.shape  # Get frame dimensions

    # --- Step 1: Process lower half for line detection ---
    frame_cropped = frame[height // 2 :, :]  # Crop the lower half
    gray = cv2.cvtColor(frame_cropped, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    edges = cv2.Canny(gray, 75, 150)  # Edge detection

    # Detect lines using Hough Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=20, maxLineGap=2)

    # Draw only vertical lines
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi  # Convert to degrees
            if 40 < abs(angle) < 135:  # Keep near-vertical lines
                cv2.line(frame_cropped, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Draw red lines

    # Overlay processed lower half back onto the full frame
    frame[height // 2 :, :] = frame_cropped

    # --- Step 2: Run YOLOv8 object detection ---
    results = model(frame)

    for result in results:
        for box in result.boxes.data:
            x1, y1, x2, y2, conf, cls = map(int, box[:6])  # Get bounding box values
            label = f"{model.names[cls]} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw green bounding boxes
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the final processed frame with both line detection and YOLO
    cv2.imshow("YOLO + Line Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
