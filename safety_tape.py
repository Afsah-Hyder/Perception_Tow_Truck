import cv2
import numpy as np

# Load the video
video_path = "/home/arfah/Desktop/FYP/Framework/WhatsApp Video 2025-02-07 at 13.24.45.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define output video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output_segmented2.mp4", fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Crop the lower half of the frame (focus only on the road)
    frame_cropped = frame[height // 2 :, :]

    # Convert to LAB & HSV
    lab = cv2.cvtColor(frame_cropped, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(frame_cropped, cv2.COLOR_BGR2HSV)

    # **1. Remove Grey using Saturation & Lightness**
    # Grey has low saturation (S < 50) and medium brightness
    lower_grey = np.array([0, 0, 50])   # Very low saturation (S), medium brightness (V)
    upper_grey = np.array([180, 50, 200]) # Excludes bright whites
    mask_grey = cv2.inRange(hsv, lower_grey, upper_grey)

    # **2. Detect White Tape in LAB (avoiding grey)**
    lower_white = np.array([220, 120, 120])  # High brightness, avoids grey
    upper_white = np.array([255, 140, 140])
    mask_white = cv2.inRange(lab, lower_white, upper_white)

    # **3. Detect Yellow Tape in HSV**
    lower_yellow = np.array([15, 80, 80])  
    upper_yellow = np.array([40, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # **4. Remove Grey from White Mask**
    mask_white = cv2.bitwise_and(mask_white, cv2.bitwise_not(mask_grey))

    # **5. Combine Final Mask (Yellow + White - Grey)**
    combined_mask = cv2.bitwise_or(mask_white, mask_yellow)
    # **6. Apply Morphology (Reduce Noise)**

    kernel = np.ones((5, 5), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

    # **7. Apply Edge Detection**
    edges = cv2.Canny(combined_mask, 100, 150)

    # **8. Detect Lines Using Hough Transform**
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=20, minLineLength=50, maxLineGap=50)
 
    # **9. Draw Detected Tape Lines**
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame_cropped, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Overlay processed lower half back onto the original frame
    frame[height // 2 :, :] = frame_cropped

    # Show the output
    cv2.imshow("Tape Detection", frame)

    # Save processed frame
    out.write(frame)

    # Press 'q' to exit early
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
