import cv2
from ultralytics import YOLO

# Load the YOLOv8 model (you can try yolov8s.pt or yolov8m.pt too)
model = YOLO("yolov8n.pt")  # 'n' is the nano version (fastest)

# Open webcam (0 = default laptop webcam)
cap = cv2.VideoCapture(0)

# Check if webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Run object detection
    results = model.predict(source=frame, conf=0.5, verbose=False)

    # Plot detection results on frame
    annotated_frame = results[0].plot()
    # Show the frame
    cv2.imshow("YOLOv8 Webcam", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows(1)
