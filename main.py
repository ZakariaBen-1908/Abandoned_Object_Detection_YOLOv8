import cv2
import numpy as np
import time
import logging
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8s.pt")

# Define classes of interest with their names
classes_of_interest = {
    24: "Backpack",
    25: "Umbrella",
    26: "Handbag",
    28: "Suitcase",
    67: "Cell Phone",
    73: "Laptop",
    77: "Bottle"
}

# Abandoned object tracking parameters
abandoned_objects = {}
abandoned_threshold = 20  # Time (seconds) before marking an object as abandoned
min_safe_distance = 20  # Pixels: Minimum distance between person and object to prevent abandonment

# Load first frame for background subtraction
firstframe = cv2.imread('captured_frame13.png')
firstframe_gray = cv2.cvtColor(firstframe, cv2.COLOR_BGR2GRAY)
firstframe_blur = cv2.GaussianBlur(firstframe_gray, (3, 3), 0)
firstframe_height, firstframe_width = firstframe_blur.shape

# Video Capture
file_path = 'rtsp://192.168.10.13:554/user=admin_password=tlJwpbo6_channel=1_stream=0.sdp?real_stream'
cap = cv2.VideoCapture(file_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (firstframe_width, firstframe_height))
    
    # Run YOLOv8 Tracking
    results = model.track(frame, persist=True, classes=list(classes_of_interest.keys()) + [0])  # Include 'person' class (ID 0)

    object_detections = []
    person_detections = []

    if results[0].boxes is not None:
        for box in results[0].boxes:
            track_id = int(box.id.item()) if box.id is not None else None
            class_id = int(box.cls.item())  # Get class ID
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # Object center

            if class_id == 0:  # Person detected
                person_detections.append((cx, cy))

            elif class_id in classes_of_interest:
                object_detections.append((track_id, class_id, cx, cy, x1, y1, x2, y2))

    # Check for abandoned objects
    for track_id, class_id, obj_cx, obj_cy, x1, y1, x2, y2 in object_detections:
        class_name = classes_of_interest.get(class_id, "Unknown")

        # Check if a person is nearby
        object_attended = any(np.hypot(obj_cx - px, obj_cy - py) < min_safe_distance for px, py in person_detections)

        if object_attended:
            if track_id in abandoned_objects:
                del abandoned_objects[track_id]  # Reset abandoned tracking if a person is close
        else:
            # Start abandonment timer
            if track_id not in abandoned_objects:
                abandoned_objects[track_id] = time.time()
            else:
                elapsed_time = time.time() - abandoned_objects[track_id]
                if elapsed_time >= abandoned_threshold:
                    cv2.putText(frame, f"Abandoned {class_name}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 0, 255), 2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    logging.info(f"Abandoned {class_name} detected at {x1},{y1} with ID {track_id}")

        # Draw tracking boxes with class names
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {track_id} - {class_name}", (x1, y1 - 25),
                    cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 0), 2)

    # Draw detected people
    for px, py in person_detections:
        cv2.circle(frame, (px, py), 10, (255, 0, 0), -1)  # Blue circle for people

    # Display the frame
    cv2.imshow('Detection', frame)
    if cv2.waitKey(15) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
