import os
import cv2
import numpy as np
from tracker import ObjectTracker
from ultralytics import YOLO
from datetime import datetime, timedelta

# Initialize Tracker and YOLO Model
tracker = ObjectTracker()
model = YOLO("yolov8m.pt")

# Classes to Track (including person class 0 for proximity check)
classes_of_interest = [24, 25, 26, 28, 63, 67, 73, 77]  # Objects
# classes_of_interest = ['backpack', 'umbrella', 'handbag', 'suitcase', 'laptop', 'cell phone', 'book']  # Objects
person_class_id = 0  # Person class

# Folder for saving images
output_folder = "abandoned_objects"
os.makedirs(output_folder, exist_ok=True)

# Track objects by position hash
tracked_objects = {}  # {position_hash: {first_detected_time, saved, last_coords}}

# Load Reference Background Frame
firstframe = cv2.imread('captured_frame13.png')
firstframe_gray = cv2.cvtColor(firstframe, cv2.COLOR_BGR2GRAY)
firstframe_blur = cv2.GaussianBlur(firstframe_gray, (3, 3), 0)
firstframe_height, firstframe_width = firstframe_blur.shape

cv2.imshow("First Frame", firstframe_blur)

cap = cv2.VideoCapture('rtsp://192.168.10.13:554/user=admin_password=tlJwpbo6_channel=1_stream=0.sdp?real_stream')

# Helper: Get rounded hash for coordinates (to reduce sensitivity to minor shifts)
def get_position_hash(x, y, w, h, tolerance=15):
    return (round(x / tolerance), round(y / tolerance), round(w / tolerance), round(h / tolerance))

# Helper: Check if a person is near the object
def person_nearby(obj_x, obj_y, obj_w, obj_h, persons, proximity_threshold=100):
    for (px, py, pw, ph) in persons:
        # Check for overlap or closeness using bounding box proximity
        if (
            (px + pw >= obj_x - proximity_threshold and px <= obj_x + obj_w + proximity_threshold) and
            (py + ph >= obj_y - proximity_threshold and py <= obj_y + obj_h + proximity_threshold)
        ):
            return True
    return False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame.shape[:2] != (firstframe_height, firstframe_width):
        frame = cv2.resize(frame, (firstframe_width, firstframe_height))

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_blur = cv2.GaussianBlur(frame_gray, (3, 3), 0)
    frame_diff = cv2.absdiff(firstframe_blur, frame_blur)

    edged = cv2.Canny(frame_diff, 5, 200)
    kernel = np.ones((10, 10), np.uint8)
    thresh = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Detect motion-based contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    motion_detections = []
    for c in contours:
        if 50 < cv2.contourArea(c) < 10000:
            x, y, w, h = cv2.boundingRect(c)
            motion_detections.append([x, y, w, h])

    # YOLOv8 object and person detection
    results = model(frame)

    filtered_detections = []
    class_ids = []
    persons = []

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1

            if class_id == person_class_id:
                persons.append((x1, y1, w, h))  # Track person locations

            if class_id in classes_of_interest:
                filtered_detections.append([x1, y1, w, h])
                class_ids.append(class_id)

    # Update tracker
    _, abandoned_objects = tracker.update(filtered_detections, class_ids)

    now = datetime.now()

    for obj in abandoned_objects:
        _, x, y, w, h, _, class_id = obj

        if class_id not in classes_of_interest:
            continue

        # Draw the bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, "Abandoned Object", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Position hash to track this object across frames
        pos_hash = get_position_hash(x, y, w, h)

        if pos_hash not in tracked_objects:
            tracked_objects[pos_hash] = {
                "first_detected_time": now,
                "saved": False,
                "last_coords": (x, y, w, h)
            }

        tracked_obj = tracked_objects[pos_hash]

        # Check if a person is nearby (suppress capture & alert if true)
        if person_nearby(x, y, w, h, persons):
            print(f"👥 Person near object at ({x}, {y}) - No alert triggered")
            continue

        # Save image once
        if not tracked_obj["saved"]:
            obj_image = frame[y:y + h, x:x + w]
            timestamp = now.strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(output_folder, f"abandoned_{timestamp}_x{x}_y{y}.png")
            cv2.imwrite(filename, obj_image)
            print(f"✅ Saved abandoned object at ({x}, {y}) to {filename}")
            tracked_obj["saved"] = True

        # Trigger alert if object stays > 20 seconds (and no nearby person)
        if now - tracked_obj["first_detected_time"] > timedelta(seconds=20):
            print(f"⚠️ ALERT: Object at ({x}, {y}) has stayed for more than 20 seconds with no nearby person!")

        tracked_obj["last_coords"] = (x, y, w, h)

    # Show output
    cv2.imshow('Abandoned Object Detection', frame)

    if cv2.waitKey(15) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
