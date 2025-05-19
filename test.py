import cv2
from ultralytics import YOLO


model = YOLO("yolov8n.pt")

classes_of_interest = [24, 25, 26, 28, 63, 67, 73, 77]  # backpack, umbrella, handbag, suitcase, laptop, phone, book, scissors

video_source = "rtsp://192.168.10.13:554/user=admin_password=tlJwpbo6_channel=1_stream=0.sdp?real_stream"
# video_source = 0  # Use for webcam
# video_source = "path/to/video.mp4"  # Use for local video file

cap = cv2.VideoCapture(video_source)

if not cap.isOpened():
    print(f"❌ Failed to open video source: {video_source}")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to read frame. Exiting...")
        break
    
    frame = cv2.resize(frame, (640, 480))

    results = model(frame, conf=0.4)

    # Draw bounding boxes only for objects in classes_of_interest
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            if class_id not in classes_of_interest:
                continue  # Skip if class is not in the target list

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]
            class_name = model.names[class_id]

            # Draw rectangle and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{class_name} {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Show video feed with filtered detections
    cv2.imshow("Filtered YOLOv8n Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
