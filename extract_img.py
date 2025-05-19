import cv2

# Open the camera (local webcam: 0, or replace with IP camera URL)
Cam = 'rtsp://192.168.10.13:554/user=admin_password=tlJwpbo6_channel=1_stream=0.sdp?real_stream'
cap = cv2.VideoCapture(Cam)

if not cap.isOpened():
    print("Error: Unable to open camera.")
    exit()

print("Press 'c' to capture an image, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    frame = cv2.resize(frame, (640, 480))

    # Show the live camera feed
    cv2.imshow('Camera Feed', frame)

    # Wait for key press
    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):  # Press 'c' to capture and save an image
        cv2.imwrite('img.png', frame)
        print("Image captured and saved")

    elif key == ord('q'):  # Press 'q' to quit
        print("Exiting...")
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()

