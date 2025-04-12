import cv2
import mediapipe as mp
import numpy as np
import math

# Init MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Angle Calculation
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180.0:
        angle = 360 - angle
    return int(angle)

# Camera Setup
cap1 = cv2.VideoCapture("rtsp://admin:admin@123@192.168.0.14:554/1/2?transmode=unicast&profile=vam")
cap2 = cv2.VideoCapture("rtsp://admin:admin@123@192.168.0.11:554/1/2?transmode=unicast&profile=va")

# Set buffer size to reduce latency
cap1.set(cv2.CAP_PROP_BUFFERSIZE, 2)
cap2.set(cv2.CAP_PROP_BUFFERSIZE, 2)

frame_counter = 0  # Counter for skipping frames

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap1.isOpened() and cap2.isOpened():
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            print("Couldn't read from one of the cameras or corrupted frame detected")
            continue  # Skip to the next iteration instead of breaking

        frame_counter += 1

        # Only process every alternate frame
        if frame_counter % 2 == 0:
            def process_frame(frame):
                try:
                    frame = cv2.resize(frame, (640, 480))

                    crop_width = 300  # You can change this to desired cropped width
                    frame_height, frame_width, _ = frame.shape
                    start_x = (frame_width - crop_width) // 2
                    end_x = start_x + crop_width
                    frame = frame[:, start_x:end_x]

                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = pose.process(image)
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                    if results.pose_landmarks:
                        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                        landmarks = results.pose_landmarks.landmark

                        # Example: Angle at right elbow
                        shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * frame.shape[1],
                                    landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * frame.shape[0]]
                        elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * frame.shape[1],
                                 landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * frame.shape[0]]
                        wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * frame.shape[1],
                                 landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * frame.shape[0]]

                        angle = calculate_angle(shoulder, elbow, wrist)

                        cv2.putText(image, str(angle),
                                    tuple(np.multiply(elbow, [1, 1]).astype(int)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                    return image
                except Exception as e:
                    print(f"Error processing frame: {e}")
                    return np.zeros((480, 640, 3), dtype=np.uint8)  # Return a blank frame on error

            out1 = process_frame(frame1)
            out2 = process_frame(frame2)

            combined = cv2.hconcat([out1, out2])
            cv2.imshow("Camera 1 and Camera 2 - Pose Detection", combined)

        # Even on skipped frames, check for quit key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap1.release()
cap2.release()
cv2.destroyAllWindows()
