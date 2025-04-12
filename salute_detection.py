import cv2
import mediapipe as mp
import numpy as np
import sqlite3
import time
import threading
import queue

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# ✅ Database Connection
conn = sqlite3.connect("salute_results.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
    CREATE TABLE IF NOT EXISTS results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        angle REAL,
        palm_angle REAL,
        status TEXT,
        suggestion TEXT,
        screenshot_path TEXT
    )
""")
conn.commit()

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360.0 - angle
        
    return angle 

# ✅ Initialize Camera
cap = cv2.VideoCapture("rtsp://admin:admin@123@192.168.0.14:554/1/1?transportmode=unicast&profile=vam")  # ✅ IP Camera URL
# cap = cv2.VideoCapture(0)  # ✅ Webcam
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  

frame_queue = queue.Queue()  # ✅ Queue for multi-threading
last_store_time = 0  
frame_skip = 2  # ✅ Process every 2nd frame
frame_count = 0  

# ✅ Multi-threading for reading frames
def read_frames():
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_queue.put(frame)

frame_thread = threading.Thread(target=read_frames, daemon=True)
frame_thread.start()

# ✅ Mediapipe Pose Model
with mp_pose.Pose(min_detection_confidence=0.3, min_tracking_confidence=0.3) as pose:
    while cap.isOpened():
        if frame_queue.empty():
            continue

        frame = frame_queue.get()
        frame = cv2.resize(frame, (840, 600))  # ✅ Resize frame for performance

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue  # ✅ Skip alternate frames to improve performance
        
        current_time = time.time()
        
        # ✅ Center Calculation
        h, w, _ = frame.shape
        cx, cy = w // 2, h // 2
        roi_width, roi_height = 250, 600
        x1 = max(0, cx - roi_width // 2)
        y1 = max(0, cy - roi_height // 2)
        x2 = min(w, cx + roi_width // 2)
        y2 = min(h, cy + roi_height // 2)

        # ✅ Extract Centered ROI
        centered_image = frame[y1:y2, x1:x2]
        image = frame[y1:y2, x1:x2]

        # ✅ Convert ROI to RGB for MediaPipe
        roi_rgb = cv2.cvtColor(centered_image, cv2.COLOR_BGR2RGB)
        roi_rgb.flags.writeable = False
        results = pose.process(roi_rgb)
        roi_rgb.flags.writeable = True
        centered_image = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2BGR)

        try:
            if results.pose_landmarks: 
                landmarks = results.pose_landmarks.landmark
                
                # Get coordinates
                shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                right_index = [landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].y]
            
                # Calculate angle
                elbow_angle = calculate_angle(shoulder, elbow, wrist)
                arm_angle = calculate_angle(hip, shoulder, elbow)
                wrist_angle = calculate_angle(elbow, wrist, right_index)
                arm_straight_angle = calculate_angle(left_shoulder, shoulder, elbow)

                angle = "Elbow Angle: " + str(int(elbow_angle)) + " Arm Angle: " + str(int(arm_angle)) + " Wrist Angle: " + str(int(wrist_angle))
                status = "Salute is wrong"

                # Determine suggestion based on arm_angle
                if 85 <= arm_angle <= 120 and 20 <= elbow_angle <= 30 and 168 <= wrist_angle <= 180:
                    suggestion = "Perfect position"
                    status = "Salute is Correct"
                elif arm_angle < 85:
                    suggestion = "Raise arm slowly [" + str(int(arm_angle)) + "]"
                elif arm_angle > 120:
                    suggestion = "Lower arm slowly [" + str(int(arm_angle)) + "]"
                elif elbow_angle < 20:
                    suggestion = "Raise elbow slightly [" + str(int(elbow_angle)) + "]"
                elif elbow_angle > 30:
                    suggestion = "Lower elbow slightly [" + str(int(elbow_angle)) + "]"
                elif wrist_angle < 168:
                    suggestion = "Raise wrist slightly [" + str(int(wrist_angle)) + "]"
                elif wrist_angle > 180:
                    suggestion = "Lower wrist slightly [" + str(int(wrist_angle)) + "]"

                # Display suggestion on the screen
                color = (0, 255, 0) if "Correct" in status else (0, 0, 255)
                cv2.putText(centered_image, f'Suggestion: {suggestion}', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                # Visualize angles
                cv2.putText(centered_image, str(int(elbow_angle)), 
                            tuple(np.multiply(elbow, [roi_width, roi_height]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(centered_image, str(int(arm_angle)),
                            tuple(np.multiply(shoulder, [roi_width, roi_height]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(centered_image, str(int(wrist_angle)),
                            tuple(np.multiply(wrist, [roi_width, roi_height]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(centered_image, str(int(arm_straight_angle)),
                            tuple(np.multiply(left_shoulder, [roi_width, roi_height]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2, cv2.LINE_AA)

                # ✅ Save Screenshot & Data (Every 1 second)
                if int(current_time) - int(last_store_time) >= 1 and 80 <= arm_angle <= 150 and 15 <= elbow_angle <= 40:
                    last_store_time = current_time

                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    centered_image_path = f"static/screenshots/centered_salute_{int(time.time())}.jpg"

                    # Save screenshot without suggestion
                    screenshot_without_text = centered_image.copy()
                    cv2.imwrite(centered_image_path, image)

                    cursor.execute("INSERT INTO results (timestamp, angle, status, suggestion, screenshot_path) VALUES (?, ?, ?, ?, ?)",
                                (timestamp, angle, status, suggestion, centered_image_path))
                    conn.commit()

        except Exception as e:
            print(e)

        # ✅ Display centered image
        cv2.imshow('Salute', centered_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # ✅ Reduced delay for smoother video
            break

cap.release()
cv2.destroyAllWindows()
conn.close()
