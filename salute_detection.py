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
cap = cv2.VideoCapture("rtsp://admin:Admin@123@192.168.0.14:554/1/2?transportmode=unicast&profile=va")  # ✅ IP Camera URL
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
        frame = cv2.resize(frame, (640, 480))  # ✅ Resize frame for performance

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue  # ✅ Skip alternate frames to improve performance
        
        current_time = time.time()
        
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
      
        # Make detection
        results = pose.process(image)
    
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
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
                arm_straight_angle = calculate_angle(left_shoulder,shoulder,elbow)

                angle = "Elbow Angle: " + str(int(elbow_angle)) + " Arm Angle: " + str(int(arm_angle)) + " Wrist Angle: " + str(int(wrist_angle))
                status = "Salute is wrong"

                # Determine suggestion based on arm_angle
                if 85 <= arm_angle <= 125 and 47 <= elbow_angle <= 75 and 135 <= wrist_angle <= 170:
                    suggestion = "Perfect position"
                    status = "Salute is Correct"
                elif arm_angle < 85:
                    suggestion = "Raise arm slowly [" + str(int(arm_angle)) + "]"
                elif arm_angle > 125:
                    suggestion = "Lower arm slowly [" + str(int(arm_angle)) + "]"
                elif elbow_angle < 47:
                    suggestion = "Raise elbow slightly [" + str(int(elbow_angle)) + "]"
                elif elbow_angle > 75:
                    suggestion = "Lower elbow slightly [" + str(int(elbow_angle)) + "]"
                elif wrist_angle < 135:
                    suggestion = "Raise wrist slightly [" + str(int(wrist_angle)) + "]"
                elif wrist_angle > 170:
                    suggestion = "Lower wrist slightly [" + str(int(wrist_angle)) + "]"

                # Display suggestion on the screen
                color = (0, 255, 0) if "Correct" in status else (0, 0, 255)
                cv2.putText(image, f'Suggestion: {suggestion}', (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                
                # # Visualize angle
                cv2.putText(image, str(int(elbow_angle)), 
                            tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA
                                    )
                cv2.putText(image, str(int(arm_angle)),
                            tuple(np.multiply(shoulder, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA
                                    )
                cv2.putText(image, str(int(wrist_angle)),
                            tuple(np.multiply(wrist, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA
                                    )
                cv2.putText(image, str(int(arm_straight_angle)),
                            tuple(np.multiply(left_shoulder, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA
                                    )

                # ✅ Save Screenshot & Data (Every 1 second)
                if int(current_time) - int(last_store_time) >= 1 and 60 <= arm_angle <= 150 and 30 <= elbow_angle <= 80:
                    last_store_time = current_time

                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    screenshot_path = f"static/screenshots/salute_{int(time.time())}.jpg"
                    cv2.imwrite(screenshot_path, frame)

                    cursor.execute("INSERT INTO results (timestamp, angle, status, suggestion, screenshot_path) VALUES (?, ?, ?, ?, ?)",
                                (timestamp, angle, status, suggestion, screenshot_path))
                    conn.commit()        

        except Exception as e:
            print(e)
        # # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )
        # Display on screen
        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # ✅ Reduced delay for smoother video
            break

cap.release()
cv2.destroyAllWindows()
conn.close()
