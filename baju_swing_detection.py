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
    CREATE TABLE IF NOT EXISTS baju_swing_result (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        angle REAL,
        status TEXT,
        suggestion TEXT,
        screenshot_path TEXT
    )
""")
conn.commit()

def calculate_angle(a, b, c):
    """Calculate angle between three points"""
    a = np.array(a)  # First point (shoulder)
    b = np.array(b)  # Middle point (elbow)
    c = np.array(c)  # End point (wrist)
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(cosine_angle))
    
    return angle


# ✅ Initialize Camera
cap = cv2.VideoCapture("rtsp://admin:admin@123@192.168.0.11:554/1/2?transportmode=unicast&profile=va")  # ✅ IP Camera URL
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

ARM_CONNECTIONS = frozenset([
    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
    (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
    (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
    (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
])

# ✅ Mediapipe Pose Model
with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
    while cap.isOpened():
        if frame_queue.empty():
            continue

        frame = frame_queue.get()
        frame = cv2.resize(frame, (840, 600))  # ✅ Resize frame for performance

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

            # Get required joint positions
            r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
            r_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y]
            r_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y]

            l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
            l_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y]
            l_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y]

            # Calculate angles
            right_elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
            left_elbow_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)

            angle = "right elbow Angle: " + str(int(right_elbow_angle)) + " left elbow Angle: " + str(int(left_elbow_angle)) + " right elbow Angle: "
            status= "swing your arm"

            # # Display angles
            cv2.putText(image, f'R Arm: {int(right_elbow_angle)}', 
                        (250, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(image, f'L Arm: {int(left_elbow_angle)}', 
                        (250, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Check correctness of swing
            if 30 <= right_elbow_angle <= 45 or 15 <= right_elbow_angle <= 25:
                right_status = "Correct"
                cv2.putText(image, "Right Arm Swing: Correct", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                right_status = "Wrong"
                cv2.putText(image, "Right Arm Swing: Wrong", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            if 30 <= left_elbow_angle <= 45 or 15 <= left_elbow_angle <= 25:
                left_status = "Correct"
                cv2.putText(image, "Left Arm Swing: Correct", (50, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                left_status = "Wrong"
                cv2.putText(image, "Left Arm Swing: Wrong", (50, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Save screenshot
            screenshot_path = f"static/screenshots/swing_{int(current_time)}.jpg"
            cv2.imwrite(screenshot_path, frame)

            # Insert data into database
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(current_time))
            cursor.execute("""
                INSERT INTO baju_swing_result (timestamp, angle, status, suggestion, screenshot_path)
                VALUES (?, ?, ?, ?, ?)
            """, (timestamp, angle, "right:"+right_status+" left:"+left_status, "Adjust your arm swing if needed", screenshot_path))
            conn.commit()

            # Draw only arms
            mp_drawing.draw_landmarks(image, results.pose_landmarks, 
                                      frozenset([
                                          (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
                                          (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
                                          (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
                                          (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
                                      ]))


        except Exception as e:
            print(e)
         # Render detections
        # mp_drawing.draw_landmarks(
        #     image, 
        #     results.pose_landmarks, 
        #     ARM_CONNECTIONS,  # Custom connections for arms only
        #     mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
        #     mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        # )
        # Display on screen
        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # ✅ Reduced delay for smoother video
            break

cap.release()
cv2.destroyAllWindows()
conn.close()
