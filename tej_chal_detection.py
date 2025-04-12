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
    CREATE TABLE IF NOT EXISTS tej_march_result (
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
    a, b, c = np.array(a), np.array(b), np.array(c)
    ab = a - b
    cb = c - b
    angle = np.arccos(np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb)))
    return np.degrees(angle)

def calculate_angle_2d(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360.0 - angle
        
    return angle 

# ✅ Initialize Camera
cap = cv2.VideoCapture("rtsp://admin:admin@123@192.168.0.11:554/1/2?transportmode=unicast&profile=va")  # ✅ IP Camera URL
# cap = cv2.VideoCapture(0)  # ✅ Webcam
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  

frame_queue = queue.Queue()
frame_skip = 2
frame_count = 0
Z_THRESHOLD = -0.2
prev_leg = None

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
with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
    while cap.isOpened():
        if frame_queue.empty():
            continue

        frame = frame_queue.get()
        frame = cv2.resize(frame, (840, 600))
        crop_width = 300  # You can change this to desired cropped width
        frame_height, frame_width, _ = frame.shape
        start_x = (frame_width - crop_width) // 2
        end_x = start_x + crop_width
        frame = frame[:, start_x:end_x]
        h, w, _ = frame.shape

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                def get_landmark_points(landmark):
                    return [landmark.x, landmark.y, landmark.z], (int(landmark.x * w), int(landmark.y * h))

                left_hip, lh_pos = get_landmark_points(landmarks[mp_pose.PoseLandmark.LEFT_HIP])
                left_knee, lk_pos = get_landmark_points(landmarks[mp_pose.PoseLandmark.LEFT_KNEE])
                left_ankle, la_pos = get_landmark_points(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE])
                left_wrist, lw_pos = get_landmark_points(landmarks[mp_pose.PoseLandmark.LEFT_WRIST])

                right_hip, rh_pos = get_landmark_points(landmarks[mp_pose.PoseLandmark.RIGHT_HIP])
                right_knee, rk_pos = get_landmark_points(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE])
                right_ankle, ra_pos = get_landmark_points(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE])
                right_wrist, rw_pos = get_landmark_points(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST])

                # ✅ Check Z-depth for leg raise
                left_z = left_ankle[2]
                right_z = right_ankle[2]

                # check Z-depth for hand raise
                left_wrist_z = left_wrist[2]
                right_wrist_z = right_wrist[2]

                # ✅ Show angles on both sides always
                # Left
                left_hip_angle = calculate_angle([lh_pos[0], lh_pos[1]], [lk_pos[0], lk_pos[1]], [la_pos[0], la_pos[1]])
                left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
                left_ankle_angle = calculate_angle(left_knee, left_ankle, [left_ankle[0], left_ankle[1] + 0.1, left_ankle[2]])
                cv2.putText(image, f"{left_hip_angle:.1f}", lh_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)
                cv2.putText(image, f"{left_knee_angle:.1f}", lk_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)
                cv2.putText(image, f"{left_ankle_angle:.1f}", la_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)

                # Right
                right_hip_angle = calculate_angle([rh_pos[0], rh_pos[1]], [rk_pos[0], rk_pos[1]], [ra_pos[0], ra_pos[1]])
                right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
                right_ankle_angle = calculate_angle(right_knee, right_ankle, [right_ankle[0], right_ankle[1] + 0.1, right_ankle[2]])
                cv2.putText(image, f"{right_hip_angle:.1f}", rh_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)
                cv2.putText(image, f"{right_knee_angle:.1f}", rk_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)
                cv2.putText(image, f"{right_ankle_angle:.1f}", ra_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)

                leg_raised = None
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                screenshot_path = f"static/screenshots/baju_swing_{int(time.time())}.jpg"
                status = "kadam tal wrong"
                suggestion = "Please raise your leg higher."
                angle =f"left_z: {left_z} | Right_z: {right_z} | right_wrist_z: {right_wrist_z} | left_wrist_z: {left_wrist_z} "
                # ✅ Check if leg is raised correctly
                if left_z < Z_THRESHOLD and right_wrist_z < Z_THRESHOLD:
                    status = "left leg and right hand Correct"
                    suggestion = "Left leg and right hand raised correctly."
                    cv2.imwrite(screenshot_path, frame)
                    cursor.execute("INSERT INTO tej_march_result (timestamp, angle, status, suggestion, screenshot_path) VALUES (?, ?, ?, ?, ?)",
                                (timestamp, angle, status, suggestion, screenshot_path))
                    conn.commit()   
                    leg_raised = "left"
                    prev_leg = "left"
                elif right_z < Z_THRESHOLD and left_wrist_z < Z_THRESHOLD:
                    status = "right leg and left hand Correct"
                    suggestion = "Right leg and left hand raised correctly."
                    cv2.imwrite(screenshot_path, frame)
                    cursor.execute("INSERT INTO tej_march_result (timestamp, angle, status, suggestion, screenshot_path) VALUES (?, ?, ?, ?, ?)",
                                (timestamp, angle, status, suggestion, screenshot_path))
                    conn.commit()  
                    leg_raised = "right"
                    prev_leg = "right"
                else:
                    status = "tej chal Wrong"
                    suggestion = "tej chal wrong"
                    cv2.imwrite(screenshot_path, frame)
                    cursor.execute("INSERT INTO tej_march_result (timestamp, angle, status, suggestion, screenshot_path) VALUES (?, ?, ?, ?, ?)",
                                (timestamp, angle, status, suggestion, screenshot_path))
                    conn.commit()  


                # ✅ Display Z index and leg raised info on screen
                z_info = f"Left Z: {left_z:.2f} | Right Z: {right_z:.2f} | Left Wrist Z: {left_wrist_z:.2f} | Right Wrist Z: {right_wrist_z:.2f}"
                cv2.putText(image, z_info, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 100), 2)
                if leg_raised:
                    cv2.putText(image, f"Leg Raised: {leg_raised.upper()}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)

               

                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        except Exception as e:
            print(e)

        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
conn.close()
