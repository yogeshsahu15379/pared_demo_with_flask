import math
import os
import cv2
import mediapipe as mp
import numpy as np
import sqlite3
import time
import threading
import queue
import argparse
# from config import Config
# from models.drill import DRILL_CAMERA_URL_MAP, DRILL_SLUG_MAP

parser = argparse.ArgumentParser()
parser.add_argument("--user_id", required=True)
parser.add_argument("--user_session_id", required=True)
parser.add_argument("--table_name", required=True)

args = parser.parse_args()

# print(Config,"+++++++")

user_id = args.user_id
user_session_id = args.user_session_id
table_name = args.table_name

print(f"Received user_id: {user_id}")
print(f"Received user_session_id: {user_session_id}")
print(f"Received table_name: {table_name}")

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# ✅ Database Connection
conn = sqlite3.connect("test_db.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute(f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        angle REAL,
        palm_angle REAL,
        status TEXT,
        suggestion TEXT,
        screenshot_path TEXT,
        user_id TEXT,
        session_id TEXT
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

def get_distance(pose1, pose2):
    return round(math.sqrt((pose2[0] - pose1[0])**2 + (pose2[1] - pose1[1])**2) * 100,2)

# ✅ Initialize Camera
cap = cv2.VideoCapture(
    "rtsp://admin:admin@123@192.168.0.14:554/1/2?transmode=unicast&profile=vam"
)  # Replace with your video source
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
                left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                right_foot_index = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y]
                left_foot_index = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]
                right_heel = [landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y]
                left_heel = [landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y]
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                # Calculate angle
                elbow_angle = calculate_angle(shoulder, elbow, wrist)
                arm_angle = calculate_angle(hip, shoulder, elbow)
                wrist_angle = calculate_angle(elbow, wrist, right_index)
                arm_straight_angle = calculate_angle(left_shoulder, shoulder, elbow)
                left_hand_angle = calculate_angle(left_shoulder,left_elbow,left_wrist)
                both_foot_distence = get_distance(right_foot_index,left_foot_index)
                both_heel_distence = get_distance(left_heel,right_heel)
                left_shoulder_angle = calculate_angle(shoulder,left_shoulder,left_wrist)
                left_elbow_angle = calculate_angle(left_shoulder,left_elbow,left_wrist)


                angle = "Elbow Angle: " + str(int(elbow_angle)) + " Arm Angle: " + str(int(arm_angle)) + " Wrist Angle: " + str(int(wrist_angle)) + " Foot distence"+ str(both_foot_distence) + " left shoulder angle: " + str(int(left_shoulder_angle)) +"heel distance: " + str(both_heel_distence)+ "left elbow angle: "+ str(int(left_elbow_angle))
                status = "Salute is wrong"

                # Determine suggestion based on arm_angle
                if 85 <= arm_angle <= 120 and 15 <= elbow_angle <= 35 and 160 <= wrist_angle <= 180 and 5< both_foot_distence <9.5 and 90< left_shoulder_angle < 105 and 160 < left_elbow_angle < 180 and both_heel_distence < 4.5:
                    suggestion = "Perfect position"
                    status = "Salute is Correct"
                elif both_foot_distence <5:
                    suggestion = "Dono pair thoda aur khole"
                elif both_heel_distence >4.5:
                    suggestion = "dono heel apas pr chipki hui honi chahiye. "

                elif both_foot_distence > 9.5 :
                    suggestion = f"Dono panjo k beech ki duri ko {int(both_foot_distence - 9.5)}inch kam kre."
                elif 105 < left_shoulder_angle < 90:
                    suggestion = "left hand left pair k pant ki pocket k sath chupka hua hona chahiye"
                elif 160 > left_elbow_angle < 180:
                    suggestion = "left kohgni body k sath chipki hui honi chahiye."
                elif arm_angle < 85:
                    suggestion = "Raise arm slowly [" + str(int(arm_angle)) + "]"
                elif arm_angle > 120:
                    suggestion = "Lower arm slowly [" + str(int(arm_angle)) + "]"
                elif elbow_angle < 15:
                    suggestion = "Raise elbow slightly [" + str(int(elbow_angle)) + "]"
                elif elbow_angle > 35:
                    suggestion = "Lower elbow slightly [" + str(int(elbow_angle)) + "]"
                elif wrist_angle < 160:
                    suggestion = "Raise wrist slightly [" + str(int(wrist_angle)) + "]"
                elif wrist_angle > 180:
                    suggestion = "Lower wrist slightly [" + str(int(wrist_angle)) + "]"

                # Display suggestion on the screen
                color = (0, 255, 0) if "Correct" in status else (0, 0, 255)
                cv2.putText(centered_image, f'Suggestion: {suggestion}', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 2)
                
                # cv2.putText(centered_image,f'heel distence: {both_heel_distence}',(10,200),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5,color,2)
                
                # cv2.putText(centered_image,f'foot distence: {both_foot_distence}',(10,150),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5,color,2)
                
                # if 6.5< both_foot_distence <9.5 :
                # cv2.putText(centered_image,f"shoulder angle {left_shoulder_angle}", (10,250),
                #             cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

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
                cv2.putText(centered_image, str(int(left_hand_angle)),
                            tuple(np.multiply(left_elbow, [roi_width, roi_height]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2, cv2.LINE_AA)

                # ✅ Save Screenshot & Data (Every 1 second)
                if int(current_time) - int(last_store_time) >= 1 and 80 <= arm_angle <= 150 and 15 <= elbow_angle <= 40:
                    last_store_time = current_time

                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    # Get current file's directory
                    current_dir = os.path.dirname(os.path.abspath(__file__))

                    # Go one level up
                    parent_dir = os.path.dirname(current_dir)

                    # Create screenshots path in the parent folder
                    screenshot_dir = os.path.join(parent_dir, "screenshots")

                    # Create the directory if it doesn't exist
                    os.makedirs(screenshot_dir, exist_ok=True)

                    centered_image_path = os.path.join(
                        screenshot_dir, 
                        f"{user_session_id}_centered_salute_{int(time.time())}.jpg"
                    )

                    # Save screenshot without suggestion
                    screenshot_without_text = centered_image.copy()
                    cv2.imwrite(centered_image_path, image)

                    cursor.execute(
                        f"INSERT INTO {table_name} (timestamp, angle, status, suggestion, screenshot_path, user_id, session_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (timestamp, angle, status, suggestion, centered_image_path, user_id, user_session_id),
                    )
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

