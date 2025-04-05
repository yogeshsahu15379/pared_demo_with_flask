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
cap = cv2.VideoCapture("rtsp://admin:admin@123@192.168.0.10:554/1/2?transportmode=unicast&profile=va")  # ✅ IP Camera URL
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
        
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
      
        # Make detection
        results = pose.process(image)
    
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        try:
            if results.pose_landmarks:  # ✅ Check if pose landmarks are detected
                landmarks = results.pose_landmarks.landmark
                
                # Get coordinates
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                right_index = [landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].y]
                right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                right_foot_index = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y]
                right_heel = [landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y]
                
                # Get coordinates for left side                
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                left_index = [landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].x, landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].y]
                left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                left_foot_index = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x, landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]
                left_heel = [landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y]
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
               
                # Calculate angle
                right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
                right_arm_angle = calculate_angle(right_hip, right_shoulder, right_elbow)
                right_wrist_angle = calculate_angle(right_elbow, right_wrist, right_index)
                right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
                right_ankle_angle = calculate_angle(right_knee, right_ankle, right_foot_index)
                right_hip_angle = calculate_angle(right_shoulder,right_hip,right_ankle)
                
                left_ankle_angle = calculate_angle(left_knee, left_ankle, left_foot_index)
                left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
                left_wrist_angle = calculate_angle(left_elbow, left_wrist, left_index)
                left_arm_angle = calculate_angle(left_hip, left_shoulder, left_elbow)
                left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                left_hip_angle = calculate_angle(left_shoulder,left_hip,left_ankle)
                # TO-Do will remove this later
                arm_straight_angle = calculate_angle(left_shoulder, right_shoulder, right_elbow)

                angle = "Elbow Angle: " + str(int(right_elbow_angle)) + " Arm Angle: " + str(int(right_arm_angle)) + " Wrist Angle: " + str(int(right_wrist_angle))
                status = "Salute is wrong"
                suggestion = "Slow March"

                # if right_knee_angle < 150 and left_knee_angle > 170:
                #     if 172 <= right_elbow_angle <= 180 and 170 <= right_wrist_angle <= 180 and 85 <= right_knee_angle <= 100 and 95 <= right_ankle_angle <= 110:
                #         suggestion = "Perfect Right Leg Up Position"
                #         status = "Salute is Correct"
                #     else:
                #         if right_elbow_angle < 172:
                #             suggestion = f"Straighten right elbow slightly [{int(right_elbow_angle)}]"
                #         elif right_elbow_angle > 180:
                #             suggestion = f"Bend right elbow slightly [{int(right_elbow_angle)}]"
                #         elif right_wrist_angle < 170:
                #             suggestion = f"Straighten right wrist slightly [{int(right_wrist_angle)}]"
                #         elif right_wrist_angle > 180:
                #             suggestion = f"Bend right wrist slightly [{int(right_wrist_angle)}]"
                #         elif right_knee_angle < 85:
                #             suggestion = f"Raise right knee slightly [{int(right_knee_angle)}]"
                #         elif right_knee_angle > 100:
                #             suggestion = f"Lower right knee slightly [{int(right_knee_angle)}]"
                #         elif right_ankle_angle < 95:
                #             suggestion = f"Raise right ankle slightly [{int(right_ankle_angle)}]"
                #         elif right_ankle_angle > 110:
                #             suggestion = f"Lower right ankle slightly [{int(right_ankle_angle)}]"
                    
                # elif left_knee_angle < 150 and right_knee_angle > 170:
                #     if 172 <= left_elbow_angle <= 180 and 170 <= left_wrist_angle <= 180 and 85 <= left_knee_angle <= 100 and 95 <= left_ankle_angle <= 110:
                #         suggestion = "Perfect Right Leg Up Position"
                #         status = "Salute is Correct"
                #     else:
                #         if left_elbow_angle < 172:
                #             suggestion = f"Straighten left elbow slightly [{int(left_elbow_angle)}]"
                #         elif left_elbow_angle > 180:
                #             suggestion = f"Bend left elbow slightly [{int(left_elbow_angle)}]"
                #         elif left_wrist_angle < 170:
                #             suggestion = f"Straighten left wrist slightly [{int(left_wrist_angle)}]"
                #         elif left_wrist_angle > 180:
                #             suggestion = f"Bend left wrist slightly [{int(left_wrist_angle)}]"
                #         elif left_knee_angle < 85:
                #             suggestion = f"Raise left knee slightly [{int(left_knee_angle)}]"
                #         elif left_knee_angle > 100:
                #             suggestion = f"Lower left knee slightly [{int(left_knee_angle)}]"
                #         elif left_ankle_angle < 95:
                #             suggestion = f"Raise left ankle slightly [{int(left_ankle_angle)}]"
                #         elif left_ankle_angle > 110:
                #             suggestion = f"Lower left ankle slightly [{int(left_ankle_angle)}]"
                # else:
                #     suggestion = "Invalid state: Both legs are either up or grounded."
                

                # Display suggestion on the screen
                color = (0, 255, 0) if "Correct" in status else (0, 0, 255)
                cv2.putText(image, f'Suggestion: {suggestion}', (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(image, f'Suggestion: Slow March module', (50, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # # Visualize angle
                cv2.putText(image, str(int(right_elbow_angle)), 
                               tuple(np.multiply(right_elbow, [840, 600]).astype(int)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA
                                    )
                cv2.putText(image, str(int(right_arm_angle)),
                            tuple(np.multiply(right_shoulder, [840, 600]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA
                                    )
                cv2.putText(image, str(int(right_wrist_angle)),
                            tuple(np.multiply(right_wrist, [840, 600]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA
                                    )
                cv2.putText(image, str(int(arm_straight_angle)),
                            tuple(np.multiply(left_shoulder, [840, 600]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA
                                    )
                
                cv2.putText(image, str(int(right_knee_angle)),
                            tuple(np.multiply(right_knee, [840, 600]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA
                                    )
                
                cv2.putText(image, str(int(right_hip_angle)),
                            tuple(np.multiply(right_hip, [840, 600]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA
                                    )
                
                cv2.putText(image, str(int(right_ankle_angle)),
                            tuple(np.multiply(right_ankle, [840, 600]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA
                                    )
                cv2.putText(image,str(int(left_ankle_angle)),
                            tuple(np.multiply(left_ankle, [840, 600]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA
                                    )
                cv2.putText(image, str(int(left_elbow_angle)), 
                               tuple(np.multiply(left_elbow, [840, 600]).astype(int)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA
                                    )
                cv2.putText(image, str(int(left_arm_angle)),
                            tuple(np.multiply(left_shoulder, [840, 600]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA
                                    )
                cv2.putText(image, str(int(left_wrist_angle)),
                            tuple(np.multiply(left_wrist, [840, 600]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA
                                    )
                cv2.putText(image, str(int(left_knee_angle)),
                            tuple(np.multiply(left_knee, [840, 600]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA
                                    )
                cv2.putText(image, str(int(left_elbow_angle)),
                            tuple(np.multiply(left_elbow, [840, 600]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA
                                    )
                 
                cv2.putText(image, str(int(left_hip_angle)),
                            tuple(np.multiply(left_hip, [840, 600]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA
                                    )
                
                

                # # ✅ Save Screenshot & Data (Every 1 second)
                # if int(current_time) - int(last_store_time) >= 1 and 60 <= right_arm_angle <= 150 and 30 <= right_elbow_angle <= 80:
                #     last_store_time = current_time

                #     timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                #     screenshot_path = f"static/screenshots/salute_{int(time.time())}.jpg"
                #     cv2.imwrite(screenshot_path, frame)

                #     cursor.execute("INSERT INTO results (timestamp, angle, status, suggestion, screenshot_path) VALUES (?, ?, ?, ?, ?)",
                #                    (timestamp, angle, status, suggestion, screenshot_path))
                #     conn.commit()
            # else:
                # print("No pose landmarks detected.")  # ✅ Log when no landmarks are detected

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
