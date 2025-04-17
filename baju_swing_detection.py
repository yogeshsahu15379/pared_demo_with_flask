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
        palm_angle REAL, b 
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
# rtsp://192.168.0.11:554/1/1?transmode=unicast&profile=va
# ✅ Initialize Camera
cap = cv2.VideoCapture("rtsp://admin:admin@123@192.168.0.12:554/1/2?transmode=unicast&profile=va")   
# testing
# cap = cv2.VideoCapture(0)  # ✅ IP Camera URL

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

low_knee_position = None
previous_knee_position = None
data ={
    "timestamp": None,
    "angle": None,
    "status": None,
    "suggestion": None,
    "screenshot_path": None
}

# ✅ Mediapipe Pose Model
with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
    while cap.isOpened():
        if frame_queue.empty():
            continue

        frame = frame_queue.get()
        frame = cv2.resize(frame, (840, 600))  # ✅ Resize frame for performance

        crop_width = 300  # You can change this to desired cropped width
        frame_height, frame_width, _ = frame.shape
        start_x = (frame_width - crop_width) // 2
        end_x = start_x + crop_width
        frame = frame[:, start_x:end_x]

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
                right_hip_angle = calculate_angle(right_shoulder,right_hip,right_knee)
                
                left_ankle_angle = calculate_angle(left_knee, left_ankle, left_foot_index)
                left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
                left_wrist_angle = calculate_angle(left_elbow, left_wrist, left_index)
                left_arm_angle = calculate_angle(left_hip, left_shoulder, left_elbow)
                left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                left_hip_angle = calculate_angle(left_shoulder,left_hip,left_knee)
                # TO-Do will remove this later
                arm_straight_angle = calculate_angle(left_shoulder, right_shoulder, right_elbow)

                angle = "right elbow: " + str(int(right_elbow_angle)) + " left elbow: " + str(int(left_elbow_angle)) + " right shoulder: " + str(int(right_arm_angle)) +" left shoulder: "+ str(int(left_arm_angle))
                status = "both hands are straight."
                leg= "no leg up"
                suggestion = "baju swing" 
                
                if(right_elbow_angle >155 and left_elbow_angle > 155 and right_arm_angle >90 and left_arm_angle > 90 ):
                    suggestion = "baju swing correct"
                    status = "Correct"

                else:
                    status = "baju swing Wrong"
                    if right_elbow_angle < 155:
                        suggestion =f"Right hand should be straight [right elbow: {int(right_elbow_angle)}]"
                    elif left_elbow_angle < 155:
                        suggestion = f"Left hand should be straight: [left elbow: {int(left_elbow_angle)}]"
                    elif right_arm_angle > 90:
                        suggestion = f"Rise Right hand Slowly: [right shoulder: {int(right_arm_angle)}]"
                    elif left_arm_angle > 90:
                        suggestion = f"Rise Left hand Slowly: [left shoulder: {int(left_arm_angle)}]"
                
                if right_arm_angle > 80 or left_arm_angle >80:
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    screenshot_path = f"static/screenshots/baju_swing_{int(time.time())}.jpg"
                    cv2.imwrite(screenshot_path, frame)
                    cursor.execute("INSERT INTO baju_swing_result (timestamp, angle, status, suggestion, screenshot_path) VALUES (?, ?, ?, ?, ?)",
                                (timestamp, angle, status, suggestion, screenshot_path))
                    conn.commit()    
 # # Visualize angle
                cv2.putText(image, str(int(right_elbow_angle)), 
                            tuple(np.multiply(right_elbow, [300, 600]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA
                                    )
                cv2.putText(image, str(int(right_arm_angle)),
                            tuple(np.multiply(right_shoulder, [300, 600]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA
                                    )
                cv2.putText(image, str(int(right_wrist_angle)),
                            tuple(np.multiply(right_wrist, [300, 600]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA
                                    )
                cv2.putText(image, str(int(left_elbow_angle)), 
                            tuple(np.multiply(left_elbow, [300, 600]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA
                                    )
                cv2.putText(image, str(int(left_arm_angle)),
                            tuple(np.multiply(left_shoulder, [300, 600]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA
                                    )
                cv2.putText(image, str(int(left_wrist_angle)),
                            tuple(np.multiply(left_wrist, [300, 600]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA
                                    )
                # Display suggestion on the screen
                color = (0, 255, 0) if "Correct" in status else (0, 0, 255)
                cv2.putText(image, f'Suggestion: {suggestion}', (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)


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
