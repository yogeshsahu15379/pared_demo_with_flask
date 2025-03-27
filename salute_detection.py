import cv2
import mediapipe as mp
import numpy as np
import sqlite3
import time

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

def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
        
    return angle 

def calculate_distance(point1, point2):
    point1 = np.array(point1)
    point2 = np.array(point2)
    distance = np.linalg.norm(point1 - point2)
    return distance

last_store_time = 0  

cap = cv2.VideoCapture(0)
## Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        
        current_time = time.time()
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
      
        # Make detection
        results = pose.process(image)
    
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            
            # Get coordinates
            shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            right_index = [landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].y]
            right_eye_outer= [landmarks[mp_pose.PoseLandmark.RIGHT_EYE_OUTER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_EYE_OUTER.value].y]
            thumb = [landmarks[mp_pose.PoseLandmark.RIGHT_THUMB.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_THUMB.value].y]
            right_pinky= [landmarks[mp_pose.PoseLandmark.RIGHT_PINKY.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_PINKY.value].y]
            # Calculate angle
            elbow_angle = calculate_angle(shoulder, elbow, wrist)
            arm_angle = calculate_angle(hip, shoulder, elbow)
            wrist_angle = calculate_angle(elbow, wrist, right_index )
            # distance = calculate_distance(right_eye_outer, thumb)

            status = "सुधार की जरूरत है"
            # Determine suggestion based on arm_angle
            if 85 <= arm_angle <= 92 and 42 <= elbow_angle <= 47 and 170 <= wrist_angle <= 175:
                suggestion = "Perfect position"
                status="Salute is Correct"
            elif arm_angle < 85:
                suggestion = "अपने BAJU को ऊपर उठाएं ["+str(arm_angle)+"]"
            elif arm_angle > 92:
                suggestion = "Lower arm slowly ["+str(arm_angle)+"]"
            elif elbow_angle < 42:
                suggestion = "अपने कोहनी को ऊपर उठाएं ["+str(elbow_angle)+"]"
            elif elbow_angle > 47:
                suggestion = "Lower elbow slightly ["+str(elbow_angle)+"]"
            elif wrist_angle < 170:
                suggestion = "अपने कलाई को ऊपर उठाएं ["+str(wrist_angle)+"]"
            elif wrist_angle > 175:
                suggestion = "Lower wrist slightly ["+str(wrist_angle)+"]"

            # Display suggestion on the screen
            cv2.putText(image, f'Suggestion: {suggestion}', (50, 250), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Visualize angle
            cv2.putText(image, str(int(elbow_angle)), 
                           tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
            cv2.putText(image, str(int(arm_angle)),
                        tuple(np.multiply(shoulder, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
            cv2.putText(image, str(int(wrist_angle)),
                        tuple(np.multiply(wrist, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
            cv2.putText(image, f'Angle of Elbow: {int(elbow_angle)}', (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)    
            cv2.putText(image, f'Angle of Arm: {int(arm_angle)}', (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(image, f'Angle of wrist: {int(wrist_angle)}', (50, 150), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)     
            # cv2.putText(image, f'distence: {int(distance)}', (50, 200), 
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 25), 2)  

              # ✅ Save Screenshot & Data (Every 1 second)
            if int(current_time) - int(last_store_time) >= 1:
                last_store_time = current_time

                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                screenshot_path = f"static/screenshots/salute_{int(time.time())}.jpg"
                cv2.imwrite(screenshot_path, frame)

                cursor.execute("INSERT INTO results (timestamp, angle, status, suggestion, screenshot_path) VALUES (?, ?, ?, ?, ?)",
                               (timestamp, "angle", status, suggestion, screenshot_path))
                conn.commit()        
        # except:
        except Exception as e:
            print(e)
            
            
        

        # # Render detections
        # mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
        #                         mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
        #                         mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
        #                          )          
        # Display on screen
          
        
        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
conn.close()