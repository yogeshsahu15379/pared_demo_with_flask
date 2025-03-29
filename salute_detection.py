import cv2
import mediapipe as mp
import numpy as np
import sqlite3
import time
import threading
import queue

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic  # ✅ Corrected import

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

def get_distance(landmark_list):
    if len(landmark_list) < 2:
        return 0  # Return 0 if there are not enough landmarks
    x1, y1 = landmark_list[0].x, landmark_list[0].y  # ✅ Access x, y attributes explicitly
    x2, y2 = landmark_list[1].x, landmark_list[1].y  # ✅ Access x, y attributes explicitly
    distance = np.hypot(x2 - x1, y2 - y1)
    return np.interp(distance, [0, 1], [0, 1000])  # Scale the distance

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
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # ✅ Reduce frame width for faster processing
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)  # ✅ Reduce frame height for faster processing
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  

frame_queue = queue.Queue()  # ✅ Queue for multi-threading
last_store_time = 0  
frame_skip = 4  # ✅ Process every 4th frame to reduce load
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

# ✅ Mediapipe Holistic Model
with mp_holistic.Holistic(min_detection_confidence=0.2, min_tracking_confidence=0.2) as holistic:  # ✅ Lower confidence thresholds
    while cap.isOpened():
        if frame_queue.empty():
            continue

        frame = frame_queue.get()
        frame = cv2.resize(frame, (640, 480))  # ✅ Resize frame for performance (optional)

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue  # ✅ Skip frames to improve performance
        
        current_time = time.time()
        
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
      
        # Make detection
        results = holistic.process(image)
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        try:
            # ✅ Extract pose and hand landmarks
            landmarks = results.pose_landmarks.landmark if results.pose_landmarks else []
            right_hand_landmarks = results.right_hand_landmarks.landmark if results.right_hand_landmarks else []
            left_hand_landmarks = results.left_hand_landmarks.landmark if results.left_hand_landmarks else []

            # Ensure required pose landmarks exist
            if len(landmarks) > max(mp_holistic.PoseLandmark.RIGHT_SHOULDER.value, 
                                     mp_holistic.PoseLandmark.LEFT_SHOULDER.value, 
                                     mp_holistic.PoseLandmark.RIGHT_ELBOW.value, 
                                     mp_holistic.PoseLandmark.RIGHT_WRIST.value, 
                                     mp_holistic.PoseLandmark.RIGHT_HIP.value, 
                                     mp_holistic.PoseLandmark.RIGHT_INDEX.value):
                shoulder = [landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value].y]
                left_shoulder = [landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW.value].y]
                wrist = [landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST.value].y]
                hip = [landmarks[mp_holistic.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_holistic.PoseLandmark.RIGHT_HIP.value].y]
                right_index = [landmarks[mp_holistic.PoseLandmark.RIGHT_INDEX.value].x, landmarks[mp_holistic.PoseLandmark.RIGHT_INDEX.value].y]

                # Calculate angles for pose
                elbow_angle = calculate_angle(shoulder, elbow, wrist)
                arm_angle = calculate_angle(hip, shoulder, elbow)
                wrist_angle = calculate_angle(elbow, wrist, right_index)
                right_shoulder_align_left_shoulder = calculate_angle(left_shoulder, shoulder, elbow)
            else:
                elbow_angle = arm_angle = wrist_angle = right_shoulder_align_left_shoulder = None

            # Ensure required hand landmarks exist
            if len(right_hand_landmarks) > 8:  # Ensure landmarks 4, 5, and 8 exist
                thumb_distance = get_distance([right_hand_landmarks[4], right_hand_landmarks[5]])
                index_finger_eyebrow__distance = get_distance([right_hand_landmarks[8], landmarks[6]]) if len(landmarks) > 6 else None
            else:
                thumb_distance = index_finger_eyebrow__distance = None

            # Validate calculated values before comparison
            if (elbow_angle is not None and arm_angle is not None and wrist_angle is not None and 
                thumb_distance is not None and index_finger_eyebrow__distance is not None and 
                right_shoulder_align_left_shoulder is not None):
                if (90 <= arm_angle <= 115 and 47 <= elbow_angle <= 65 and 155 <= wrist_angle <= 170 and 
                    20 <= thumb_distance <= 30 and 20 <= index_finger_eyebrow__distance <= 50 and 
                    170 <= right_shoulder_align_left_shoulder <= 175):
                    suggestion = "Perfect position"
                    status = "Salute is Correct"
                elif arm_angle < 90:
                    suggestion = "Raise arm slowly [" + str(int(arm_angle)) + "]"
                elif arm_angle > 115:
                    suggestion = "Lower arm slowly [" + str(int(arm_angle)) + "]"
                elif elbow_angle < 47:
                    suggestion = "Raise elbow slightly [" + str(int(elbow_angle)) + "]"
                elif elbow_angle > 65:
                    suggestion = "Lower elbow slightly [" + str(int(elbow_angle)) + "]"
                elif wrist_angle < 155:
                    suggestion = "Raise wrist slightly [" + str(int(wrist_angle)) + "]"
                elif wrist_angle > 170:
                    suggestion = "Lower wrist slightly [" + str(int(wrist_angle)) + "]"
                elif thumb_distance < 20:
                    suggestion = "Move thumb closer to index finger [" + str(int(thumb_distance)) + "]"
                elif thumb_distance > 30:
                    suggestion = "Move thumb away from index finger [" + str(int(thumb_distance)) + "]"
                elif index_finger_eyebrow__distance < 20:
                    suggestion = "Move index finger closer to eyebrow [" + str(int(index_finger_eyebrow__distance)) + "]"
                elif index_finger_eyebrow__distance > 50:
                    suggestion = "Move index finger away from eyebrow [" + str(int(index_finger_eyebrow__distance)) + "]"
                elif right_shoulder_align_left_shoulder < 170:
                    suggestion = "Align right shoulder with left shoulder [" + str(int(right_shoulder_align_left_shoulder)) + "]"
                elif right_shoulder_align_left_shoulder > 175:
                    suggestion = "Align right shoulder with left shoulder [" + str(int(right_shoulder_align_left_shoulder)) + "]"
            else:
                suggestion = "Unable to calculate pose or hand landmarks"
                status = "Error"

            # Display suggestion on the screen
            color = (0, 255, 0) if "Correct" in status else (0, 0, 255)
            cv2.putText(image, f'Suggestion: {suggestion}', (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # Display calculated angles and distances
            cv2.putText(image, f'Elbow Angle: {int(elbow_angle)} deg' if elbow_angle is not None else 'Elbow Angle: N/A', 
                        (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
            cv2.putText(image, f'Arm Angle: {int(arm_angle)} deg' if arm_angle is not None else 'Arm Angle: N/A', 
                        (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
            cv2.putText(image, f'Wrist Angle: {int(wrist_angle)} deg' if wrist_angle is not None else 'Wrist Angle: N/A', 
                        (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
            cv2.putText(image, f'Shoulder Alignment: {int(right_shoulder_align_left_shoulder)} deg' if right_shoulder_align_left_shoulder is not None else 'Shoulder Alignment: N/A', 
                        (50, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
            cv2.putText(image, f'Thumb Distance: {int(thumb_distance)}' if thumb_distance is not None else 'Thumb Distance: N/A', 
                        (50, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
            cv2.putText(image, f'Index-Eyebrow Distance: {int(index_finger_eyebrow__distance)}' if index_finger_eyebrow__distance is not None else 'Index-Eyebrow Distance: N/A', 
                        (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)

            # ✅ Save Screenshot & Data (Every 1 second)
            if int(current_time) - int(last_store_time) >= 1:
                last_store_time = current_time

                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                screenshot_path = f"static/screenshots/salute_{int(time.time())}.jpg"
                cv2.imwrite(screenshot_path, frame)

                cursor.execute("INSERT INTO results (timestamp, angle, status, suggestion, screenshot_path) VALUES (?, ?, ?, ?, ?)",
                               (timestamp, "angle", status, suggestion, screenshot_path))
                conn.commit()        

        except Exception as e:
            print(e)
        # ✅ Render detections (Optional: Disable for faster processing)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
        
        # ✅ Draw hand landmarks
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
        
        # Display on screen
        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # ✅ Reduced delay for smoother video
            break

cap.release()
cv2.destroyAllWindows()
conn.close()
