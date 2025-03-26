# import cv2
# import mediapipe as mp
# import numpy as np
# import sqlite3
# import time

# # MediaPipe Hand Tracking
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# # Database Connection
# conn = sqlite3.connect("salute_results.db", check_same_thread=False)
# cursor = conn.cursor()
# cursor.execute("""
#     CREATE TABLE IF NOT EXISTS results (
#         id INTEGER PRIMARY KEY AUTOINCREMENT,
#         timestamp TEXT,
#         angle REAL,
#         status TEXT,
#         suggestion TEXT,
#         screenshot_path TEXT
#     )
# """)
# conn.commit()

# # Angle Calculation Function
# def calculate_angle(a, b, c):
#     a, b, c = np.array(a), np.array(b), np.array(c)
#     ba = a - b
#     bc = c - b
#     cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
#     angle = np.degrees(np.arccos(cosine_angle))
#     return angle

# # Camera Setup
# # cap = cv2.VideoCapture("http://192.168.72.99:8080/video")
# cap = cv2.VideoCapture(0)
# last_store_time = 0  # ✅ Last time data was stored

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
    
#     current_time = time.time()


#     frame = cv2.flip(frame, 1)
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     # Process Hands
#     results = hands.process(rgb_frame)

#     if results.multi_hand_landmarks:
#         for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
#             handedness = results.multi_handedness[idx].classification[0].label  # Right / Left
            
#             if handedness == "Left":
#                 status = "Salute is Wrong"
#                 suggestion = "Use your right hand for salute."
#                 angle=0
#             else:

#                 # Extract Key Points
#                 wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
#                 index_finger = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
#                 elbow = [wrist.x - 0.2, wrist.y - 0.2]  # Approx elbow position

#                 height, width, _ = frame.shape
#                 wrist = (int(wrist.x * width), int(wrist.y * height))
#                 index_finger = (int(index_finger.x * width), int(index_finger.y * height))
#                 elbow = (int(elbow[0] * width), int(elbow[1] * height))

#                 # Calculate Angle
#                 angle = 180 - calculate_angle(elbow, wrist, index_finger)
#                 status = "Salute is Correct" if 42 <= angle <= 48 else "Salute is Wrong"
#                 suggestion = "Raise hand slightly" if angle < 42 else "Lower hand slightly" if angle > 48 else "Perfect!"

#             # Display on Screen
#             cv2.putText(frame, f'Angle: {int(angle)}°', (50, 50), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#             cv2.putText(frame, status, (50, 90), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if "Wrong" in status else (0, 255, 0), 2)
#             cv2.putText(frame, f'Suggestion: {suggestion}', (50, 130), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

#             # Save Screenshot & Data
#             # if "Correct" in status:
#             if current_time - last_store_time >= 1:
#                 last_store_time = current_time

#                 timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
#                 screenshot_path = f"static/screenshots/salute_{int(time.time())}.jpg"
#                 cv2.imwrite(screenshot_path, frame)

#                 cursor.execute("INSERT INTO results (timestamp, angle, status, suggestion, screenshot_path) VALUES (?, ?, ?, ?, ?)", 
#                                 (timestamp, angle, status, suggestion, screenshot_path))
#                 conn.commit()

#     cv2.imshow("Real-Time Salute Detection", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
# conn.close()

import cv2
import mediapipe as mp
import numpy as np
import sqlite3
import time

# ✅ MediaPipe Hand Tracking Setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

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

# ✅ Function to calculate the angle between three points
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)  
    ba = a - b  # Vector from A to B
    bc = c - b  # Vector from B to C

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(cosine_angle))

    return angle

# ✅ Function to calculate Palm Angle
def calculate_palm_angle(index_tip, wrist):
    index_tip, wrist = np.array(index_tip), np.array(wrist)
    vector = index_tip - wrist  
    palm_angle = np.degrees(np.arctan2(vector[1], vector[0]))

    return abs(palm_angle)  # Absolute value to handle different orientations

# ✅ Camera Setup
# cap = cv2.VideoCapture("rtsp://admin:admin@123@192.168.0.13:554/mode=real&idc=1 &ids=1")
cap = cv2.VideoCapture(0)
last_store_time = 0  

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ✅ Process Hands
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            hand_label = results.multi_handedness[idx].classification[0].label  
            is_right_hand = (hand_label == "Right")  

            if not is_right_hand:
                status = "Salute is Wrong"
                suggestion = "Use your right hand for salute."
                angle = None  
                palm_angle = None  
            else:
                # ✅ Extract Key Points
                height, width, _ = frame.shape
                wrist = (int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * width),
                         int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * height))

                elbow = (int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x * width),
                         int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * height))

                shoulder = (wrist[0], wrist[1] - 100)  # Approximate shoulder above wrist

                index_tip = (int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * width),
                             int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * height))

                # ✅ Calculate Angles
                angle = calculate_angle(shoulder, elbow, wrist)  # Arm angle
                # palm_angle = calculate_palm_angle(index_tip, wrist)  # Palm angle

                # ✅ Army Salute Detection (80°-100° arm angle, 40°-50° palm angle)
                # if 80 <= angle <= 100 and 40 <= palm_angle <= 50:
                if 80 <=angle <= 100:
                    status = "Salute is Correct"
                    suggestion="perfect"
                else:
                    status = "Salute is Wrong"
                    suggestion = "Ensure arm is at 80°-100°"

            # ✅ Display on Screen
            cv2.putText(frame, f'Angle: {int(angle) if angle else "N/A"}°', (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # cv2.putText(frame, f'Palm Angle: {int(palm_angle) if palm_angle else "N/A"}°', (50, 90),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(frame, status, (50, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if "Wrong" in status else (0, 255, 0), 2)
            cv2.putText(frame, f'Suggestion: {suggestion if suggestion else "N/A"}', (50, 170),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            # ✅ Save Screenshot & Data (Every 1 second)
            if current_time - last_store_time >= 1:
                last_store_time = current_time

                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                screenshot_path = f"static/screenshots/salute_{int(time.time())}.jpg"
                cv2.imwrite(screenshot_path, frame)

                cursor.execute("INSERT INTO results (timestamp, angle, status, suggestion, screenshot_path) VALUES (?, ?, ?, ?, ?)",
                               (timestamp, angle, status, suggestion, screenshot_path))
                conn.commit()

    cv2.imshow("Real-Time Army Salute Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
conn.close()
