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
    CREATE TABLE IF NOT EXISTS hill_march_result (
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

# ✅ Initialize Camera
cap = cv2.VideoCapture("rtsp://admin:admin@123@192.168.0.11:554/1/2?transmode=unicast&profile=va")   
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

frame_queue = queue.Queue()
frame_skip = 2
frame_count = 0
Z_THRESHOLD = -0.2
leg_in_air = False

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
        h, w, _ = frame.shape
        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        # ✅ Center Calculation
        cx, cy = w // 2, h // 2
        roi_width, roi_height = 300, 600
        x1 = max(0, cx - roi_width // 2)
        y1 = max(0, cy - roi_height // 2)
        x2 = min(w, cx + roi_width // 2)
        y2 = min(h, cy + roi_height // 2)

        # ✅ Draw Center and ROI on Original Frame
        frame_display = frame.copy()
        cv2.line(frame_display, (cx - 20, cy), (cx + 20, cy), (0, 255, 0), 2)  # Center +
        cv2.line(frame_display, (cx, cy - 20), (cx, cy + 20), (0, 255, 0), 2)
        cv2.rectangle(frame_display, (x1, y1), (x2, y2), (255, 0, 0), 2)       # ROI Box

        # ✅ Extract ROI and send to MediaPipe
        roi = frame[y1:y2, x1:x2]
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        roi_rgb.flags.writeable = False
        results = pose.process(roi_rgb)
        roi_rgb.flags.writeable = True
        image = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2BGR)

        try:
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                def get_landmark_points(landmark):
                    return [landmark.x, landmark.y, landmark.z], (
                        int(landmark.x * roi.shape[1]),
                        int(landmark.y * roi.shape[0])
                    )

                # Get joint positions and angles
                left_hip, lh_pos = get_landmark_points(landmarks[mp_pose.PoseLandmark.LEFT_HIP])
                left_knee, lk_pos = get_landmark_points(landmarks[mp_pose.PoseLandmark.LEFT_KNEE])
                left_ankle, la_pos = get_landmark_points(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE])
                right_hip, rh_pos = get_landmark_points(landmarks[mp_pose.PoseLandmark.RIGHT_HIP])
                right_knee, rk_pos = get_landmark_points(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE])
                right_ankle, ra_pos = get_landmark_points(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE])

                left_z = left_ankle[2]
                right_z = right_ankle[2]

                left_hip_angle = calculate_angle([lh_pos[0], lh_pos[1]], [lk_pos[0], lk_pos[1]], [la_pos[0], la_pos[1]])
                left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
                left_ankle_angle = calculate_angle(left_knee, left_ankle, [left_ankle[0], left_ankle[1] + 0.1, left_ankle[2]])

                right_hip_angle = calculate_angle([rh_pos[0], rh_pos[1]], [rk_pos[0], rk_pos[1]], [ra_pos[0], ra_pos[1]])
                right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
                right_ankle_angle = calculate_angle(right_knee, right_ankle, [right_ankle[0], right_ankle[1] + 0.1, right_ankle[2]])

                # ✅ Draw angles at joints
                for point, angle in zip([lh_pos, lk_pos, la_pos, rh_pos, rk_pos, ra_pos],
                                        [left_hip_angle, left_knee_angle, left_ankle_angle,
                                        right_hip_angle, right_knee_angle, right_ankle_angle]):
                    cv2.putText(image, f"{angle:.1f}", point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                # ✅ Determine suggestion & status
                leg_raised = None
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                screenshot_path = f"static/screenshots/hill_march_{int(time.time())}.jpg"
                status = "Hill march wrong"
                suggestion = "Please raise your leg higher."
                angle = f"left_z: {left_z} | Right_z: {right_z} | left_hip_angle: {left_hip_angle:.1f} | left_knee_angle: {left_knee_angle:.1f} | left_ankle_angle: {left_ankle_angle:.1f} | right_hip_angle: {right_hip_angle:.1f} | right_knee_angle: {right_knee_angle:.1f} | right_ankle_angle: {right_ankle_angle:.1f}"

                if not leg_in_air and (left_z < Z_THRESHOLD or right_z < Z_THRESHOLD):
                    if left_z < Z_THRESHOLD:
                        status = "Left leg Correct"
                        suggestion = "Left leg raised correctly."
                        leg_raised = "left"
                    elif right_z < Z_THRESHOLD:
                        status = "Right leg Correct"
                        suggestion = "Right leg raised correctly."
                        leg_raised = "right"

                    # ✅ Save to DB and Screenshot
                    cv2.imwrite(screenshot_path, roi)
                    cursor.execute("INSERT INTO hill_march_result (timestamp, angle, status, suggestion, screenshot_path) VALUES (?, ?, ?, ?, ?)",
                                (timestamp, angle, status, suggestion, screenshot_path))
                    conn.commit()
                    leg_in_air = True

                # ✅ Reset
                if left_z >= Z_THRESHOLD and right_z >= Z_THRESHOLD:
                    leg_in_air = False

                # ✅ Draw Z-depth & Suggestions
                z_info = f"Left Z: {left_z:.2f} | Right Z: {right_z:.2f}"
                cv2.putText(image, z_info, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)

                if leg_raised:
                    cv2.putText(image, f"Leg Raised: {leg_raised.upper()}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.putText(image, f"Status: {status}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(image, f"Suggestion: {suggestion}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # ✅ Draw full pose
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        except Exception as e:
            print("Error:", e)

        cv2.imshow("Processed Feed (ROI)", image)
        cv2.imshow("Full Frame with Center & ROI", frame_display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
conn.close()
