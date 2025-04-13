import cv2
import mediapipe as mp
import numpy as np
import sqlite3
import time
import threading
import queue
    
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# ✅ Database
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
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360.0 - angle if angle > 180.0 else angle

# ✅ Camera Setup
camera_urls = [
    "rtsp://admin:admin@123@192.168.0.10:554/1/1?transportmode=unicast&profile=vam",
    "rtsp://admin:admin@123@192.168.0.11:554/1/1?transportmode=unicast&profile=va"
]
caps = [cv2.VideoCapture(url) for url in camera_urls]

for cap in caps:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

frame_queues = [queue.Queue(maxsize=5) for _ in caps]
frame_skip = 2
frame_counters = [0 for _ in caps]

# ✅ Frame Reading Thread
def read_frames(cap, frame_queue):
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if not frame_queue.full():
            frame_queue.put(frame)

threads = [
    threading.Thread(target=read_frames, args=(caps[i], frame_queues[i]), daemon=True)
    for i in range(len(caps))
]
for t in threads:
    t.start()

# ✅ Pose Detection
with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
    while all(cap.isOpened() for cap in caps):
        for i, frame_queue in enumerate(frame_queues):
            if frame_queue.empty():
                continue

            frame_counters[i] += 1
            if frame_counters[i] % frame_skip != 0:
                continue

            frame = frame_queue.get()
            frame = cv2.resize(frame, (640, 480))
            height, width, _ = frame.shape
            cropped_frame = frame[:int(height * 0.6), :]

            image = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                if results.pose_landmarks:
                    lm = results.pose_landmarks.landmark

                    right_shoulder = [lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                      lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    right_elbow = [lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                                   lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    right_wrist = [lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                                   lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                    right_palm = [lm[mp_pose.PoseLandmark.RIGHT_INDEX.value].x,
                                  lm[mp_pose.PoseLandmark.RIGHT_INDEX.value].y]

                    right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
                    palm_angle = calculate_angle(right_elbow, right_wrist, right_palm)

                    accuracy = max(0, 100 - abs(90 - right_elbow_angle))

                    cv2.putText(image, f'Elbow Angle: {int(right_elbow_angle)}', (30, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    cv2.putText(image, f'Palm Angle: {int(palm_angle)}', (30, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
                    cv2.putText(image, f'Accuracy: {int(accuracy)}%', (30, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(image, 'Suggestion: Slow March Module', (30, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    # ✅ Draw landmarks
                    mp_drawing.draw_landmarks(
                        image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                    )
            except Exception as e:
                print(f"[Camera {i+1}] Error:", e)

            # ✅ Display window for each camera
            cv2.imshow(f'Mediapipe Feed - Camera {i+1}', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# ✅ Cleanup
for cap in caps:
    cap.release()
cv2.destroyAllWindows()
conn.close()
