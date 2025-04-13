# import cv2
# import mediapipe as mp
# import numpy as np

# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose()
# mp_drawing = mp.solutions.drawing_utils

# cap = cv2.VideoCapture("rtsp://admin:Admin@123@192.168.0.13:554/1/2?transmode=unicast&profile=vam")

# current_angle = 0
# current_status = "Analyzing..."
# frame_counter = 0  # Add a global frame counter


# def calculate_angle(a, b, c):
#     a = np.array(a)
#     b = np.array(b)
#     c = np.array(c)
#     ba = a - b
#     bc = c - b
#     cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
#     angle = np.arccos(np.clip(cosine, -1.0, 1.0))
#     return np.degrees(angle)


# def get_frame():
#     global current_angle, current_status, frame_counter
#     success, frame = cap.read()
#     if not success:
#         return None

#     frame_counter += 1
#     if frame_counter % 2 != 0:  # Skip alternate frames
#         return None

#     image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = pose.process(image)

#     if results.pose_landmarks:
#         mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
#         landmarks = results.pose_landmarks.landmark
#         h, w, _ = frame.shape

#         shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w,
#                     landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h]
#         elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * w,
#                  landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * h]
#         wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * w,
#                  landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * h]

#         angle = calculate_angle(shoulder, elbow, wrist)
#         current_angle = int(angle)
#         current_status = "Correct" if 80 <= angle <= 100 else "Wrong"
#         cv2.putText(frame, f'Suggestion: {angle}', (50, 50),
#                             cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
#     return frame


# def gen_frames():
#     while True:
#         frame = get_frame()
#         if frame is None:
#             continue
#         ret, buffer = cv2.imencode('.jpg', frame)
#         frame = buffer.tobytes()
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# def get_pose_data():
#     return current_angle, current_status


#----------------------------------------------------#
import cv2
import numpy as np
import mediapipe as mp
import threading

mp_pose = mp.solutions.pose

output_frame = None
pose_data = None
pose_lock = threading.Lock()

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def frame_worker():
    global output_frame, pose_data

    cap = cv2.VideoCapture("rtsp://admin:Admin@123@192.168.0.13:554/1/2?transmode=unicast&profile=vam")
    frame_count = 0
    process_every_nth_frame = 2

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % process_every_nth_frame != 0:
                continue

            frame = cv2.flip(frame, 1)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            results = pose.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            angle = 0
            status = "No pose detected"

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                shoulder = [landmarks[11].x, landmarks[11].y]
                elbow = [landmarks[13].x, landmarks[13].y]
                wrist = [landmarks[15].x, landmarks[15].y]

                angle = calculate_angle(shoulder, elbow, wrist)

                if angle > 160:
                    status = "Arm straight"
                elif angle < 30:
                    status = "Arm bent"
                else:
                    status = "In between"

                mp.solutions.drawing_utils.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
                )

            with pose_lock:
                output_frame = image.copy()
                pose_data = {
                    "angle": round(angle, 2),
                    "status": status
                }

    cap.release()

def get_frame():
    with pose_lock:
        if output_frame is None:
            return None
        _, buffer = cv2.imencode('.jpg', output_frame)
        return buffer.tobytes()

def get_pose_data():
    with pose_lock:
        return pose_data

