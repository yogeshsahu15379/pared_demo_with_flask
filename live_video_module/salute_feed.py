import cv2
import mediapipe as mp
import numpy as np


mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

frame_skip = 3
frame_count = 0

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360.0 - angle
        
    return angle 


def generate_frames():
    global frame_count
    cap = cv2.VideoCapture("rtsp://admin:admin@123@192.168.0.14:554/1/2?transmode=unicast&profile=vam")  # Replace with your video source

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while True:
                success, frame = cap.read()
                if not success:
                    break

                frame = cv2.resize(frame, (840, 600))
                

                frame_count += 1

                if frame_count % frame_skip == 0:
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
                    # ✅ Convert ROI to RGB for MediaPipe
                    roi_rgb = cv2.cvtColor(centered_image, cv2.COLOR_BGR2RGB)
                    roi_rgb.flags.writeable = False
                    results = pose.process(roi_rgb)
                    roi_rgb.flags.writeable = True
                    centered_image = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2BGR)
                    frame = centered_image

                    if results.pose_landmarks:
                        landmarks = results.pose_landmarks.landmark
                        # Get coordinates
                        shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                        elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                        wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                        hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                        right_index = [landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].y]
                    
                        # Calculate angle
                        elbow_angle = calculate_angle(shoulder, elbow, wrist)
                        arm_angle = calculate_angle(hip, shoulder, elbow)
                        wrist_angle = calculate_angle(elbow, wrist, right_index)
                        arm_straight_angle = calculate_angle(left_shoulder, shoulder, elbow)

                        angle = "Elbow Angle: " + str(int(elbow_angle)) + " Arm Angle: " + str(int(arm_angle)) + " Wrist Angle: " + str(int(wrist_angle))
                        status = "Salute is wrong"

                        # Determine suggestion based on arm_angle
                        if 85 <= arm_angle <= 120 and 20 <= elbow_angle <= 30 and 168 <= wrist_angle <= 180:
                            suggestion = "Perfect position"
                            status = "Salute is Correct"
                        elif arm_angle < 85:
                            suggestion = "Raise arm slowly [" + str(int(arm_angle)) + "]"
                        elif arm_angle > 120:
                            suggestion = "Lower arm slowly [" + str(int(arm_angle)) + "]"
                        elif elbow_angle < 20:
                            suggestion = "Raise elbow slightly [" + str(int(elbow_angle)) + "]"
                        elif elbow_angle > 30:
                            suggestion = "Lower elbow slightly [" + str(int(elbow_angle)) + "]"
                        elif wrist_angle < 168:
                            suggestion = "Raise wrist slightly [" + str(int(wrist_angle)) + "]"
                        elif wrist_angle > 180:
                            suggestion = "Lower wrist slightly [" + str(int(wrist_angle)) + "]"

                        # Display suggestion on the screen
                        color = (0, 255, 0) if "Correct" in status else (0, 0, 255)
                        cv2.putText(frame, f'Suggestion: {suggestion}', (10, 430),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                        # Visualize angles
                        # cv2.putText(centered_image, str(int(elbow_angle)), 
                        #             tuple(np.multiply(elbow, [roi_width, roi_height]).astype(int)), 
                        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2, cv2.LINE_AA)
                        # cv2.putText(centered_image, str(int(arm_angle)),
                        #             tuple(np.multiply(shoulder, [roi_width, roi_height]).astype(int)),
                        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2, cv2.LINE_AA)
                        # cv2.putText(centered_image, str(int(wrist_angle)),
                        #             tuple(np.multiply(wrist, [roi_width, roi_height]).astype(int)),
                        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2, cv2.LINE_AA)
                        # cv2.putText(centered_image, str(int(arm_straight_angle)),
                        #             tuple(np.multiply(left_shoulder, [roi_width, roi_height]).astype(int)),
                        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2, cv2.LINE_AA)

                        mp_drawing.draw_landmarks(
                            frame,
                            results.pose_landmarks,
                            mp_pose.POSE_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1)
                        )

                    ret, buffer = cv2.imencode('.jpg', frame)
                    frame_bytes = buffer.tobytes()

                    yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()
