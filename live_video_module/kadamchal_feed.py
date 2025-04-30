import cv2
import mediapipe as mp
import numpy as np


mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

frame_skip = 2
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

def kadamtal_generate_frames():
    global frame_count
    # Lazy Import to prevent circular import
    from models.drill import DRILL_CAMERA_URL_MAP, DrillType
    cap = cv2.VideoCapture(DRILL_CAMERA_URL_MAP.get(DrillType.KADAMTAL))   

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
                roi_width, roi_height = 300, 600
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

                    angle = "right knee Angle: " + str(int(right_knee_angle)) + " left knee Angle: " + str(int(left_knee_angle)) + " right elbow Angle: " + str(int(right_elbow_angle))
                    status = "both leg are grounded."
                    leg= "no leg up"

                    if left_hip_angle < 130 and left_ankle_angle < 115 and 55<= left_knee_angle <= 100 and 150 <= left_elbow_angle <= 180:
                        suggestion = "Perfect left Leg Up Position"
                        status = "left kadam is Correct"
                    else:
                        status = "left kadam is wrong"
                        if left_hip_angle > 130:
                            suggestion = f"rise your left leg [left_hip_angle: {int(left_hip_angle)}]"
                        elif left_ankle_angle > 115:
                            suggestion = f"rise your left foot [left_ankle_angle :{int(left_ankle_angle)}]"
                        elif left_knee_angle < 55:
                            suggestion = f"move your left leg forward [left_knee_angle : {int(left_knee_angle)}]"
                        elif left_knee_angle > 100:
                            suggestion = f"move your left leg backward [left_knee_angle: {int(left_knee_angle)}]"
                        elif left_elbow_angle < 150:
                            suggestion = f"Straighten left elbow slightly [left_elbow_angle: {int(left_elbow_angle)}]"
                        # elif left_wrist_angle < 170:
                        #     suggestion = f"Left wrist should be straight  [left_wrist_angle :{int(left_wrist_angle)}]"
                        color = (0, 255, 0) if "Correct" in status else (0, 0, 255)
                        cv2.putText(frame, f'Suggestion: {suggestion}', (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    mp_drawing.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                    )

                ret, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()

                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()
