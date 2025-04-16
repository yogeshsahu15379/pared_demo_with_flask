import cv2
import numpy as np
import mediapipe as mp
import threading

mp_pose = mp.solutions.pose

output_frame = None
pose_data = None
frame_text = None  # Added to store frame text
pose_lock = threading.Lock()

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360.0 - angle
        
    return angle 

def update_frame_and_text(new_frame, new_text):
    """Update the frame and frame text from kadamtal_detection."""
    global output_frame, frame_text
    with pose_lock:
        output_frame = new_frame
        frame_text = new_text

def kadamtal_frame_worker():
    global output_frame, pose_data
    # cap= cv2.VideoCapture(0)
    cap = cv2.VideoCapture("rtsp://192.168.1.7:8080/h264_ulaw.sdp", cv2.CAP_FFMPEG)
    frame_count = 0
    process_every_nth_frame = 2

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Frame drop / decode error")
                continue  # instead of break

            frame_count += 1
            if frame_count % process_every_nth_frame != 0:
                continue

            # Remove horizontal flip to avoid mirror effect
            # frame = cv2.flip(frame, 1)
            
            # Crop the frame to center width=300 and height=600
            frame_height, frame_width = frame.shape[:2]
            center_x, center_y = frame_width // 2, frame_height // 2
            crop_width, crop_height = 300, 600
            x1 = max(center_x - crop_width // 2, 0)
            y1 = max(center_y - crop_height // 2, 0)
            x2 = min(center_x + crop_width // 2, frame_width)
            y2 = min(center_y + crop_height // 2, frame_height)
            frame = frame[y1:y2, x1:x2]

            # Use the cropped frame for pose detection
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            results = pose.process(image)  # Pose detection is limited to the cropped region

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            angle = 0
            status = "No pose detected"
            suggestion = ""

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

                angle = "right knee Angle: " + str(int(right_knee_angle)) + " left knee Angle: " + str(int(left_knee_angle)) + " right elbow Angle: " + str(int(right_elbow_angle))
                status = "both leg are grounded."
                leg= "no leg up"


                if right_knee_angle < 150 and left_knee_angle > 170:
                    leg= "right"
                    angle = "right knee Angle: " + str(int(right_knee_angle)) + " left knee Angle: " + str(int(left_knee_angle)) + " right elbow Angle: " + str(int(right_elbow_angle)) + " left elbow Angle: " + str(int(left_elbow_angle)) + "right hip Angle: " + str(int(right_hip_angle)) + " left hip Angle: " + str(int(left_hip_angle)) + " right ankle Angle: " + str(int(right_ankle_angle)) + " left ankle Angle: " + str(int(left_ankle_angle))

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
                    
                    cv2.putText(image, str(int(right_knee_angle)),
                                tuple(np.multiply(right_knee, [300, 600]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA
                                        )
                    
                    cv2.putText(image, str(int(right_hip_angle)),
                                tuple(np.multiply(right_hip, [300, 600]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA
                                        )
                    
                    cv2.putText(image, str(int(right_ankle_angle)),
                                tuple(np.multiply(right_ankle, [300, 600]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA
                                        )

                    if right_hip_angle < 130 and right_ankle_angle < 115 and 55<= right_knee_angle <= 100 and 150 <= right_elbow_angle <= 180:
                        suggestion = "Perfect Right Leg Up Position"
                        status = "right kadam is Correct"
                    else:
                        status = "right kadam is wrong"
                        if right_hip_angle > 130:
                            suggestion = f"rise your right leg [right_hip_angle : {int(right_hip_angle)}]"
                        elif right_ankle_angle > 115:
                            suggestion = f"rise your right foot [right_ankle_angle: {int(right_ankle_angle)}] "
                        elif right_knee_angle < 55:
                            suggestion = f"move your right leg forward [right_knee_angle: {int(right_knee_angle)}]"
                        elif right_knee_angle > 100:
                            suggestion = f"move your right leg backward [right_knee_angle: {int(right_knee_angle)}]"
                        elif right_elbow_angle < 150:
                            suggestion = f"Straighten right elbow slightly [right_elbow_angle: {int(right_elbow_angle)}]"
                        # elif right_wrist_angle < 170:
                        #     suggestion = f"Straighten Right Wrist [right_wrist_angle: {int(right_wrist_angle)}]"
                    
                elif left_knee_angle < 150 and right_knee_angle > 170:
                    leg= "left"
                    angle = "right knee Angle: " + str(int(right_knee_angle)) + " left knee Angle: " + str(int(left_knee_angle)) + " right elbow Angle: " + str(int(right_elbow_angle)) + " left elbow Angle: " + str(int(left_elbow_angle)) + "right hip Angle: " + str(int(right_hip_angle)) + " left hip Angle: " + str(int(left_hip_angle)) + " right ankle Angle: " + str(int(right_ankle_angle)) + " left ankle Angle: " + str(int(left_ankle_angle))

                    cv2.putText(image,str(int(left_ankle_angle)),
                                tuple(np.multiply(left_ankle, [300, 600]).astype(int)),
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
                    cv2.putText(image, str(int(left_knee_angle)),
                                tuple(np.multiply(left_knee, [300, 600]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA
                                        )
                    cv2.putText(image, str(int(left_elbow_angle)),
                                tuple(np.multiply(left_elbow, [300, 600]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA
                                        )
                    
                    cv2.putText(image, str(int(left_hip_angle)),
                                tuple(np.multiply(left_hip, [300, 600]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA
                                        )
                    
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

                else:
                    suggestion = "Invalid state: Both legs are either up or grounded."
                

                # Display suggestion on the screen
                color = (0, 255, 0) if "Correct" in status else (0, 0, 255)
                cv2.putText(image, f'Suggestion: {suggestion}', (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            with pose_lock:
                output_frame = image.copy()
                pose_data = {
                    "angle":angle,
                    "status": status,
                    "suggestion": suggestion
                }
                # Add frame text to the image
                if frame_text:
                    cv2.putText(output_frame, frame_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cap.release()

def get_kadamtal_frame():
    with pose_lock:
        if output_frame is None:
            return None
        _, buffer = cv2.imencode('.jpg', output_frame)
        return buffer.tobytes()

def get_pose_data():
    with pose_lock:
        return pose_data

