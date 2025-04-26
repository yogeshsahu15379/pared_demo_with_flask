import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

frame_skip = 3
frame_count = 0

def slow_chal_generate_frames():
    global frame_count
    cap = cv2.VideoCapture("rtsp://admin:admin@123@192.168.0.13:554/1/2?transmode=unicast&profile=va")   

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
