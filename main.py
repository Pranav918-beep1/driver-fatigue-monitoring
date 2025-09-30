import cv2
import mediapipe as mp
import numpy as np
import csv
import os

# --- MediaPipe setup ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=1,
                                  refine_landmarks=True,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

# --- Utility functions ---
def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def eye_aspect_ratio(landmarks, eye_indices):
    # EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
    p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in eye_indices]
    return (euclidean_distance(p2, p6) + euclidean_distance(p3, p5)) / (2.0 * euclidean_distance(p1, p4))

def mouth_aspect_ratio(landmarks, mouth_indices):
    # MAR = (||p2 - p8|| + ||p3 - p7|| + ||p4 - p6||) / (2 * ||p1 - p5||)
    p1, p2, p3, p4, p5, p6, p7, p8 = [landmarks[i] for i in mouth_indices]
    return (euclidean_distance(p2, p8) + euclidean_distance(p3, p7) + euclidean_distance(p4, p6)) / (2.0 * euclidean_distance(p1, p5))

# Eye + mouth landmark indices (from MediaPipe FaceMesh)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [61, 81, 13, 311, 291, 308, 402, 14]

# Thresholds (tune these values based on testing)
EAR_THRESH = 0.25
MAR_THRESH = 0.6

def process_video(video_path, output_csv, output_video):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # VideoWriter for saving output
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

    frame_num = 0
    blink_count, yawn_count = 0, 0

    # Event logging
    os.makedirs("results", exist_ok=True)
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["frame", "event", "value"])

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_num += 1

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            status_text = "Awake"

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                h, w, _ = frame.shape
                coords = [(int(l.x * w), int(l.y * h)) for l in landmarks]

                # EAR (both eyes average)
                left_ear = eye_aspect_ratio(coords, LEFT_EYE)
                right_ear = eye_aspect_ratio(coords, RIGHT_EYE)
                ear = (left_ear + right_ear) / 2.0

                # MAR
                mar = mouth_aspect_ratio(coords, MOUTH)

                if ear < EAR_THRESH:
                    blink_count += 1
                    status_text = "Drowsy"
                    writer.writerow([frame_num, "blink", round(ear, 3)])

                if mar > MAR_THRESH:
                    yawn_count += 1
                    status_text = "Drowsy"
                    writer.writerow([frame_num, "yawn", round(mar, 3)])

                # Draw eye + mouth landmarks
                for idx in LEFT_EYE + RIGHT_EYE + MOUTH:
                    cv2.circle(frame, coords[idx], 2, (0, 255, 0), -1)

            # Overlay text
            cv2.putText(frame, f"Status: {status_text}", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f"Blinks: {blink_count}  Yawns: {yawn_count}", (30, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            # Write frame to output video
            out.write(frame)

            # Optional live preview
            cv2.imshow("Fatigue Monitor", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# --- Main Runner ---
if __name__ == "__main__":
    video_folder = "videos"
    os.makedirs("results", exist_ok=True)

    for filename in os.listdir(video_folder):
        if filename.endswith(".mp4") or filename.endswith(".avi"):
            video_path = os.path.join(video_folder, filename)
            csv_path = os.path.join("results", f"events_{filename.split('.')[0]}.csv")
            output_video = os.path.join("results", f"{filename.split('.')[0]}_output.mp4")
            print(f"Processing {video_path}...")
            process_video(video_path, csv_path, output_video)

    print("✅ All videos processed. Results saved in 'results/' folder.")
