import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import time
from collections import deque

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# CORRECT Landmark indices for MediaPipe FaceMesh
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
MOUTH_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 320, 307, 375, 291, 409, 270, 269, 267]
MOUTH_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13]

class AccurateFatigueDetector:
    def __init__(self):
        # Tuned thresholds - these often work better
        self.EAR_THRESHOLD = 0.21      # Lower = more sensitive to eye closure
        self.MAR_THRESHOLD = 0.65      # Higher = more sensitive to mouth opening
        
        # State management
        self.blink_count = 0
        self.yawn_count = 0
        self.eye_closed_frames = 0
        self.mouth_open_frames = 0
        self.eye_closed = False
        self.mouth_open = False
        
        # Cooldown to avoid multiple detections
        self.blink_cooldown = 0
        self.yawn_cooldown = 0
        self.cooldown_frames = 15
        
        # Smoothing
        self.ear_history = deque(maxlen=3)
        self.mar_history = deque(maxlen=3)
        
    def euclidean_distance(self, point1, point2):
        return np.linalg.norm(np.array(point1) - np.array(point2))
    
    def eye_aspect_ratio(self, landmarks):
        """More accurate EAR calculation using 6 points per eye"""
        # Left Eye EAR - using 6 key points
        left_vertical1 = self.euclidean_distance(landmarks[159], landmarks[145])  # top-bottom
        left_vertical2 = self.euclidean_distance(landmarks[158], landmarks[153])  # top-bottom
        left_horizontal = self.euclidean_distance(landmarks[33], landmarks[133])  # left-right
        left_ear = (left_vertical1 + left_vertical2) / (2.0 * left_horizontal)
        
        # Right Eye EAR
        right_vertical1 = self.euclidean_distance(landmarks[386], landmarks[374])
        right_vertical2 = self.euclidean_distance(landmarks[385], landmarks[380])
        right_horizontal = self.euclidean_distance(landmarks[362], landmarks[263])
        right_ear = (right_vertical1 + right_vertical2) / (2.0 * right_horizontal)
        
        # Return average EAR
        return (left_ear + right_ear) / 2.0
    
    def mouth_aspect_ratio(self, landmarks):
        """More accurate MAR calculation"""
        # Outer mouth points for better accuracy
        vertical1 = self.euclidean_distance(landmarks[13], landmarks[14])    # inner top-bottom
        vertical2 = self.euclidean_distance(landmarks[78], landmarks[308])   # left corner vertical
        vertical3 = self.euclidean_distance(landmarks[80], landmarks[310])   # right corner vertical
        
        horizontal = self.euclidean_distance(landmarks[61], landmarks[291])  # mouth width
        
        mar = (vertical1 + vertical2 + vertical3) / (3.0 * horizontal)
        return mar
    
    def smooth_value(self, history, new_value):
        """Apply temporal smoothing"""
        history.append(new_value)
        return np.mean(history)
    
    def detect_events(self, landmarks):
        """Improved event detection with state management"""
        events = []
        
        # Calculate EAR and MAR with smoothing
        ear = self.eye_aspect_ratio(landmarks)
        smoothed_ear = self.smooth_value(self.ear_history, ear)
        
        mar = self.mouth_aspect_ratio(landmarks)
        smoothed_mar = self.smooth_value(self.mar_history, mar)
        
        # Debug prints (remove after testing)
        # print(f"EAR: {smoothed_ear:.3f}, MAR: {smoothed_mar:.3f}")
        
        # Blink detection with state machine
        if smoothed_ear < self.EAR_THRESHOLD:
            if not self.eye_closed:
                self.eye_closed = True
                self.eye_closed_frames = 0
            else:
                self.eye_closed_frames += 1
        else:
            if self.eye_closed:
                # Only register as blink if closed for 2-10 frames
                if 2 <= self.eye_closed_frames <= 10 and self.blink_cooldown == 0:
                    self.blink_count += 1
                    events.append("blink")
                    self.blink_cooldown = self.cooldown_frames
                self.eye_closed = False
        
        # Yawn detection with state machine
        if smoothed_mar > self.MAR_THRESHOLD:
            if not self.mouth_open:
                self.mouth_open = True
                self.mouth_open_frames = 0
            else:
                self.mouth_open_frames += 1
        else:
            if self.mouth_open:
                # Only register as yawn if open for 10+ frames
                if self.mouth_open_frames >= 10 and self.yawn_cooldown == 0:
                    self.yawn_count += 1
                    events.append("yawn")
                    self.yawn_cooldown = self.cooldown_frames
                self.mouth_open = False
        
        # Update cooldowns
        if self.blink_cooldown > 0:
            self.blink_cooldown -= 1
        if self.yawn_cooldown > 0:
            self.yawn_cooldown -= 1
        
        return smoothed_ear, smoothed_mar, events

def draw_enhanced_landmarks(frame, landmarks_coords, ear, mar, blink_count, yawn_count, events):
    """Draw comprehensive visualization"""
    h, w, _ = frame.shape
    
    # Draw eye landmarks with different colors
    for idx in LEFT_EYE:
        cv2.circle(frame, landmarks_coords[idx], 2, (0, 255, 255), -1)  # Yellow
    for idx in RIGHT_EYE:
        cv2.circle(frame, landmarks_coords[idx], 2, (255, 255, 0), -1)  # Cyan
    
    # Draw mouth landmarks
    for idx in MOUTH_OUTER:
        cv2.circle(frame, landmarks_coords[idx], 2, (0, 255, 0), -1)    # Green
    for idx in MOUTH_INNER:
        cv2.circle(frame, landmarks_coords[idx], 2, (255, 0, 255), -1)  # Magenta
    
    # Draw eye contours
    left_eye_points = np.array([landmarks_coords[i] for i in LEFT_EYE[:6]], np.int32)
    right_eye_points = np.array([landmarks_coords[i] for i in RIGHT_EYE[:6]], np.int32)
    cv2.polylines(frame, [left_eye_points], True, (0, 255, 0), 1)
    cv2.polylines(frame, [right_eye_points], True, (0, 255, 0), 1)
    
    # Draw mouth contours
    mouth_outer_points = np.array([landmarks_coords[i] for i in MOUTH_OUTER[:8]], np.int32)
    cv2.polylines(frame, [mouth_outer_points], True, (255, 0, 0), 1)
    
    # Determine status with better logic
    if events:
        status = "DROWSY"
        status_color = (0, 0, 255)  # Red
    elif ear < 0.25:  # Approaching drowsiness
        status = "ALERT (Low EAR)"
        status_color = (0, 165, 255)  # Orange
    elif mar > 0.6:   # Approaching yawning
        status = "ALERT (High MAR)"
        status_color = (0, 165, 255)  # Orange
    else:
        status = "ALERT"
        status_color = (0, 255, 0)  # Green
    
    # Display comprehensive information
    y_offset = 30
    line_height = 25
    
    info_lines = [
        f"Status: {status}",
        f"Blinks: {blink_count} | Yawns: {yawn_count}",
        f"EAR: {ear:.3f} (thresh: {detector.EAR_THRESHOLD:.3f})",
        f"MAR: {mar:.3f} (thresh: {detector.MAR_THRESHOLD:.3f})",
        f"Events: {', '.join(events) if events else 'None'}",
        f"Eye State: {'CLOSED' if detector.eye_closed else 'OPEN'}",
        f"Mouth State: {'OPEN' if detector.mouth_open else 'CLOSED'}"
    ]
    
    for i, line in enumerate(info_lines):
        cv2.putText(frame, line, (10, y_offset + i * line_height),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
    
    # Add threshold adjustment instructions
    cv2.putText(frame, "Press 'E'/e to adjust EAR threshold", (w - 300, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(frame, "Press 'M'/m to adjust MAR threshold", (w - 300, 50),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    return frame

def process_video_with_debug(video_path, output_csv, output_video):
    """Process video with real-time threshold adjustment"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))
    
    global detector
    detector = AccurateFatigueDetector()
    
    frame_count = 0
    start_time = time.time()
    
    with open(output_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['frame', 'timestamp', 'event', 'ear', 'mar', 'blink_count', 'yawn_count'])
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            current_time = time.time() - start_time
            
            # Convert to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)
            
            events = []
            ear, mar = 0, 0
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                h, w, _ = frame.shape
                landmarks_coords = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
                
                ear, mar, events = detector.detect_events(landmarks)
                
                # Draw enhanced visualization
                frame = draw_enhanced_landmarks(frame, landmarks_coords, ear, mar, 
                                              detector.blink_count, detector.yawn_count, events)
            
            # Write events to CSV
            for event in events:
                csv_writer.writerow([
                    frame_count, round(current_time, 2), event,
                    round(ear, 3), round(mar, 3),
                    detector.blink_count, detector.yawn_count
                ])
            
            # Write frame
            out.write(frame)
            
            # Display with real-time threshold adjustment
            cv2.imshow('Fatigue Detection (Press Q to quit)', frame)
            
            # Real-time threshold adjustment
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('e'):  # Decrease EAR threshold (more sensitive)
                detector.EAR_THRESHOLD = max(0.15, detector.EAR_THRESHOLD - 0.01)
                print(f"EAR threshold decreased to: {detector.EAR_THRESHOLD:.3f}")
            elif key == ord('E'):  # Increase EAR threshold (less sensitive)
                detector.EAR_THRESHOLD = min(0.35, detector.EAR_THRESHOLD + 0.01)
                print(f"EAR threshold increased to: {detector.EAR_THRESHOLD:.3f}")
            elif key == ord('m'):  # Decrease MAR threshold (less sensitive)
                detector.MAR_THRESHOLD = max(0.4, detector.MAR_THRESHOLD - 0.01)
                print(f"MAR threshold decreased to: {detector.MAR_THRESHOLD:.3f}")
            elif key == ord('M'):  # Increase MAR threshold (more sensitive)
                detector.MAR_THRESHOLD = min(0.9, detector.MAR_THRESHOLD + 0.01)
                print(f"MAR threshold increased to: {detector.MAR_THRESHOLD:.3f}")
            elif key == ord('r'):  # Reset counts
                detector.blink_count = 0
                detector.yawn_count = 0
                print("Counts reset!")
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"\nProcessing complete!")
    print(f"Final counts: {detector.blink_count} blinks, {detector.yawn_count} yawns")
    print(f"Final thresholds: EAR={detector.EAR_THRESHOLD:.3f}, MAR={detector.MAR_THRESHOLD:.3f}")

def main():
    """Main function"""
    video_folder = "videos"
    output_folder = "outputs"
    results_folder = "results"
    
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(results_folder, exist_ok=True)
    
    print("🚀 Starting Fatigue Detection with Real-time Adjustment")
    print("Controls:")
    print("  E/e - Adjust EAR threshold (eye sensitivity)")
    print("  M/m - Adjust MAR threshold (mouth sensitivity)")
    print("  r   - Reset counters")
    print("  q   - Quit")
    print("\nAdjust thresholds until you see correct detections!")
    
    for filename in os.listdir(video_folder):
        if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            video_path = os.path.join(video_folder, filename)
            base_name = os.path.splitext(filename)[0]
            
            output_csv = os.path.join(results_folder, f"events_{base_name}.csv")
            output_video = os.path.join(output_folder, f"{base_name}_output.mp4")
            
            print(f"\n🎬 Processing: {filename}")
            process_video_with_debug(video_path, output_csv, output_video)
    
    print("\n✅ All videos processed!")

# Global detector for real-time adjustment
detector = None

if __name__ == "__main__":
    main()