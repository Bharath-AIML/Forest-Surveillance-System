from ultralytics import YOLO
import cv2
import numpy as np
import time
import os
from datetime import datetime

VIDEO_SOURCE = 0                
MODEL_WEIGHTS = "yolov8n.pt"
CONF_THRESH = 0.35
FRAME_WIDTH = 960
OUTPUT_VIDEO = None             
TEXT_ALERT_FILE = "text.txt"    
# -----------------------------

os.makedirs("results", exist_ok=True)

model = YOLO(MODEL_WEIGHTS)

def write_text_alert(message):
    """Append a message line to TEXT_ALERT_FILE (in working dir)."""
    try:
        with open(TEXT_ALERT_FILE, "a") as f:
            f.write(message + "\n")
    except Exception as e:
        print("Failed to write text alert:", e)

def get_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

# Video capture
cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    raise RuntimeError(f"Could not open video source: {VIDEO_SOURCE}")

writer = None
if OUTPUT_VIDEO:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (FRAME_WIDTH, int(FRAME_WIDTH * 9/16)))

print("Starting live detection. Press 'q' to quit.")

human_present_prev = False  # state to avoid repeated writes every frame

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame grab failed — exiting.")
            break

        # Resize for speed
        h, w = frame.shape[:2]
        scale = FRAME_WIDTH / float(w)
        if scale != 1.0:
            frame_resized = cv2.resize(frame, (FRAME_WIDTH, int(h * scale)))
        else:
            frame_resized = frame

        # Run detection
        results = model(frame_resized, conf=CONF_THRESH, imgsz=640)[0]

        humans = []
        elephants = []

        if results.boxes is not None and len(results.boxes) > 0:
            for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
                cls_id = int(cls)
                cls_name = results.names[cls_id]
                x1, y1, x2, y2 = map(int, box.tolist())
                label = f"{cls_name} {conf:.2f}"

                if cls_name == "person":
                    humans.append((x1, y1, x2, y2))
                    color = (0, 200, 0)
                elif cls_name == "elephant":
                    elephants.append((x1, y1, x2, y2))
                    color = (0, 128, 255)
                else:
                    color = (160, 160, 160)

                cv2.rectangle(frame_resized, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame_resized, label, (x1, max(15, y1-6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # If humans appear now but weren't present in previous frame => write alert
        human_present_now = len(humans) > 0
        if human_present_now and not human_present_prev:
            # Compose message with timestamp
            t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            msg = f"WARNING: HUMAN DETECTED at {t}"
            print(msg)
            write_text_alert(msg)

        # Optionally: proximity checks, alerts for human+elephant as before (not required)
        # ... (you can copy previous proximity-check code here)

        # Update prev flag
        human_present_prev = human_present_now

        # Show frame and handle write
        cv2.imshow("Live Anti-Poaching", frame_resized)
        if writer:
            writer.write(frame_resized)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

finally:
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("Stopped.")
