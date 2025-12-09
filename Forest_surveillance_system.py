from ultralytics import YOLO
import cv2
import os
import numpy as np

model = YOLO("yolov8n.pt")

INPUT_DIR = "input_images"
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PROXIMITY_THRESHOLD = 0.2

def get_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def is_close(box1, box2, img_shape):
    c1 = get_center(box1)
    c2 = get_center(box2)
    dist = np.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)
    diag = np.sqrt(img_shape[0]**2 + img_shape[1]**2)
    return dist < PROXIMITY_THRESHOLD * diag

#  Loop over all images in input folder
for file in os.listdir(INPUT_DIR):
    if not file.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    img_path = os.path.join(INPUT_DIR, file)
    results = model(img_path, conf=0.5)

    img = cv2.imread(img_path)
    humans, animals = [], []    

    for box, cls in zip(results[0].boxes.xyxy, results[0].boxes.cls):
        cls_name = results[0].names[int(cls)]
        x1, y1, x2, y2 = map(int, box)

        if cls_name == "person":
            humans.append((x1, y1, x2, y2))
            color = (0, 255, 0)
        elif cls_name in ["dog", "cat", "bird", "sheep", "cow", "horse", "elephant", "bear", "zebra", "giraffe"]:
            animals.append((x1, y1, x2, y2))
            color = (255, 0, 0)
        else:
            continue

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, cls_name, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Check proximity
    alert_flag = False
    for h in humans:
        for a in animals:
            if is_close(h, a, img.shape):
                alert_flag = True
                cv2.putText(img, "ALERT: Possible Poaching!", (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                break

    # Save output
    out_path = os.path.join(OUTPUT_DIR, file)
    cv2.imwrite(out_path, img)
    print(f"Processed {file} → Alert: {alert_flag}")
