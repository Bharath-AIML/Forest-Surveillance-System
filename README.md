# 🌲 Forest Surveillance System

AI-based forest surveillance system using **YOLOv8** and **OpenCV** for real-time **human and animal detection** with **anti-poaching alerts**.  
This project aims to assist in wildlife conservation by providing an automated monitoring and alerting mechanism.

---

## 🚀 Project Overview

Poaching and illegal human intrusion remain major threats to wildlife, especially endangered animals such as elephants. Traditional surveillance systems rely heavily on manual monitoring and lack real-time alerting.

This project provides:
- ✅ Automated **image-based poaching detection**
- ✅ **Live CCTV-style human detection** using webcam
- ✅ **YOLOv8 deep learning model** for object detection
- ✅ **Proximity-based alert system**
- ✅ **Text-based alert logging with timestamps**
- ✅ Works for both **day and night scenarios**

---

## 🧠 Features

- Real-time human detection using laptop webcam  
- Human + animal proximity detection from images  
- Automatic alert generation  
- Timestamp-based alert logging in `text.txt`  
- Bounding box visualization on detected objects  
- Lightweight and efficient for local deployment  

---

## 🛠️ Tech Stack

- **Programming Language:** Python 3.9+
- **Deep Learning Model:** YOLOv8 (Ultralytics)
- **Libraries:** OpenCV, NumPy, Torch, Matplotlib
- **Dataset Sources:** Kaggle, Roboflow (Thermal & RGB)

---

## 📂 Project Structure

Forest-Surveillance-System/
├── input_images/ # Test input images
├── Forest_surveillance_system.py # Image-based anti-poaching detection
├── Live_detection.py # Real-time webcam detection
├── requirements.txt # Project dependencies
├── README.md # Project documentation
└── .gitignore

## ⚙️ Installation

Make sure Python 3.9+ is installed. Then install the dependencies:

```bash
pip install -r requirements.txt
```

▶️ How to Run
🔹 1. Image-Based Anti-Poaching Detection
python Forest_surveillance_system.py:


Place your test images inside the input_images/ folder

The system detects humans and animals

If proximity is close → it raises:

ALERT: Possible Poaching!


Output images with bounding boxes are saved in results/

🔹 2. Live CCTV Detection (Webcam)
python Live_detection.py:


Uses your laptop webcam as a live forest CCTV

When a human is detected:

A warning is printed in the terminal

A line is written into text.txt:

WARNING: HUMAN DETECTED at 2025-10-09 17:25:42


Press q or ctrl+c to exit live detection.


📊 Output Examples:

✅ Annotated images with bounding boxes

✅ Terminal alert messages

✅ Text-based alert logs with timestamp


🎯 Applications:

Wildlife protection & conservation

Anti-poaching surveillance

Smart forest monitoring

Night-time animal tracking

AI-based security systems


📚 Learning Outcomes:

Practical implementation of YOLOv8

Real-time object detection using OpenCV

Working with thermal and RGB image concepts

AI for social and environmental good

Alert system design & automation


🔮 Future Enhancements:

Integration of a custom thermal-trained YOLO model for improved night-time accuracy.

Human–elephant proximity alert integration in live CCTV detection.

Deployment on drones and edge devices for large-scale forest monitoring.

Mobile alerts via SMS and Email for instant authority notification.

GPS-based intrusion mapping and real-time location tracking.

Scalability of the system for vast and non-forest areas such as restricted zones, industrial zones, border areas, and high-security locations, where continuous human monitoring is difficult.

Replacement of manual monitoring staff by intelligent AI-based monitoring agents, reducing human workload and operational costs.

Deployment in garbage disposal restricted zones to detect illegal dumping activities and monitor violations 24/7.

Adaptation of this monitoring framework for any application where surveillance is required by retraining the model with domain-specific datasets.


👨‍💻 Author:

Bharath B
B.E Computer Science and Engineering (AI & ML)
Chennai Institute of Technology


⚖️ License:

This project is licensed under the MIT License — free to use for educational and research purposes.


⚠️ Disclaimer:

This project is developed strictly for educational and research purposes only.
It is not intended for real-world military or harmful surveillance use.
