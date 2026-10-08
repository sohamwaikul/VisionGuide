# VisionGuide AI 👁️🔊

> **An assistive accessibility application that helps visually impaired individuals navigate safely using real-time computer vision and 3D spatial audio.**

---

## ❓ Problem Statement

Over 285 million visually impaired people worldwide face daily mobility challenges when navigating indoor and outdoor environments:
* **Traditional tools** like white canes detect ground-level obstacles on physical contact, but cannot identify specific elevated objects, doors, or hazards ahead.
* **Existing digital assistants** rely on cloud processing that introduces high latency, require internet connections, or overwhelm users with long, text-heavy spoken descriptions.

---

## 💡 What Is VisionGuide AI?

**VisionGuide AI** transforms any standard smartphone or camera device into a real-time visual assistant:
1. **Real-Time Visual Processing:** Captures camera frames and identifies obstacles (phones, chairs, people, vehicles, doors) using an optimized object detection pipeline.
2. **3D Positional Audio Feedback:** Translates object coordinates into 3D spatial audio using the Web Audio API. A sound playing in the left ear indicates an obstacle on the left, giving natural auditory awareness without visual clutter.
3. **Low-Latency Design:** Operates locally with minimal delay to provide timely warning signals in dynamic environments.

---

## ⭐ Key Features

* **🚧 Real-Time Object Detection:** Scans surroundings to detect key indoor and outdoor navigation hazards.
* **🎧 3D Spatial Audio Guidance:** Uses stereo panning algorithms to simulate realistic directionality (-1.0 Far Left to +1.0 Far Right).
* **⚡ Audio Unlock & Fallback:** Includes interactive audio permissions handling compatible with modern mobile and web browsers.
* **🌐 Zero Special Hardware Required:** Runs in standard web browsers using standard device camera inputs.

---

## 🛠️ Tech Stack

* **Backend Engine:** Python, FastAPI, Uvicorn
* **Computer Vision Model:** Ultralytics YOLOv8 (`yolov8s.pt` / `yolov8n.pt`), OpenCV, NumPy
* **Frontend Interface:** HTML5, JavaScript (ES6+), Web Audio API (StereoPannerNode)

---

## 🚀 Getting Started

### Prerequisites
* **Python 3.10+**
* A web browser with camera and web audio support (Chrome, Edge, Safari)

---

### Step 1: Clone the Repository
```bash
git clone [https://github.com/sohamwaikul/VisionGuide.git](https://github.com/sohamwaikul/VisionGuide.git)
cd VisionGuide
