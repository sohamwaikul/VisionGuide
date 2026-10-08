from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from ultralytics import YOLO

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Switched to 'yolov8s.pt' for significantly higher accuracy
# (It will download automatically on the first run)
model = YOLO("yolov8s.pt")

@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if frame is None:
        return {"detections": []}

    # Run YOLO detection with higher confidence threshold (0.50)
    results = model(frame, conf=0.50)
    detections = []
    frame_height, frame_width, _ = frame.shape

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = float(box.conf[0])
            label = model.names[int(box.cls[0])]

            # Calculate relative horizontal position (-1.0 = left, 0.0 = center, 1.0 = right)
            center_x = (x1 + x2) / 2
            position = (center_x - (frame_width / 2)) / (frame_width / 2)

            detections.append({
                "object": label,
                "confidence": round(confidence, 2),
                "position": round(position, 2)
            })

    # Sort detections by confidence so the most accurate object is prioritized
    detections.sort(key=lambda x: x["confidence"], reverse=True)

    return {"detections": detections}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
