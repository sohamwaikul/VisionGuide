from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from ultralytics import YOLO

app = FastAPI(title="VisionGuide AI Backend", version="2.3.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

model = YOLO("yolov8m.pt")

FOCAL_RATIO = 1.25  # focal length = 1.25 x frame width (800 px at 640 px wide)
KNOWN_HEIGHTS = {
    "person": 1.7, "cell phone": 0.15, "chair": 0.9, "bottle": 0.25, "laptop": 0.35,
    "cup": 0.12, "car": 1.5, "bicycle": 1.0, "motorcycle": 1.1, "bus": 3.0,
    "truck": 2.8, "dog": 0.5, "cat": 0.3, "default": 0.5,
}
ASSISTIVE_CLASSES = {
    "person", "cell phone", "chair", "bottle", "cup", "laptop", "car", "bicycle",
    "motorcycle", "bus", "truck", "traffic light", "stop sign", "cat", "dog",
}

# FIX 1: one low threshold (0.25) let weak, wrong guesses through. Each class now
# needs its own minimum confidence; small items that YOLO often confuses are stricter.
DEFAULT_CONF = 0.50
CLASS_CONF = {"person": 0.35, "car": 0.45, "bus": 0.45, "cell phone": 0.60,
              "cup": 0.60, "bottle": 0.55, "laptop": 0.55}
MIN_BOX_AREA = 0.004  # FIX 2: ignore tiny boxes (under 0.4% of the frame) - mostly noise
SESSIONS = {}         # FIX 3: last frame's boxes per client, to confirm objects across frames


def direction_of(p: float) -> str:
    return ("Far Left" if p < -0.6 else "Left" if p < -0.2 else
            "Center" if p <= 0.2 else "Right" if p <= 0.6 else "Far Right")


def distance_of(label: str, box_h: float, frame_w: int) -> float:
    if box_h <= 0:
        return 5.0
    h = KNOWN_HEIGHTS.get(label, KNOWN_HEIGHTS["default"])
    return round(float(min(h * FOCAL_RATIO * frame_w / box_h, 20.0)), 2)


def iou(a, b) -> float:
    w = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def brighten_if_dark(frame):
    """FIX 4: the old contrast/sharpen filters distort images and cause wrong labels,
    so only lift very dark frames and leave everything else untouched."""
    mean = float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())
    if mean >= 60:
        return frame
    gamma = max(0.5, mean / 60.0)
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(frame, lut)


@app.post("/detect")
def detect_objects(
    file: UploadFile = File(...),
    filter_assistive: bool = Query(default=False),
    session: str = Query(default="default"),
):
    try:
        frame = cv2.imdecode(np.frombuffer(file.file.read(), np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image frame provided")

        frame = brighten_if_dark(frame)
        fh, fw = frame.shape[:2]

        # low base conf, then filter per class; agnostic NMS stops one object getting two labels
        res = model(frame, conf=0.25, iou=0.5, imgsz=640, max_det=30,
                    agnostic_nms=True, verbose=False)[0]

        prev = SESSIONS.get(session, [])
        current, detections = [], []

        for box in res.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            label = model.names[int(box.cls[0])]
            bb = (x1, y1, x2, y2)

            if filter_assistive and label not in ASSISTIVE_CLASSES:
                continue
            if conf < CLASS_CONF.get(label, DEFAULT_CONF):
                continue
            if (x2 - x1) * (y2 - y1) < MIN_BOX_AREA * fw * fh:
                continue

            current.append((label, bb))
            # keep it only if very confident, or seen in the previous frame too
            seen_before = any(l == label and iou(bb, b) > 0.3 for l, b in prev)
            if not (conf >= 0.80 or seen_before):
                continue

            pan = ((x1 + x2) / 2 - fw / 2) / (fw / 2)
            prox = ((y1 + y2) / 2) / fh
            dist = distance_of(label, y2 - y1, fw)
            status = ("DANGER" if dist < 1.2 or prox > 0.75 else
                      "WARNING" if dist < 2.5 or prox > 0.50 else "SAFE")

            detections.append({
                "object": label, "confidence": round(conf, 2),
                "bbox": [round(x1), round(y1), round(x2), round(y2)],
                "position": round(pan, 2), "panning": round(pan, 2),
                "proximity": round(prox, 2), "direction": direction_of(pan),
                "estimated_distance_m": dist, "safety_status": status,
            })

        SESSIONS[session] = current
        if len(SESSIONS) > 50:
            SESSIONS.pop(next(iter(SESSIONS)))

        detections.sort(key=lambda d: (d["estimated_distance_m"], -d["confidence"]))
        hazard = None
        if detections:
            t = detections[0]
            hazard = {
                "object": t["object"], "direction": t["direction"],
                "distance_m": t["estimated_distance_m"], "status": t["safety_status"],
                "alert_speech": f"Caution: {t['object']} {t['direction']} at {t['estimated_distance_m']} meters",
            }

        return {"status": "success", "frame_size": [fw, fh], "primary_hazard": hazard,
                "total_detected": len(detections), "detections": detections}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
