from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import shutil
import os
import cv2
import torch
import numpy as np
import base64
import asyncio
import json

app = FastAPI()

# Allow all origins (adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO("./best.pt")

CATEGORY_MAP = {
    "Battery":        "Special Waste",
    "Bulb":           "Special Waste",

    "Bottle":         "Recyclable",
    "Can":            "Recyclable",
    "Carton":         "Recyclable",
    "Glass Bottle":   "Recyclable",
    "Paper":          "Recyclable",
    "Plastic":        "Recyclable",
    "Plastic Bottle": "Recyclable",

    "Organic":        "Biodegradable",

    "Cup":            "Residual / Non-Recyclable",
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─── Helper: run YOLO on a decoded image ──────────────────────────────────────
def run_inference(image: np.ndarray) -> dict:
    h, w = image.shape[:2]
    results = model.predict(source=image, conf=0.10, iou=0.50, verbose=False)[0]
    detections = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        category = CATEGORY_MAP.get(label, "Unknown")

        detections.append({
            "type":       category,
            "label":      label,
            "confidence": round(conf, 3),
            "box":        {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        })

    return {
        "device_used":    DEVICE,
        "total_detected": len(detections),
        "detections":     detections,
        "image_width":    w,   # actual pixels YOLO saw — client uses for scaling
        "image_height":   h,
    }


# ─── Existing REST endpoint (unchanged behaviour) ─────────────────────────────
@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    temp_path = f"temp_{os.getpid()}.jpg"
    output_path = "output.jpg"

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    image = cv2.imread(temp_path)
    result = run_inference(image)

    # Draw boxes on output image
    for d in result["detections"]:
        b = d["box"]
        cv2.rectangle(image, (b["x1"], b["y1"]), (b["x2"], b["y2"]), (0, 255, 0), 2)
        text = f"{d['label']} ({d['confidence']})"
        cv2.putText(image, text, (b["x1"], b["y1"] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imwrite(output_path, image)

    if os.path.exists(temp_path):
        os.remove(temp_path)

    result["output_image"] = "output.jpg"
    return result


# ─── WebSocket endpoint for live / real-time classification ───────────────────
# Protocol:
#   Client → Server : JSON { "frame": "<base64-encoded JPEG>" }
#   Server → Client : JSON { device_used, total_detected, detections[], frame_ms }
@app.websocket("/detect/live")
async def detect_live(websocket: WebSocket):
    await websocket.accept()
    print(f"[WS] Client connected — device: {DEVICE}")

    try:
        while True:
            # Receive JSON message with base64 frame
            try:
                raw = await asyncio.wait_for(websocket.receive_text(), timeout=10.0)
            except asyncio.TimeoutError:
                # Send a keep-alive ping
                await websocket.send_text(json.dumps({"ping": True}))
                continue

            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"error": "Invalid JSON"}))
                continue

            if "frame" not in payload:
                await websocket.send_text(json.dumps({"error": "Missing 'frame' key"}))
                continue

            # Decode base64 → numpy image
            try:
                b64 = payload["frame"]
                # Strip data URI prefix if present (data:image/jpeg;base64,...)
                if "," in b64:
                    b64 = b64.split(",", 1)[1]
                img_bytes = base64.b64decode(b64)
                np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
                image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                if image is None:
                    await websocket.send_text(json.dumps({"error": "Could not decode image"}))
                    continue
            except Exception as e:
                await websocket.send_text(json.dumps({"error": f"Decode error: {str(e)}"}))
                continue

            # Run inference (offload to thread so the event loop doesn't block)
            import time
            t0 = time.perf_counter()
            result = await asyncio.get_event_loop().run_in_executor(
                None, run_inference, image
            )
            frame_ms = round((time.perf_counter() - t0) * 1000)

            result["frame_ms"] = frame_ms
            await websocket.send_text(json.dumps(result))

    except WebSocketDisconnect:
        print("[WS] Client disconnected")
    except Exception as e:
        print(f"[WS] Error: {e}")
        try:
            await websocket.send_text(json.dumps({"error": str(e)}))
            await websocket.close()
        except Exception:
            pass