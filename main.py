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
import time
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from typing import List, Dict, Any, Optional

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("./best.pt")
model.to(DEVICE)


INFER_EXECUTOR = ThreadPoolExecutor(max_workers=1)

CATEGORY_MAP = {
    # Biodegradable Waste
    "Biodegradable": "Biodegradable",
    
    #  Recyclable Waste
    "Bottle cap": "Recyclable",
    "Can": "Recyclable",
    "Card": "Recyclable",
    "Cup": "Recyclable",
    "Glass bottle": "Recyclable",
    "Paper": "Recyclable",
    "Plastic": "Recyclable",
    "Plastic bottle": "Recyclable",
    "Detergent bottle": "Recyclable",
    "Food container": "Recyclable",
    "Bottle Cap": "Recyclable",
    
    #  Residual Waste
    "Sachet": "Residual",
    "Straw": "Residual",
    "Gloves": "Residual",
    "Plastic bag": "Residual",
    "Wrapper": "Residual",
    
    #  Special Waste
    "Battery": "Special Waste",
    "Bulb": "Special Waste",
    
    #  E-Waste
    "CD": "E-Waste",
}


PLASTIC_TYPE_MAP = {
    "PET": "PET (#1)",
    "HDPE": "HDPE (#2)",
    "LDPE": "LDPE (#4)",
    "PP": "PP (#5)",
    "Polypropylene": "PP (#5)",
    "Polycarbonate": "#7 Other",
    "Multi-layer": "#7 Other",
}


class DetectionSmoother:
    """Merges detections across the last `window` frames."""
    IOU_THRESH = 0.35
    MIN_HITS = 2
    WINDOW = 4

    def __init__(self):
        self.history: List[List[Dict]] = []

    @staticmethod
    def _iou(a: Dict, b: Dict) -> float:
        ax1, ay1, ax2, ay2 = a["box"]["x1"], a["box"]["y1"], a["box"]["x2"], a["box"]["y2"]
        bx1, by1, bx2, by2 = b["box"]["x1"], b["box"]["y1"], b["box"]["x2"], b["box"]["y2"]
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        if inter == 0:
            return 0.0
        ua = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
        return inter / ua if ua > 0 else 0.0

    def update(self, new_detections: List[Dict]) -> List[Dict]:
        self.history.append(new_detections)
        if len(self.history) > self.WINDOW:
            self.history.pop(0)

        by_label: Dict[str, List[Dict]] = defaultdict(list)
        for frame in self.history:
            for det in frame:
                by_label[det["label"]].append(det)

        stable = []
        for label, dets in by_label.items():
            clusters: List[List[Dict]] = []
            for det in dets:
                placed = False
                for cluster in clusters:
                    if self._iou(det, cluster[0]) >= self.IOU_THRESH:
                        cluster.append(det)
                        placed = True
                        break
                if not placed:
                    clusters.append([det])

            for cluster in clusters:
                if len(cluster) < self.MIN_HITS:
                    continue

                avg_x1 = int(np.mean([d["box"]["x1"] for d in cluster]))
                avg_y1 = int(np.mean([d["box"]["y1"] for d in cluster]))
                avg_x2 = int(np.mean([d["box"]["x2"] for d in cluster]))
                avg_y2 = int(np.mean([d["box"]["y2"] for d in cluster]))
                avg_conf = round(float(np.mean([d["confidence"] for d in cluster])), 3)

                # Ensure coordinates are within image bounds
                avg_x1 = max(0, avg_x1)
                avg_y1 = max(0, avg_y1)
                avg_x2 = max(avg_x1 + 1, avg_x2)
                avg_y2 = max(avg_y1 + 1, avg_y2)

                stable.append({
                    "type": cluster[0]["type"],
                    "plastic_type": cluster[0].get("plastic_type", "Unknown"),
                    "label": label,
                    "confidence": avg_conf,
                    "box": {"x1": avg_x1, "y1": avg_y1, "x2": avg_x2, "y2": avg_y2},
                })

        return stable

# ─── Helper: Get plastic type from label ──────────────────────────────────────
def get_plastic_type(label: str) -> str:
    """Map label to plastic type based on your classification table."""
    plastic_mapping = {
        "Plastic bottle": "PET (#1)",
        "Detergent bottle": "HDPE (#2)",
        "Plastic bag": "LDPE (#4)",
        "Wrapper": "LDPE (#4)",
        "Food container": "PP (#5) Polypropylene",
        "Bottle cap": "PP (#5) Polypropylene",
        "Straw": "PP (#5) Polypropylene",
        "Sachet": "#7 Other (Multi-layer Plastic)",
        "CD": "#7 Other (Polycarbonate)",
    }
    return plastic_mapping.get(label, "Unknown")

# ─── Helper: YOLO Inference with FIXED bounding boxes ──────────────────────────────────────────────────
def run_inference(image: np.ndarray) -> Dict[str, Any]:
    """Run YOLO inference on the image with properly scaled bounding boxes."""
    original_h, original_w = image.shape[:2]
    
    # Calculate scaling factors for letterboxing (preserving aspect ratio)
    target_size = 640
    scale = min(target_size / original_w, target_size / original_h)
    
    # Calculate new dimensions
    new_w = int(original_w * scale)
    new_h = int(original_h * scale)
    
    # Calculate padding
    pad_w = (target_size - new_w) // 2
    pad_h = (target_size - new_h) // 2
    
    # Resize image
    resized = cv2.resize(image, (new_w, new_h))
    
    # Create letterboxed image (pad with 114,114,114 as per YOLO default)
    letterboxed = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    letterboxed[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
    
    # Run inference
    results = model.predict(
        source=letterboxed,
        conf=0.30,
        iou=0.45,
        device=DEVICE,
        verbose=False,
    )[0]

    detections = []
    for box in results.boxes:
        # Get coordinates from letterboxed image (640x640)
        x1_lb, y1_lb, x2_lb, y2_lb = map(int, box.xyxy[0])
        
        # Remove padding to get coordinates on the resized image
        x1_resized = x1_lb - pad_w
        y1_resized = y1_lb - pad_h
        x2_resized = x2_lb - pad_w
        y2_resized = y2_lb - pad_h
        
        # Scale back to original image size
        x1 = max(0, int(x1_resized / scale))
        y1 = max(0, int(y1_resized / scale))
        x2 = min(original_w, int(x2_resized / scale))
        y2 = min(original_h, int(y2_resized / scale))
        
        # Ensure coordinates are valid (x1 < x2, y1 < y2)
        if x1 >= x2:
            x2 = x1 + 1
        if y1 >= y2:
            y2 = y1 + 1
        
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        
        detections.append({
            "type": CATEGORY_MAP.get(label, "Unknown"),
            "plastic_type": get_plastic_type(label),
            "label": label,
            "confidence": round(conf, 3),
            "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        })

    return {
        "device_used": DEVICE,
        "total_detected": len(detections),
        "detections": detections,
        "image_width": original_w,
        "image_height": original_h,
        "processing_scale": scale,
        "original_dimensions": {"width": original_w, "height": original_h}
    }

# ─── REST Endpoint ────────────────────────────────────────────────────────────
@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    """Detect objects in a single uploaded image with properly aligned bounding boxes."""
    temp_path = f"temp_{os.getpid()}_{int(time.time())}.jpg"
    try:
        content = await file.read()
        with open(temp_path, "wb") as buffer:
            buffer.write(content)
        
        image = cv2.imread(temp_path)
        if image is None:
            return {"error": "Could not read image file"}
        
        # Get original dimensions
        original_h, original_w = image.shape[:2]
        
        result = run_inference(image)
        
        # Add visualization guide for debugging
        result["bounding_box_info"] = {
            "note": "Boxes are properly scaled to original image dimensions",
            "coordinate_system": f"x: 0-{original_w}, y: 0-{original_h}"
        }
        
        return result
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

# ─── WebSocket for real-time detection with FIXED boxes ────────────────────────────────────────
@app.websocket("/detect/live")
async def detect_live(websocket: WebSocket):
    """WebSocket endpoint for real-time video stream detection with accurate bounding boxes."""
    await websocket.accept()
    print(f"[WS] Connected | Device: {DEVICE}")

    smoother = DetectionSmoother()
    latest_frame: Optional[Dict] = None
    inference_running = False
    loop = asyncio.get_event_loop()
    frame_counter = 0
    fps_stats = []
    last_fps_time = time.time()

    async def inference_worker():
        """Background task that processes frames and sends results."""
        nonlocal latest_frame, inference_running, frame_counter, fps_stats, last_fps_time
        inference_running = True

        while True:
            if latest_frame is None:
                await asyncio.sleep(0.01)
                continue

            frame_data = latest_frame
            latest_frame = None
            image = frame_data["img"]

            t0 = time.perf_counter()

            try:
                raw_result = await loop.run_in_executor(
                    INFER_EXECUTOR, run_inference, image
                )
                
                # Calculate FPS
                frame_time = time.perf_counter() - t0
                fps_stats.append(frame_time)
                if len(fps_stats) > 30:
                    fps_stats.pop(0)
                avg_fps = len(fps_stats) / sum(fps_stats) if fps_stats else 0
                
                frame_counter += 1
                
                # Send FPS info every second
                current_time = time.time()
                if current_time - last_fps_time >= 1.0:
                    print(f"[FPS] {avg_fps:.1f} fps")
                    last_fps_time = current_time
                
            except Exception as exc:
                print(f"[WS] Inference error: {exc}")
                await asyncio.sleep(0.05)
                continue

            frame_ms = round((time.perf_counter() - t0) * 1000)

            # Apply temporal smoothing
            stable_detections = smoother.update(raw_result["detections"])

            result = {
                **raw_result,
                "detections": stable_detections,
                "total_detected": len(stable_detections),
                "frame_ms": frame_ms,
                "fps": round(avg_fps, 1),
                "frame_number": frame_counter,
                "bounding_box_info": {
                    "note": "Boxes are properly scaled to original frame dimensions",
                    "coordinate_system": f"x: 0-{raw_result['image_width']}, y: 0-{raw_result['image_height']}"
                }
            }

            try:
                await websocket.send_text(json.dumps(result))
            except Exception:
                break

            await asyncio.sleep(0)

        inference_running = False

    # Start the inference worker
    worker_task = asyncio.create_task(inference_worker())

    try:
        while True:
            raw = await websocket.receive_text()

            try:
                payload = json.loads(raw)
                b64 = payload.get("frame", "")
                if not b64:
                    continue

                # Handle data URL format
                if "," in b64:
                    b64 = b64.split(",")[1]

                img_bytes = base64.b64decode(b64)
                np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
                image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                if image is None:
                    continue

                # Store only the latest frame
                latest_frame = {"img": image, "t": time.time()}

            except json.JSONDecodeError as e:
                print(f"[WS] JSON decode error: {e}")
                continue
            except Exception as e:
                print(f"[WS] Frame processing error: {e}")
                continue

    except WebSocketDisconnect:
        print("[WS] Client disconnected")
    except Exception as e:
        print(f"[WS] Unexpected error: {e}")
    finally:
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass
        try:
            await websocket.close()
        except Exception:
            pass
        print("[WS] Cleaned up")

# ─── Health check endpoint ────────────────────────────────────────────────────
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "device": DEVICE,
        "model_loaded": model is not None
    }

# ─── Get model info ───────────────────────────────────────────────────────────
@app.get("/model/info")
async def model_info():
    """Get information about the loaded model."""
    return {
        "device": DEVICE,
        "classes": model.names,
        "num_classes": len(model.names)
    }

# ─── Main entry point ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    print(f"🚀 Server starting on http://0.0.0.0:8000")
    print(f"📱 Device: {DEVICE}")
    print(f"🎯 Model classes: {len(model.names)}")
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )