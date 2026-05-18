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

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Load Model ───────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model  = YOLO("./best.pt")
model.to(DEVICE)

# ── Single-thread executor: only ONE inference runs at a time.
# Multiple concurrent WS clients share this pool; no GPU contention.
INFER_EXECUTOR = ThreadPoolExecutor(max_workers=1)

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

# ─── Temporal smoother ───────────────────────────────────────────────────────
# Keeps a short history of detections and merges them so boxes don't
# flash on/off when the model misses a single frame.
class DetectionSmoother:
    """
    Merges detections across the last `window` frames.
    A detection is kept if it appeared in at least `min_hits` of those frames.
    Boxes are averaged over matching detections (same label, overlapping IoU).
    """
    IOU_THRESH  = 0.35   # boxes with IoU > this are considered the "same" object
    MIN_HITS    = 2      # must appear in this many of the last N frames to be shown
    WINDOW      = 4      # how many frames to look back

    def __init__(self):
        self.history: list[list[dict]] = []   # list of per-frame detection lists

    @staticmethod
    def _iou(a: dict, b: dict) -> float:
        ax1, ay1, ax2, ay2 = a["box"]["x1"], a["box"]["y1"], a["box"]["x2"], a["box"]["y2"]
        bx1, by1, bx2, by2 = b["box"]["x1"], b["box"]["y1"], b["box"]["x2"], b["box"]["y2"]
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih   = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter    = iw * ih
        if inter == 0:
            return 0.0
        ua = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
        return inter / ua if ua > 0 else 0.0

    def update(self, new_detections: list[dict]) -> list[dict]:
        self.history.append(new_detections)
        if len(self.history) > self.WINDOW:
            self.history.pop(0)

        # Flatten all detections, grouped by label
        from collections import defaultdict
        by_label: dict[str, list[dict]] = defaultdict(list)
        for frame in self.history:
            for det in frame:
                by_label[det["label"]].append(det)

        stable = []
        for label, dets in by_label.items():
            # Cluster overlapping boxes together
            clusters: list[list[dict]] = []
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
                    continue   # not stable enough yet

                # Average the box coordinates and confidence
                avg_x1   = int(np.mean([d["box"]["x1"] for d in cluster]))
                avg_y1   = int(np.mean([d["box"]["y1"] for d in cluster]))
                avg_x2   = int(np.mean([d["box"]["x2"] for d in cluster]))
                avg_y2   = int(np.mean([d["box"]["y2"] for d in cluster]))
                avg_conf = round(float(np.mean([d["confidence"] for d in cluster])), 3)

                stable.append({
                    "type":       cluster[0]["type"],
                    "label":      label,
                    "confidence": avg_conf,
                    "box": {"x1": avg_x1, "y1": avg_y1, "x2": avg_x2, "y2": avg_y2},
                })

        return stable


# ─── Helper: YOLO Inference ──────────────────────────────────────────────────
def run_inference(image: np.ndarray) -> dict:
    image = cv2.resize(image, (640, 640))
    h, w  = image.shape[:2]

    results = model.predict(
        source=image,
        conf=0.30,      # slightly higher threshold → fewer false-positive flashes
        iou=0.45,
        device=DEVICE,
        verbose=False,
    )[0]

    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf   = float(box.conf[0])
        cls_id = int(box.cls[0])
        label  = model.names[cls_id]

        detections.append({
            "type":       CATEGORY_MAP.get(label, "Unknown"),
            "label":      label,
            "confidence": round(conf, 3),
            "box":        {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        })

    return {
        "device_used":   DEVICE,
        "total_detected": len(detections),
        "detections":    detections,
        "image_width":   w,
        "image_height":  h,
    }


# ─── REST Endpoint ────────────────────────────────────────────────────────────
@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    temp_path = f"temp_{os.getpid()}.jpg"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    image  = cv2.imread(temp_path)
    result = run_inference(image)
    os.remove(temp_path)
    return result


# ─── WebSocket (REAL-TIME) ────────────────────────────────────────────────────
@app.websocket("/detect/live")
async def detect_live(websocket: WebSocket):
    await websocket.accept()
    print(f"[WS] Connected | Device: {DEVICE}")

    smoother = DetectionSmoother()

    # ── Per-connection state ──────────────────────────────────────────────────
    # latest_frame: the most recent decoded image waiting to be inferred.
    # We overwrite it whenever a new frame arrives so we always process
    # the freshest image, not a stale one from the queue.
    latest_frame: dict | None = None          # {"img": np.ndarray, "t": float}
    inference_running          = False
    loop                       = asyncio.get_event_loop()

    async def inference_worker():
        """
        Runs in a tight loop, grabbing the latest frame and running inference.
        Sends result back over the socket. Yields to the event loop between
        iterations so receive_text() can run concurrently.
        """
        nonlocal latest_frame, inference_running
        inference_running = True

        while True:
            # Wait until a frame is available
            if latest_frame is None:
                await asyncio.sleep(0.02)
                continue

            # Grab and clear the slot
            frame_data  = latest_frame
            latest_frame = None
            image        = frame_data["img"]

            t0 = time.perf_counter()

            # Offload blocking YOLO call to our single-thread executor
            try:
                raw_result = await loop.run_in_executor(
                    INFER_EXECUTOR, run_inference, image
                )
            except Exception as exc:
                print(f"[WS] Inference error: {exc}")
                await asyncio.sleep(0.05)
                continue

            frame_ms = round((time.perf_counter() - t0) * 1000)

            # Apply temporal smoothing
            stable_detections = smoother.update(raw_result["detections"])

            result = {
                **raw_result,
                "detections":    stable_detections,
                "total_detected": len(stable_detections),
                "frame_ms":      frame_ms,
            }

            try:
                await websocket.send_text(json.dumps(result))
            except Exception:
                break   # socket closed

            # Small yield so the receive loop gets CPU time
            await asyncio.sleep(0)

        inference_running = False

    # Start the worker as a background task
    worker_task = asyncio.create_task(inference_worker())

    try:
        while True:
            raw = await websocket.receive_text()

            # ── Decode as fast as possible; drop bad frames silently ─────────
            try:
                payload = json.loads(raw)
                b64     = payload.get("frame", "")
                if not b64:
                    continue

                if "," in b64:
                    b64 = b64.split(",")[1]

                img_bytes = base64.b64decode(b64)
                np_arr    = np.frombuffer(img_bytes, dtype=np.uint8)
                image     = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                if image is None:
                    continue

                # ── Overwrite latest_frame (drop any unprocessed old frame) ──
                # This is the key: we never queue frames, we always use the
                # most recent one. The inference worker will pick it up.
                latest_frame = {"img": image, "t": time.time()}

            except Exception as e:
                print(f"[WS] Decode error: {e}")
                continue

    except WebSocketDisconnect:
        print("[WS] Disconnected")

    except Exception as e:
        print(f"[WS] Error: {e}")

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