from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
import shutil
import os

app = FastAPI()

model = YOLO("./best.pt")

# Waste category mapping
CATEGORY_MAP = {
    "Battery": "Special Waste",
    "Bulb": "Special Waste",

    "Bottle": "Recyclable",
    "Can": "Recyclable",
    "Carton": "Recyclable",
    "Glass Bottle": "Recyclable",
    "Paper": "Recyclable",
    "Plastic": "Recyclable",
    "Plastic Bottle": "Recyclable",

    "Organic": "Biodegradable",

    "Cup": "Residual Waste"
}

@app.post("/detect")
async def detect(file: UploadFile = File(...)):

    temp_path = "temp.jpg"

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    results = model.predict(
        source=temp_path,
        conf=0.10,
        iou=0.50
    )[0]

    detections = []

    for box in results.boxes:

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])

        label = model.names[cls_id]

        category = CATEGORY_MAP.get(label, "Unknown")

        detections.append({
            "type": category,
            "label": label,
            "confidence": round(conf, 3),
            "box": {
                "x1": round(x1, 2),
                "y1": round(y1, 2),
                "x2": round(x2, 2),
                "y2": round(y2, 2)
            }
        })

    if os.path.exists(temp_path):
        os.remove(temp_path)

    return {
        "total_detected": len(detections),
        "detections": detections
    }