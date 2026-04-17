from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
import shutil
import os
import cv2
import torch  

app = FastAPI()

model = YOLO("./best.pt")

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
    output_path = "output.jpg"

    # Save uploaded image
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Read image using OpenCV
    image = cv2.imread(temp_path)

    # YOLO prediction
    results = model.predict(
        source=image,
        conf=0.10,
        iou=0.50
    )[0]

    detections = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])

        label = model.names[cls_id]
        category = CATEGORY_MAP.get(label, "Unknown")

        # 🔹 Draw bounding box (OpenCV)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        text = f"{label} ({round(conf,2)})"
        cv2.putText(image, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        detections.append({
            "type": category,
            "label": label,
            "confidence": round(conf, 3),
            "box": {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2
            }
        })

    # Save processed image
    cv2.imwrite(output_path, image)

    # OPTIONAL torch usage (check GPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Cleanup
    if os.path.exists(temp_path):
        os.remove(temp_path)

    return {
        "device_used": device,
        "total_detected": len(detections),
        "detections": detections,
        "output_image": "output.jpg"
    }