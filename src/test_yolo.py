import os
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np

# --- Paths ---
# Get absolute path to root folder
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Model path
MODEL_PATH = os.path.join(ROOT_DIR, "model", "best.pt")

# Sample images folder
IMAGE_FOLDER = os.path.join(ROOT_DIR, "sample_images")

# Load YOLO model
model = YOLO(MODEL_PATH)

# List all image files in the sample_images folder
image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

if not image_files:
    print(f"No images found in {IMAGE_FOLDER}")
    exit()

# For testing, take the first image (or you can loop through all)
img_path = os.path.join(IMAGE_FOLDER, image_files[0])
img = Image.open(img_path)
img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

# Run detection
results = model.predict(source=img_cv, imgsz=640, conf=0.25)

# Draw boxes and show
for box in results[0].boxes:
    cls_id = int(box.cls[0].item())
    label = results[0].names[cls_id]
    x1, y1, x2, y2 = [int(c) for c in box.xyxy[0].tolist()]
    cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img_cv, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# Show image
cv2.imshow("Detection", img_cv)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Print detected labels and boxes
for box in results[0].boxes:
    print("Label:", results[0].names[int(box.cls[0].item())], "Box:", box.xyxy)
