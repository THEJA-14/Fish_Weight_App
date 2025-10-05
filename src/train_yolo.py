import os
from ultralytics import YOLO

# Automatically find dataset.yaml correctly
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # goes up one level from src/
DATA_YAML = os.path.join(BASE_DIR, 'data', 'data.yaml')

model = YOLO('yolov8n.pt')

model.train(
    data=DATA_YAML,   # now correct path
    epochs=50,
    imgsz=640,
    batch=16,
    name='fish_length_detector'
)
