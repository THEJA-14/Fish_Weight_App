# import streamlit as st
# import pickle
# import os
# import sys

# # Add src folder to Python path
# sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# from model import NearestRowModel

# # Paths
# ARTIFACTS_DIR = "artifacts"
# MODEL_PATH = os.path.join(ARTIFACTS_DIR, "nearest_row_model.pkl")
# HEALTH_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "health_classifier.pkl")

# # Load weight prediction model
# try:
#     with open(MODEL_PATH, "rb") as f:
#         model = pickle.load(f)
# except Exception as e:
#     st.error(f"Error loading weight model: {e}")
#     st.stop()

# # Load health classifier
# try:
#     with open(HEALTH_MODEL_PATH, "rb") as f:
#         health_model = pickle.load(f)
# except Exception as e:
#     st.error(f"Error loading health classifier: {e}")
#     st.stop()

# st.title("🐟 Fish Weight Predictor + Health Classifier")

# # Species dropdown
# species_list = list(model.species_dict.keys())
# species = st.selectbox("Select Species", species_list)

# # Numeric inputs
# length1 = st.number_input("Length1 (cm)", min_value=0.0, format="%.2f")
# length2 = st.number_input("Length2 (cm)", min_value=0.0, format="%.2f")
# length3 = st.number_input("Length3 (cm)", min_value=0.0, format="%.2f")
# height = st.number_input("Height (cm)", min_value=0.0, format="%.2f")
# width = st.number_input("Width (cm)", min_value=0.0, format="%.2f")

# if st.button("Predict Weight & Health"):
#     try:
#         # Predict weight
#         weight = model.predict(length1, length2, length3, height, width, species)
#         st.success(f"Predicted Weight: {weight:.2f} grams")

#         # Prepare features for health classifier
#         species_code = species_list.index(species)  # encode species
#         X_new = [[length1, length2, length3, height, width, species_code, weight]]

#         # Predict health
#         health_status = health_model.predict(X_new)[0]
#         st.info(f"Health Status: {health_status}")

#     except Exception as e:
#         st.error(f"Prediction error: {e}")


import streamlit as st
import pickle
import os
import sys
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# Add src folder to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# --- Paths ---
ARTIFACTS_DIR = "artifacts"
MODELS_DIR = "model"
HYBRID_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "hybrid_model.pkl")
HEALTH_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "health_classifier.pkl")
YOLO_MODEL_PATH = os.path.join(MODELS_DIR, "best.pt")

# --- Load Hybrid Weight Model ---
try:
    with open(HYBRID_MODEL_PATH, "rb") as f:
        hybrid_model = pickle.load(f)
except Exception as e:
    st.error(f"❌ Error loading hybrid model: {e}")
    st.stop()

# --- Load Health Classifier ---
try:
    with open(HEALTH_MODEL_PATH, "rb") as f:
        health_model = pickle.load(f)
except Exception as e:
    st.error(f"❌ Error loading health classifier: {e}")
    st.stop()

# --- Load YOLO Model ---
try:
    yolo_model = YOLO(YOLO_MODEL_PATH)
except Exception as e:
    yolo_model = None
    st.warning(f"⚠️ YOLO model not found at {YOLO_MODEL_PATH}. Image input will be disabled.")

# --- Streamlit UI ---
st.title("🐟 Fish Weight Predictor + Health Classifier")

mode = st.radio("Choose Input Mode:", ["Manual Input", "Image Input"])
species_list = list(hybrid_model.species_dict.keys())

# --- MANUAL INPUT ---
# -------------------------
# MANUAL INPUT SECTION
# -------------------------
if mode == "Manual Input":
    species = st.selectbox("Select Fish Species", species_list)
    length1 = st.number_input("Length1 (cm)", min_value=0.0, format="%.2f")
    length2 = st.number_input("Length2 (cm)", min_value=0.0, format="%.2f")
    length3 = st.number_input("Length3 (cm)", min_value=0.0, format="%.2f")
    height = st.number_input("Height (cm)", min_value=0.0, format="%.2f")
    width = st.number_input("Width (cm)", min_value=0.0, format="%.2f")

    if st.button("Predict Weight & Health"):
        try:
            # Predict weight using hybrid model
            weight = hybrid_model.predict(length1, length2, length3, height, width, species)
            st.success(f"Predicted Weight: {weight:.2f} grams")

            # Health classifier (6 features, exclude weight)
            species_code = species_list.index(species)
            X_new = [[length1, length2, length3, height, width, species_code]]
            health_status = health_model.predict(X_new)[0]
            st.info(f"Health Status: {health_status}")

        except Exception as e:
            st.error(f"Prediction error: {e}")

# --- IMAGE INPUT (Camera Capture) ---
elif mode == "Image Input":
    if not yolo_model:
        st.error("YOLO model not loaded — cannot process images.")
        st.stop()

    st.write("📸 Capture a fish image from your camera:")

    img_file_buffer = st.camera_input("Capture Fish Image")
    user_distance = st.number_input("Your Camera Distance (cm, e.g., 50)", min_value=1.0, value=50.0)
    pixel_per_cm = st.number_input("Pixels per cm at capture distance", min_value=0.1, value=5.0, format="%.2f")
    species = st.selectbox("Select Fish Species", species_list)

    predict_button = st.button("Predict Weight & Health", disabled=(img_file_buffer is None))

    if predict_button and img_file_buffer is not None:
        img = Image.open(img_file_buffer)
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        results = yolo_model(np.array(img))
        names = results[0].names
        extracted_data = {}

        for box in results[0].boxes:
            cls_id = int(box.cls[0].item())
            label = names[cls_id]
            x1, y1, x2, y2 = [int(coord) for coord in box.xyxy[0].tolist()]

            # Draw box
            cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_cv, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Convert pixels to cm
            pixel_length = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
            length_cm = pixel_length / pixel_per_cm * (user_distance / 50.0)  # 50 cm as fixed reference
            extracted_data[label] = round(length_cm, 2)

        length1 = extracted_data.get("l1", 0)
        length2 = extracted_data.get("l2", 0)
        length3 = extracted_data.get("l3", 0)
        height = extracted_data.get("w", 0)
        width = 0.35 * height

        st.subheader("📏 Extracted Measurements (cm)")
        st.json({
            "Length1": length1,
            "Length2": length2,
            "Length3": length3,
            "Height": height,
            "Estimated Width": width
        })

        st.subheader("Detected Lengths on Image")
        st.image(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB), use_column_width=True)

        # Predict weight & health
        try:
            weight = hybrid_model.predict(length1, length2, length3, height, width, species)
            st.success(f"Predicted Weight: {weight:.2f} grams")

            species_code = species_list.index(species)
            X_new = [[length1, length2, length3, height, width, species_code]]  # exclude weight
            health_status = health_model.predict(X_new)[0]
            st.info(f"Health Status: {health_status}")

        except Exception as e:
            st.error(f"Prediction error: {e}")