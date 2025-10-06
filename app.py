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


# --- Add import at the top ---


import streamlit as st
import pickle
import os
import sys
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
from src.protein_recommendation import recommend_feed  # Import protein module

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

# ✅ Define mode here to fix NameError
mode = st.radio("Choose Input Mode:", ["Manual Input", "Image Input"])

species_list = list(hybrid_model.species_dict.keys())

# --- MANUAL INPUT ---
if mode == "Manual Input":
    species = st.selectbox("Select Fish Species", species_list, key="manual_species")
    length1 = st.number_input("Length1 (cm)", min_value=0.0, format="%.2f", key="manual_length1")
    length2 = st.number_input("Length2 (cm)", min_value=0.0, format="%.2f", key="manual_length2")
    length3 = st.number_input("Length3 (cm)", min_value=0.0, format="%.2f", key="manual_length3")
    height = st.number_input("Height (cm)", min_value=0.0, format="%.2f", key="manual_height")
    width = st.number_input("Width (cm)", min_value=0.0, format="%.2f", key="manual_width")
    temp = st.number_input("Water Temperature (°C)", min_value=0.0, value=20.0, format="%.1f", key="manual_temp")

    if st.button("Predict Weight, Health & Feed Recommendation", key="manual_predict"):
        try:
            # --- Weight Prediction ---
            weight = hybrid_model.predict(length1, length2, length3, height, width, species)
            st.success(f"Predicted Weight: {weight:.2f} grams")

            # --- Health Prediction ---
            species_code = species_list.index(species)
            X_new = [[length1, length2, length3, height, width, species_code]]
            health_status = health_model.predict(X_new)[0]
            st.info(f"Health Status: {health_status}")

            # --- Feed/Protein Recommendation ---
            recommendation = recommend_feed(
                species=species,
                current_weight=weight,
                length3=length3,
                health_status=health_status,
                fcr=1.5,
                temp=temp,
                growth_days=10
            )
            if isinstance(recommendation, dict):
                st.subheader("🍽 Feed & Protein Recommendation")
                st.json(recommendation)
            else:
                st.info(recommendation)

        except Exception as e:
            st.error(f"Prediction error: {e}")

# --- IMAGE INPUT (Folder-based Selection) ---
elif mode == "Image Input":
    if not yolo_model:
        st.error("YOLO model not loaded — cannot process images.")
        st.stop()

    # Ensure YOLO is in evaluation mode
    try:
        yolo_model.model.eval()
    except Exception:
        pass

    st.write("📂 Select a sample fish image from the local folder:")

    # Folder path for sample images
    IMAGE_FOLDER = "sample_images"
    os.makedirs(IMAGE_FOLDER, exist_ok=True)
    image_files = [f for f in os.listdir(IMAGE_FOLDER)
                   if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    if not image_files:
        st.error("No images found in the 'sample_images' folder. Please add some fish images there.")
        st.stop()

    # Let user select an image
    selected_image = st.selectbox("Choose an image", image_files)
    img_path = os.path.join(IMAGE_FOLDER, selected_image)
    img = Image.open(img_path)
    st.image(img, caption=f"Selected: {selected_image}", use_column_width=True)

    # User parameters
    user_distance = st.number_input(
        "Camera Distance (cm, e.g., 50)", min_value=1.0, value=50.0, key="image_distance"
    )
    pixel_per_cm = st.number_input(
        "Pixels per cm at capture distance", min_value=0.1, value=5.0, format="%.2f", key="image_ppcm"
    )
    species = st.selectbox("Select Fish Species", species_list, key="image_species")
    temp = st.number_input(
        "Water Temperature (°C)", min_value=0.0, value=20.0, format="%.1f", key="image_temp"
    )

    predict_button = st.button("Predict Weight, Health & Feed Recommendation", key="image_predict")

    if predict_button:
        # Convert to NumPy array (RGB) - do NOT convert to BGR
        img_array = np.array(img)

        # Run YOLO detection
        results = yolo_model.predict(source=img_array, imgsz=640, conf=0.1)  # lowered conf for safety
        names = results[0].names
        extracted_data = {}

        # DEBUG: show all predicted labels
        for box in results[0].boxes:
            cls_id = int(box.cls[0].item())
            conf = box.conf[0].item()
            label = names[cls_id]
            st.write(f"Detected: {label}, Confidence: {conf:.2f}")

        # Draw bounding boxes & extract measurements
        img_cv = img_array.copy()
        for box in results[0].boxes:
            cls_id = int(box.cls[0].item())
            label = names[cls_id]

            if label not in ["l1", "l2", "l3", "w"]:
                continue

            x1, y1, x2, y2 = [int(coord) for coord in box.xyxy[0].tolist()]
            cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_cv, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Euclidean distance in pixels → cm conversion
            pixel_length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
            length_cm = pixel_length / pixel_per_cm * (user_distance / 50.0)
            extracted_data[label] = round(length_cm, 2)

        # Extract values (0 if not detected)
        length1 = extracted_data.get("l1", 0)
        length2 = extracted_data.get("l2", 0)
        length3 = extracted_data.get("l3", 0)
        height = extracted_data.get("w", 0)
        width = 0.35 * height  # estimated

        # Show extracted lengths
        st.subheader("📏 Extracted Measurements (cm)")
        st.json({
            "Length1": length1,
            "Length2": length2,
            "Length3": length3,
            "Height": height,
            "Estimated Width": width
        })

        # Display annotated image
        st.subheader("Detected Lines on Image")
        st.image(img_cv, use_column_width=True)

        try:
            # --- Weight Prediction ---
            weight = hybrid_model.predict(length1, length2, length3, height, width, species)
            st.success(f"Predicted Weight: {weight:.2f} grams")

            # --- Health Prediction ---
            species_code = species_list.index(species)
            X_new = [[length1, length2, length3, height, width, species_code]]
            health_status = health_model.predict(X_new)[0]
            st.info(f"Health Status: {health_status}")

            # --- Feed/Protein Recommendation ---
            recommendation = recommend_feed(
                species=species,
                current_weight=weight,
                length3=length3,
                health_status=health_status,
                fcr=1.5,
                temp=temp,
                growth_days=10
            )

            if isinstance(recommendation, dict):
                st.subheader("🍽 Feed & Protein Recommendation")
                st.json(recommendation)
            else:
                st.info(recommendation)

        except Exception as e:
            st.error(f"Prediction error: {e}")
