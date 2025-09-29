import streamlit as st
import pickle
import os
import sys

# Add src folder to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from model import NearestRowModel

# Paths
ARTIFACTS_DIR = "artifacts"
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "nearest_row_model.pkl")
HEALTH_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "health_classifier.pkl")

# Load weight prediction model
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    st.error(f"Error loading weight model: {e}")
    st.stop()

# Load health classifier
try:
    with open(HEALTH_MODEL_PATH, "rb") as f:
        health_model = pickle.load(f)
except Exception as e:
    st.error(f"Error loading health classifier: {e}")
    st.stop()

st.title("🐟 Fish Weight Predictor + Health Classifier")

# Species dropdown
species_list = list(model.species_dict.keys())
species = st.selectbox("Select Species", species_list)

# Numeric inputs
length1 = st.number_input("Length1 (cm)", min_value=0.0, format="%.2f")
length2 = st.number_input("Length2 (cm)", min_value=0.0, format="%.2f")
length3 = st.number_input("Length3 (cm)", min_value=0.0, format="%.2f")
height = st.number_input("Height (cm)", min_value=0.0, format="%.2f")
width = st.number_input("Width (cm)", min_value=0.0, format="%.2f")

if st.button("Predict Weight & Health"):
    try:
        # Predict weight
        weight = model.predict(length1, length2, length3, height, width, species)
        st.success(f"Predicted Weight: {weight:.2f} grams")

        # Prepare features for health classifier
        species_code = species_list.index(species)  # encode species
        X_new = [[length1, length2, length3, height, width, species_code, weight]]

        # Predict health
        health_status = health_model.predict(X_new)[0]
        st.info(f"Health Status: {health_status}")

    except Exception as e:
        st.error(f"Prediction error: {e}")
