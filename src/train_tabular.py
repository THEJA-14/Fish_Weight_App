import pandas as pd
import os
import pickle
from model import NearestRowModel  # import the class

# Paths
DATA_PATH = "../data/Fish.csv"
ARTIFACTS_DIR = "../artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# Load dataset
data = pd.read_csv(DATA_PATH)

# Create nearest-row model
model = NearestRowModel(data)

# Save the model
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "nearest_row_model.pkl")
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

print("✅ Nearest-row model saved successfully in artifacts/")
