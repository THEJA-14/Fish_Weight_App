import os
import pickle
from hybrid_model import HybridFishWeightModel

# Base directory = project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
DATASET_PATH = os.path.join(BASE_DIR, "data", "Fish.csv")

KN_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "nearest_row_model.pkl")
XGB_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "fish_weight_xgb_model.pkl")
HYBRID_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "hybrid_model.pkl")

# Create hybrid model
hybrid_model = HybridFishWeightModel(
    knn_model_path=KN_MODEL_PATH,
    xgb_model_path=XGB_MODEL_PATH,
    dataset_path=DATASET_PATH
)

# Save hybrid model
with open(HYBRID_MODEL_PATH, "wb") as f:
    pickle.dump(hybrid_model, f)

print(f"✅ Hybrid model saved successfully at {HYBRID_MODEL_PATH}")
