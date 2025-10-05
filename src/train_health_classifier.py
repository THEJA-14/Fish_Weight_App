import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

# Paths
DATA_PATH = "../data/Fish.csv"
ARTIFACTS_DIR = "../artifacts"

# Ensure artifacts folder exists
if not os.path.exists(ARTIFACTS_DIR):
    os.makedirs(ARTIFACTS_DIR)

# Load dataset
df = pd.read_csv(DATA_PATH)  # columns: Length1, Length2, Length3, Height, Width, Species, Weight

# Species-specific constants for expected weight
species_constants = {
    "Bream": (0.0094, 3.2545),
    "Roach": (0.012, 3.1),
    "Perch": (0.01, 3.2),
    "Pike": (0.007, 3.3),
    "Smelt": (0.008, 3.1),
    "Whitefish": (0.009, 3.25),
    "Parkki": (0.010, 3.15)
}

threshold = 0.20  # ✅ 20% deviation for health classification

# Label health status
def label_health(row):
    if row['Species'] not in species_constants:
        return "Unknown"
    
    a, b = species_constants[row['Species']]
    expected_weight = a * (row['Length3'] ** b)
    lower = expected_weight * (1 - threshold)
    upper = expected_weight * (1 + threshold)
    
    if row['Weight'] < lower:
        return "Malnourished"
    elif row['Weight'] > upper:
        return "Overweight"
    else:
        return "Healthy"

df['Health'] = df.apply(label_health, axis=1)

# Encode species
df['Species_Code'] = df['Species'].astype('category').cat.codes

# ✅ Features: only size + species (no Weight to avoid leakage!)
X = df[['Length1', 'Length2', 'Length3', 'Height', 'Width', 'Species_Code']]
y = df['Health']

# Train RandomForest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)

# Save the classifier correctly inside artifacts folder
classifier_path = os.path.join(ARTIFACTS_DIR, "health_classifier.pkl")
with open(classifier_path, "wb") as f:
    pickle.dump(clf, f)

print(f"✅ Health classifier trained and saved successfully at {classifier_path}!")
