import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import tensorflow as tf
from tensorflow.keras import layers, models

# --------------------------------------------------
# 1️⃣ Paths setup
# --------------------------------------------------
# Base directory (project root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "Fish.csv")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "fish_weight_dl_model.h5")

# Ensure artifacts directory exists
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# --------------------------------------------------
# 2️⃣ Load dataset
# --------------------------------------------------
df = pd.read_csv(DATA_PATH)

X = df.drop("Weight", axis=1)
y = df["Weight"]

# --------------------------------------------------
# 3️⃣ Preprocessing (scaling + encoding)
# --------------------------------------------------
numeric_features = ["Length1", "Length2", "Length3", "Height", "Width"]
categorical_features = ["Species"]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(), categorical_features)
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# --------------------------------------------------
# 4️⃣ Build the Deep Learning model
# --------------------------------------------------
model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)  # Regression output
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# --------------------------------------------------
# 5️⃣ Train the model
# --------------------------------------------------
print("🚀 Training deep learning model...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=16,
    verbose=1
)

# --------------------------------------------------
# 6️⃣ Evaluate performance
# --------------------------------------------------
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ Test MAE: {test_mae:.2f} grams")

# --------------------------------------------------
# 7️⃣ Save the model to artifacts folder
# --------------------------------------------------
model.save(MODEL_PATH)
print(f"✅ Deep Learning model saved successfully at: {MODEL_PATH}")
