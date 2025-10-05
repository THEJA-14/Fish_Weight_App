import pandas as pd
import os
import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------------
# Paths (following your example)
# -------------------------------
DATA_PATH = "../data/Fish.csv"
ARTIFACTS_DIR = "../artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "fish_weight_xgb_model.pkl")

# -------------------------------
# Load dataset
# -------------------------------
data = pd.read_csv(DATA_PATH)

# Encode species
le = LabelEncoder()
data['Species'] = le.fit_transform(data['Species'])

# Features and target
X = data[['Species', 'Length1', 'Length2', 'Length3', 'Height', 'Width']]
y = data['Weight']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------
# Define and train XGBoost model
# -------------------------------
xgb = XGBRegressor(objective='reg:squarederror', random_state=42)

# Hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 1],
    'colsample_bytree': [0.7, 0.8, 1]
}

grid = GridSearchCV(estimator=xgb, param_grid=param_grid,
                    scoring='r2', cv=5, verbose=1, n_jobs=-1)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
print("✅ Best Parameters:", grid.best_params_)

# Evaluate
y_pred = best_model.predict(X_test)
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
print(f"R^2 Score: {r2_score(y_test, y_pred):.2f}")

# -------------------------------
# Save model in artifacts/
# -------------------------------
joblib.dump(best_model, MODEL_PATH)
print(f"✅ XGBoost model saved successfully at {MODEL_PATH}")
