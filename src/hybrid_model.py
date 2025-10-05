import pickle
import pandas as pd
from model import NearestRowModel  # your existing KNN model
from xgboost import XGBRegressor

class HybridFishWeightModel:
    def __init__(self, knn_model_path, xgb_model_path, dataset_path, threshold=0.05):
        """
        knn_model_path: path to saved NearestRowModel
        xgb_model_path: path to saved XGBoost model
        dataset_path: path to CSV dataset
        threshold: distance ratio threshold to switch to XGBoost
        """
        # Load dataset
        self.data = pd.read_csv(dataset_path)

        # Load KNN model
        with open(knn_model_path, "rb") as f:
            self.knn_model = pickle.load(f)

        # Load XGBoost model
        with open(xgb_model_path, "rb") as f:
            self.xgb_model = pickle.load(f)

        # Keep threshold for switching
        self.threshold = threshold

        # Map species to indices
        self.species_dict = {s: i for i, s in enumerate(self.data['Species'].unique())}

    def predict(self, length1, length2, length3, height, width, species):
        # Prepare input row
        species_code = self.species_dict[species]
        input_row = [length1, length2, length3, height, width, species_code]

        # Predict using nearest-row model
        knn_weight = self.knn_model.predict(length1, length2, length3, height, width, species)

        # Compute minimum distance to dataset
        dataset_features = self.data[['Length1', 'Length2', 'Length3', 'Height', 'Width', 'Species_ID']].values
        distances = ((dataset_features - input_row) ** 2).sum(axis=1) ** 0.5
        min_distance = distances.min()

        # Compute threshold for nearest-row (e.g., relative distance)
        avg_distance = distances.mean()
        if min_distance / avg_distance > self.threshold:
            # Use XGBoost if nearest neighbor is too far
            xgb_weight = self.xgb_model.predict([input_row])[0]
            return xgb_weight
        else:
            # Otherwise use KNN
            return knn_weight
