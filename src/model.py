import numpy as np

class NearestRowModel:
    def __init__(self, dataframe):
        self.data = dataframe.copy()
        # Map species name to IDs internally (optional)
        self.species_dict = {s: i for i, s in enumerate(self.data["Species"].unique())}

    def predict(self, length1, length2, length3, height, width, species_name):
        if species_name not in self.species_dict:
            raise ValueError(f"Species '{species_name}' not found in dataset")

        # Filter only rows of that species and reset index
        sp_data = self.data[self.data["Species"] == species_name].reset_index(drop=True)

        input_values = np.array([length1, length2, length3, height, width])

        # Compute Euclidean distance for all rows
        distances = sp_data[["Length1", "Length2", "Length3", "Height", "Width"]].apply(
            lambda row: np.linalg.norm(row.values - input_values), axis=1
        )

        # Get nearest row using correct indexing
        nearest_row = sp_data.iloc[distances.idxmin()]
        return nearest_row["Weight"]
