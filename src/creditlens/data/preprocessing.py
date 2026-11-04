
import pandas as pd

from creditlens.config import TARGET_COLUMN


def load_dataset(file_path):
    """Load the credit card default dataset from an Excel file."""
    return pd.read_excel(file_path, header=1, engine="xlrd")

def separate_features_and_target(data):
    """Separate input features (X) from the target variable (y)."""

    if TARGET_COLUMN not in data.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' was not found in the dataset."
        )

    X = data.drop(columns=[TARGET_COLUMN])
    y = data[TARGET_COLUMN]

    return X, y
