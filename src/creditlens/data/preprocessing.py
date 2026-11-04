
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from creditlens.config import TARGET_COLUMN, RANDOM_STATE


def load_dataset(file_path):
    """Load the credit card default dataset."""
    return pd.read_excel(file_path, header=1, engine="xlrd")


def separate_features_and_target(data):
    """Separate input features from the target variable."""

    if TARGET_COLUMN not in data.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' was not found."
        )

    X = data.drop(columns=[TARGET_COLUMN]).copy()
    y = data[TARGET_COLUMN].copy()

    # Customer ID is an identifier, not a predictive feature.
    X = X.drop(columns=["ID"], errors="ignore")

    return X, y


def build_preprocessor(X):
    """Build preprocessing pipelines for numerical and categorical data."""

    categorical_features = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numerical_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", numerical_pipeline, numerical_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )

    return preprocessor


def split_dataset(X, y):
    """Split the dataset into training and testing sets."""

    return train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )
