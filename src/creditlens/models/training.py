
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from creditlens.config import RANDOM_STATE
from creditlens.data.preprocessing import (
    build_preprocessor,
    split_dataset,
    load_dataset,
    separate_features_and_target,
)


def train_logistic_regression(X_train, y_train, preprocessor):
    """Train a Logistic Regression classification pipeline."""

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    return model


def prepare_training_data(file_path):
    """Load the dataset and prepare training and testing sets."""

    data = load_dataset(file_path)
    X, y = separate_features_and_target(data)

    X_train, X_test, y_train, y_test = split_dataset(X, y)

    preprocessor = build_preprocessor(X_train)

    return X_train, X_test, y_train, y_test, preprocessor






from sklearn.ensemble import RandomForestClassifier


def train_random_forest(X_train, y_train, preprocessor):
    """Train a Random Forest classification pipeline."""

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=12,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    return model

from pathlib import Path

from creditlens.config import MODEL_DIR


def save_model(model, model_name):
    """Save a trained model pipeline to the models directory."""

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / f"{model_name}.joblib"

    joblib.dump(model, model_path)

    print(f"Model saved: {model_path}")

    return model_path


def load_model(model_name):
    """Load a previously saved model pipeline."""

    model_path = MODEL_DIR / f"{model_name}.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Saved model not found: {model_path}"
        )

    return joblib.load(model_path)
