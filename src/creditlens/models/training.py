
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
