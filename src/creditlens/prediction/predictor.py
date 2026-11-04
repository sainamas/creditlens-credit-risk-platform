
import numpy as np
import pandas as pd


def predict_default_probability(model, customer_data):
    """
    Predict the probability of default for customer data.
    """

    if not isinstance(customer_data, pd.DataFrame):
        raise TypeError(
            "Customer data must be provided as a pandas DataFrame."
        )

    if customer_data.empty:
        raise ValueError(
            "Customer data cannot be empty."
        )

    expected_features = model.feature_names_in_

    missing_features = [
        feature
        for feature in expected_features
        if feature not in customer_data.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing required customer features: {missing_features}"
        )

    customer_data = customer_data.loc[:, expected_features].copy()

    if customer_data.isnull().values.any():
        raise ValueError(
            "Customer data contains missing values."
        )

    try:
        customer_data = customer_data.astype(float)
    except (ValueError, TypeError):
        raise ValueError(
            "All customer features must contain numeric values."
        )

    if not np.isfinite(customer_data.to_numpy()).all():
        raise ValueError(
            "Customer data contains invalid infinite values."
        )

    probabilities = model.predict_proba(customer_data)[:, 1]

    return probabilities


def classify_risk(probability):
    """
    Classify a predicted default probability
    using CreditLens illustrative risk thresholds.
    """

    if not np.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError(
            "Probability must be a finite number between 0 and 1."
        )

    if probability < 0.10:
        return "Low"

    elif probability < 0.25:
        return "Moderate"

    elif probability < 0.50:
        return "High"

    else:
        return "Very High"
