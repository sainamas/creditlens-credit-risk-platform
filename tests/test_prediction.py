
import pytest

from creditlens.models.training import load_model
from creditlens.data.preprocessing import (
    load_dataset,
    separate_features_and_target,
)
from creditlens.prediction.predictor import (
    predict_default_probability,
    classify_risk,
)


@pytest.fixture
def customer_and_model():
    data = load_dataset("data/raw/credit_card_default.xls")
    X, y = separate_features_and_target(data)

    model = load_model("random_forest")
    customer = X.iloc[[0]].copy()

    return model, customer


def test_valid_customer_prediction(customer_and_model):
    model, customer = customer_and_model

    probability = predict_default_probability(model, customer)[0]

    assert 0 <= probability <= 1


def test_missing_values_are_rejected(customer_and_model):
    model, customer = customer_and_model

    customer.iloc[0, 0] = float("nan")

    with pytest.raises(ValueError, match="missing values"):
        predict_default_probability(model, customer)


def test_missing_features_are_rejected(customer_and_model):
    model, customer = customer_and_model

    customer = customer.drop(columns=[customer.columns[0]])

    with pytest.raises(ValueError, match="Missing required"):
        predict_default_probability(model, customer)


def test_risk_classification():
    assert classify_risk(0.05) == "Low"
    assert classify_risk(0.18) == "Moderate"
    assert classify_risk(0.35) == "High"
    assert classify_risk(0.82) == "Very High"


def test_invalid_probability_is_rejected():
    with pytest.raises(ValueError):
        classify_risk(1.5)
