
import pytest

from creditlens.models.training import load_model, prepare_training_data
from creditlens.explainability.shap_explainer import (
    create_shap_explainer,
    explain_customer,
    get_global_feature_importance,
    get_customer_explanation,
    create_customer_shap_chart,
)


@pytest.fixture(scope="module")
def shap_test_data():
    X_train, X_test, y_train, y_test, preprocessor = (
        prepare_training_data("data/raw/credit_card_default.xls")
    )

    model = load_model("random_forest")

    background = X_train.sample(n=100, random_state=42)
    customers = X_test.iloc[:50].copy()

    explainer = create_shap_explainer(model, background)

    shap_values = explain_customer(
        model, explainer, customers
    )

    return model, customers, explainer, shap_values


def test_shap_explainer_creation(shap_test_data):
    model, customers, explainer, shap_values = shap_test_data

    assert explainer is not None


def test_individual_shap_explanation(shap_test_data):
    model, customers, explainer, shap_values = shap_test_data

    assert shap_values.values.shape == (50, 23, 2)


def test_global_feature_importance(shap_test_data):
    model, customers, explainer, shap_values = shap_test_data

    importance = get_global_feature_importance(
        shap_values, customers.columns
    )

    assert len(importance) == 23
    assert importance["importance"].notna().all()
    assert importance["importance"].ge(0).all()


def test_customer_explanation(shap_test_data):
    model, customers, explainer, shap_values = shap_test_data

    customer = customers.iloc[[0]]

    customer_shap_values = explain_customer(
        model, explainer, customer
    )

    explanation = get_customer_explanation(
        customer_shap_values, customer.columns
    )

    assert len(explanation) == 23
    assert "impact" in explanation.columns
    assert set(explanation["impact"]).issubset({
        "Increases default risk",
        "Decreases default risk",
        "No impact",
    })


def test_shap_chart_generation(shap_test_data):
    model, customers, explainer, shap_values = shap_test_data

    customer = customers.iloc[[0]]

    customer_shap_values = explain_customer(
        model, explainer, customer
    )

    explanation = get_customer_explanation(
        customer_shap_values, customer.columns
    )

    fig = create_customer_shap_chart(explanation)

    assert len(fig.data) > 0
    assert fig.layout.title.text == (
        "Individual Customer — SHAP Explanation"
    )
