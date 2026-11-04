import plotly.express as px
import shap
import pandas as pd


def create_shap_explainer(model, X_background):
    """
    Create a SHAP explainer for the trained Random Forest model.
    """

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    background_data = preprocessor.transform(X_background)

    explainer = shap.TreeExplainer(
        classifier,
        data=background_data,
        feature_perturbation="interventional",
        model_output="probability",
    )

    return explainer


def explain_customer(model, explainer, customer_data):
    """
    Explain the Random Forest prediction for an individual customer.
    """

    preprocessor = model.named_steps["preprocessor"]

    transformed_customer = preprocessor.transform(customer_data)

    shap_values = explainer(transformed_customer)

    return shap_values

def get_global_feature_importance(shap_values, feature_names):
    """
    Calculate global feature importance using mean absolute SHAP values.
    """

    values = shap_values.values

    # Select the default (class 1) SHAP values.
    default_shap_values = values[:, :, 1]

    importance = pd.DataFrame({
        "feature": feature_names,
        "importance": abs(default_shap_values).mean(axis=0),
    })

    importance = importance.sort_values(
        by="importance",
        ascending=False,
    )

    return importance.reset_index(drop=True)

def get_customer_explanation(shap_values, feature_names):
    """
    Identify features increasing and decreasing
    the predicted default risk for one customer.
    """

    values = shap_values.values[0, :, 1]

    explanation = pd.DataFrame({
        "feature": feature_names,
        "shap_value": values,
    })

    explanation["impact"] = explanation["shap_value"].apply(
        lambda value: (
            "Increases default risk"
            if value > 0
            else "Decreases default risk"
            if value < 0
            else "No impact"
        )
    )

    explanation["absolute_impact"] = (
        explanation["shap_value"].abs()
    )

    explanation = explanation.sort_values(
        by="absolute_impact",
        ascending=False,
    )

    return explanation.reset_index(drop=True)

def create_customer_shap_chart(explanation, top_n=10):
    """
    Create a Plotly chart showing the strongest
    SHAP contributions for an individual customer.
    """

    chart_data = explanation.head(top_n).copy()

    chart_data = chart_data.sort_values(
        by="shap_value",
        ascending=True,
    )

    chart_data["direction"] = chart_data["shap_value"].apply(
        lambda value: (
            "Increases default risk"
            if value > 0
            else "Decreases default risk"
        )
    )

    fig = px.bar(
        chart_data,
        x="shap_value",
        y="feature",
        color="direction",
        orientation="h",
        color_discrete_map={
            "Increases default risk": "#D97757",
            "Decreases default risk": "#269B78",
        },
        title="Individual Customer — SHAP Explanation",
        labels={
            "shap_value": "SHAP contribution",
            "feature": "Customer feature",
            "direction": "Impact",
        },
    )

    fig.add_vline(
        x=0,
        line_dash="dash",
        line_color="gray",
    )

    fig.update_layout(
        template="plotly_white",
        height=500,
        showlegend=True,
    )

    return fig
