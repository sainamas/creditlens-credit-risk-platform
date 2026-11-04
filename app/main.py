
from creditlens.models.training import prepare_training_data
import streamlit as st
import pandas as pd
import plotly.express as px
from creditlens.models.training import load_model
from creditlens.prediction.predictor import (
    predict_default_probability,
    classify_risk,
)

from creditlens.explainability.shap_explainer import (
    create_shap_explainer,
    explain_customer,
    get_customer_explanation,
    get_global_feature_importance,
    create_customer_shap_chart,
)

from creditlens.data.preprocessing import (
    load_dataset,
    separate_features_and_target,
)
@st.cache_resource
def get_prediction_model():
    """Load and cache the trained Random Forest model."""
    return load_model("random_forest")
@st.cache_resource
def get_shap_explainer():
    """Create and cache the Random Forest SHAP explainer."""

    X_train, X_test, y_train, y_test, preprocessor = (
        prepare_training_data(
            "data/raw/credit_card_default.xls"
        )
    )

    model = get_prediction_model()

    background = X_train.sample(
        n=100,
        random_state=42,
    )

    return create_shap_explainer(model, background)

st.set_page_config(
    page_title="CreditLens | Credit Risk Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Application title
st.title("CreditLens")
st.subheader("Credit Risk Assessment & Explainability Platform")

st.markdown(
    """
    CreditLens is an interactive credit risk platform that uses
    Machine Learning to estimate Probability of Default (PD)
    and SHAP to explain model predictions.
    """
)

st.divider()


# Sidebar navigation
st.sidebar.title("CreditLens Navigation")

page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "PD Prediction",
        "Model Comparison",
        "SHAP Explainability",
        "Expected Loss Calculator",
        "Credit Risk Fundamentals",
    ],
)


# Page content
if page == "Overview":

    st.header("Platform Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Dataset Records", "30,000")
    col2.metric("ML Models", "2")
    col3.metric("Input Features", "23")

    st.markdown(
        """
        ### Welcome to CreditLens

        Explore the platform's credit risk capabilities:

        - **PD Prediction:** Estimate a customer's probability of default.
        - **Model Comparison:** Compare Logistic Regression and Random Forest.
        - **SHAP Explainability:** Understand the factors influencing predictions.
        - **Expected Loss:** Calculate illustrative expected credit loss.
        - **Credit Risk Fundamentals:** Learn core credit risk concepts.
        """
    )


elif page == "PD Prediction":

    st.header("Customer Credit Risk Assessment")

    st.markdown(
        "Enter customer financial information to estimate "
        "the probability of default using Random Forest."
    )

    st.info(
        "The form uses the numerical coding of the UCI "
        "Default of Credit Card Clients dataset. "
        "Review the coding before submitting."
    )

    # Dataset coding
    education_labels = {
        0: "0 - Unknown",
        1: "1 - Graduate school",
        2: "2 - University",
        3: "3 - High school",
        4: "4 - Others",
        5: "5 - Unknown",
        6: "6 - Unknown",
    }

    marriage_labels = {
        0: "0 - Unknown",
        1: "1 - Married",
        2: "2 - Single",
        3: "3 - Others",
    }

    repayment_labels = {
        -2: "-2 - No consumption",
        -1: "-1 - Paid duly",
        0: "0 - Revolving credit",
        1: "1 - Payment delay 1 month",
        2: "2 - Payment delay 2 months",
        3: "3 - Payment delay 3 months",
        4: "4 - Payment delay 4 months",
        5: "5 - Payment delay 5 months",
        6: "6 - Payment delay 6 months",
        7: "7 - Payment delay 7 months",
        8: "8 - Payment delay 8 months",
    }

    with st.form("customer_prediction_form"):

        st.subheader("1. Credit and Demographic Information")

        col1, col2 = st.columns(2)

        with col1:
            limit_bal = st.number_input(
                "Credit limit (LIMIT_BAL)",
                min_value=0,
                max_value=1000000,
                value=20000,
                step=10000,
            )

            sex = st.selectbox(
                "Gender (SEX)",
                options=[1, 2],
                format_func=lambda x: (
                    "1 - Male" if x == 1 else "2 - Female"
                ),
            )

            education = st.selectbox(
                "Education (EDUCATION)",
                options=list(education_labels.keys()),
                format_func=lambda x: education_labels[x],
                index=2,
            )

        with col2:
            marriage = st.selectbox(
                "Marital status (MARRIAGE)",
                options=list(marriage_labels.keys()),
                format_func=lambda x: marriage_labels[x],
                index=2,
            )

            age = st.number_input(
                "Age (AGE)",
                min_value=18,
                max_value=100,
                value=24,
                step=1,
            )

        st.divider()

        st.subheader("2. Repayment History")

        repayment_fields = [
            "PAY_0",
            "PAY_2",
            "PAY_3",
            "PAY_4",
            "PAY_5",
            "PAY_6",
        ]

        repayment_values = {}

        repayment_columns = st.columns(3)

        for index, feature in enumerate(repayment_fields):
            with repayment_columns[index % 3]:
                repayment_values[feature] = st.selectbox(
                    feature,
                    options=list(repayment_labels.keys()),
                    format_func=lambda x: repayment_labels[x],
                    index=2,
                    key=f"prediction_{feature}",
                )

        st.divider()

        st.subheader("3. Monthly Bill Amounts")

        bill_fields = [
            "BILL_AMT1",
            "BILL_AMT2",
            "BILL_AMT3",
            "BILL_AMT4",
            "BILL_AMT5",
            "BILL_AMT6",
        ]

        bill_values = {}

        bill_columns = st.columns(3)

        for index, feature in enumerate(bill_fields):
            with bill_columns[index % 3]:
                bill_values[feature] = st.number_input(
                    feature,
                    min_value=-1000000,
                    max_value=10000000,
                    value=0,
                    step=1000,
                    key=f"prediction_{feature}",
                )

        st.divider()

        st.subheader("4. Previous Payment Amounts")

        payment_fields = [
            "PAY_AMT1",
            "PAY_AMT2",
            "PAY_AMT3",
            "PAY_AMT4",
            "PAY_AMT5",
            "PAY_AMT6",
        ]

        payment_values = {}

        payment_columns = st.columns(3)

        for index, feature in enumerate(payment_fields):
            with payment_columns[index % 3]:
                payment_values[feature] = st.number_input(
                    feature,
                    min_value=0,
                    max_value=10000000,
                    value=0,
                    step=1000,
                    key=f"prediction_{feature}",
                )

        st.divider()

        submitted = st.form_submit_button(
            "Calculate Default Probability",
            use_container_width=True,
        )

    if submitted:

        customer = {
            "LIMIT_BAL": limit_bal,
            "SEX": sex,
            "EDUCATION": education,
            "MARRIAGE": marriage,
            "AGE": age,
            **repayment_values,
            **bill_values,
            **payment_values,
        }

        customer_data = pd.DataFrame([customer])

        try:
            model = get_prediction_model()

            probability = predict_default_probability(
                model,
                customer_data,
            )[0]

            risk_category = classify_risk(probability)

            st.divider()
            st.subheader("Customer Risk Assessment")

            col1, col2 = st.columns(2)

            col1.metric(
                "Probability of Default",
                f"{probability:.2%}",
            )

            col2.metric(
                "Risk Category",
                risk_category,
            )

            st.progress(float(probability))

            st.caption(
                "This prediction is generated by the trained "
                "Random Forest model. Risk categories use "
                "CreditLens illustrative thresholds."
            )
            st.divider()
            st.subheader("Why did the model make this prediction?")

            with st.spinner("Generating SHAP explanation..."):
                explainer = get_shap_explainer()

                shap_values = explain_customer(
                    model,
                    explainer,
                    customer_data,
                )

                explanation = get_customer_explanation(
                    shap_values,
                    customer_data.columns,
                )

            st.markdown("**Top factors influencing this prediction**")

            st.plotly_chart(
                create_customer_shap_chart(explanation, top_n=10),
                use_container_width=True,
            )

            left, right = st.columns(2)

            with left:
                st.markdown("**Factors increasing default risk**")
                st.dataframe(
                    explanation[
                        explanation["impact"] == "Increases default risk"
                    ][["feature", "shap_value"]]
                    .head(5)
                    .rename(columns={
                        "feature": "Feature",
                        "shap_value": "SHAP contribution",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )

            with right:
                st.markdown("**Factors decreasing default risk**")
                st.dataframe(
                    explanation[
                        explanation["impact"] == "Decreases default risk"
                    ][["feature", "shap_value"]]
                    .head(5)
                    .rename(columns={
                        "feature": "Feature",
                        "shap_value": "SHAP contribution",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )
        except Exception as error:
            st.error(f"Prediction failed: {error}")



elif page == "Model Comparison":

    st.header("Model Performance Comparison")

    st.markdown(
        "Compare the performance of the two trained "
        "credit default classification models."
    )

    results = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1-score",
            "ROC-AUC",
        ],
        "Logistic Regression": [
            0.6797, 0.3672, 0.6202, 0.4613, 0.7081
        ],
        "Random Forest": [
            0.7778, 0.4981, 0.5931, 0.5415, 0.7747
        ],
    })

    col1, col2 = st.columns(2)

    col1.metric("Random Forest ROC-AUC", "77.47%")
    col2.metric("Random Forest Accuracy", "77.78%")

    st.subheader("Performance Metrics")

    st.dataframe(
        results.set_index("Metric").style.format("{:.2%}"),
        use_container_width=True,
    )

    st.subheader("Model Performance Chart")

    chart_data = results.melt(
        id_vars="Metric",
        var_name="Model",
        value_name="Score",
    )

    fig = px.bar(
        chart_data,
        x="Metric",
        y="Score",
        color="Model",
        barmode="group",
        text_auto=".1%",
        title="Logistic Regression vs Random Forest",
    )

    fig.update_layout(
        yaxis=dict(
            title="Performance Score",
            tickformat=".0%",
            range=[0, 1],
        ),
        xaxis_title="Evaluation Metric",
        template="plotly_dark",
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Random Forest performs better on four of the five "
        "metrics. Logistic Regression achieves slightly higher "
        "recall."
    )

    st.caption(
        "Results are from the same stratified 20% held-out test set."
    )


elif page == "SHAP Explainability":

    st.header("Global SHAP Explainability")

    st.markdown(
        """
        Analyse which customer features have the greatest influence
        on the Random Forest model's default predictions across a
        sample of customers.
        """
    )

    try:
        with st.spinner("Calculating global feature importance..."):

            X_train, X_test, y_train, y_test, preprocessor = (
                prepare_training_data(
                    "data/raw/credit_card_default.xls"
                )
            )

            model = get_prediction_model()
            explainer = get_shap_explainer()

            sample_size = min(100, len(X_test))

            X_sample = X_test.sample(
                n=sample_size,
                random_state=42,
            )

            shap_values = explain_customer(
                model,
                explainer,
                X_sample,
            )

            importance = get_global_feature_importance(
                shap_values,
                X_sample.columns,
            )

        st.success(
            f"Analysis completed using {sample_size} test customers."
        )

        st.subheader("Top 10 Most Influential Features")

        chart_data = importance.head(10).sort_values(
            by="importance",
            ascending=True,
        )

        fig = px.bar(
            chart_data,
            x="importance",
            y="feature",
            orientation="h",
            title="Global Feature Importance — Mean Absolute SHAP",
            labels={
                "importance": "Mean Absolute SHAP Value",
                "feature": "Customer Feature",
            },
        )

        fig.update_layout(
            template="plotly_white",
            height=550,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        st.subheader("Feature Importance Rankings")

        st.dataframe(
            importance.head(10).rename(
                columns={
                    "feature": "Feature",
                    "importance": "Mean Absolute SHAP Value",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.info(
            """
            **How to interpret this analysis**

            - Higher mean absolute SHAP values indicate greater
              influence on the model's predictions across the sample.
            - Global importance does not show whether a feature
              increases or decreases risk for every customer.
            - SHAP explains model behaviour, not causal relationships.
            """
        )

    except Exception as error:
        st.error(f"Global SHAP analysis failed: {error}")




elif page == "Expected Loss Calculator":
    st.header("Expected Credit Loss Calculator")
    st.info("The expected loss calculator is being developed.")


elif page == "Credit Risk Fundamentals":
    st.header("Credit Risk Fundamentals")
    st.info("Educational credit risk content is being added.")
