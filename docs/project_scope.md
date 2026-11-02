# CreditLens – Project Scope

## 1. Project Purpose

CreditLens is an explainable machine learning application designed to
assess credit default risk and provide insights into loan portfolios.

The application will allow users to submit applicant information,
receive a model-generated risk assessment, understand the factors
influencing the prediction, and review stored assessment records.

## 2. Core Features

### 2.1 Credit Risk Prediction
- Accept applicant financial and demographic inputs.
- Validate submitted information.
- Generate predictions using a trained Random Forest classifier.
- Display the predicted risk category and model probability.

### 2.2 Model Explainability
- Use SHAP to explain individual model predictions.
- Display the factors contributing to a prediction.
- Help users understand model behaviour.

### 2.3 Assessment History
- Store assessment records in SQLite.
- Retrieve previously submitted assessments.
- Display relevant assessment details.

### 2.4 Loan Portfolio Analytics
- Use SQL queries to analyse stored assessment records.
- Summarise risk categories and assessment trends.
- Present results through interactive charts.

### 2.5 Model Evaluation
- Evaluate the model using appropriate classification metrics.
- Review confusion matrix, precision, recall, F1-score and ROC-AUC.
- Record evaluation results transparently.

## 3. Technology Stack

- Python
- Streamlit
- pandas
- NumPy
- scikit-learn
- SHAP
- SQLite
- SQL
- Plotly
- pytest

## 4. Application Architecture

The application will use a modular Python architecture.

Streamlit will provide the user interface.
Separate Python modules will handle input validation, prediction,
explainability, database operations and analytics.

SQLite will store assessment history.

## 5. Out of Scope

The initial version will not include:

- Real-time integration with banking systems.
- Automated loan approval or rejection.
- Processing of real customer banking records.
- User authentication and multi-user access control.
- Production-grade credit underwriting.

## 6. Responsible Use

CreditLens is an educational and portfolio project.

Its predictions are experimental and must not be used as the sole
basis for actual lending decisions.

Model performance, limitations, data quality and potential bias
must be considered when interpreting results.

## 7. Completion Criteria

The initial version will be considered complete when:

- The application runs locally.
- A trained model generates predictions.
- Individual predictions have SHAP explanations.
- Assessment records are stored and retrieved from SQLite.
- SQL portfolio analytics work correctly.
- Automated tests pass.
- The README contains verified setup and usage instructions.
- A working demonstration is available.
