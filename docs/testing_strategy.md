# CreditLens – Testing Strategy

## 1. Objective

Verify that CreditLens produces reliable predictions, handles
invalid inputs, stores assessment records correctly and displays
accurate portfolio analytics.

## 2. Unit Testing

Test individual application components:

- Input validation.
- Data preprocessing.
- Model prediction.
- SHAP explanation generation.
- Database operations.
- SQL analytics.

## 3. Model Testing

Evaluate the trained classification model using:

- Accuracy.
- Precision.
- Recall.
- F1-score.
- ROC-AUC.
- Confusion matrix.

Use a held-out test dataset for final evaluation.

## 4. Database Testing

Verify that:

- Assessment records are inserted correctly.
- Stored records can be retrieved.
- SQL aggregation queries return expected results.
- Invalid database operations are handled safely.

## 5. Application Testing

Verify that:

- The Streamlit application starts successfully.
- Valid applicant inputs produce predictions.
- Invalid inputs produce understandable errors.
- SHAP explanations correspond to the selected prediction.
- Assessment history and portfolio analytics display real database data.

## 6. Performance Testing

Measure:

- Model prediction execution time.
- SHAP explanation execution time.
- Database query execution time.
- Overall application responsiveness.

Record actual measurements. Do not publish estimated or unverified performance claims.

## 7. Test Tools

- pytest
- scikit-learn evaluation metrics
- Python time/performance measurement utilities

## 8. Completion Criteria

All critical application tests must pass.

Any known limitations, failed tests or performance constraints
must be documented before the project is presented as complete.
