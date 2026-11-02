# CreditLens – Model Evaluation Plan

## 1. Objective

Evaluate the ability of the CreditLens classification model
to identify potential loan defaults.

## 2. Initial Model

Algorithm: Random Forest Classifier.

The model will be trained using a suitable publicly available
credit-risk dataset.

## 3. Evaluation Metrics

- Accuracy: Overall proportion of correct predictions.
- Precision: Proportion of predicted defaults that are actual defaults.
- Recall: Proportion of actual defaults correctly identified.
- F1-score: Harmonic mean of precision and recall.
- ROC-AUC: Ability to distinguish between default and non-default cases.

## 4. Evaluation Process

1. Inspect and clean the dataset.
2. Separate input features and target variable.
3. Split the data into training and testing sets.
4. Train the classification model.
5. Generate predictions on unseen test data.
6. Calculate evaluation metrics.
7. Review the confusion matrix.
8. Document the results and limitations.

## 5. Class Imbalance

Credit-risk datasets may contain fewer default cases than
non-default cases.

Accuracy alone may therefore be misleading.

Precision, recall, F1-score and ROC-AUC will also be considered.

## 6. Explainability

SHAP will be used to investigate the contribution of input
features to individual predictions.

## 7. Reporting

Only metrics produced by the implemented and tested model
will be reported.

No performance results are assumed in advance.
