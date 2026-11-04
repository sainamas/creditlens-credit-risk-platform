
from creditlens.models.training import (
    prepare_training_data,
    train_logistic_regression,
    train_random_forest,
    save_model,
)
from creditlens.models.evaluation import evaluate_model


DATASET_PATH = "data/raw/credit_card_default.xls"


def main():
    print("Loading CreditLens dataset...")

    X_train, X_test, y_train, y_test, preprocessor = (
        prepare_training_data(DATASET_PATH)
    )

    # Train Logistic Regression
    print("\nTraining Logistic Regression...")
    logistic_model = train_logistic_regression(
        X_train, y_train, preprocessor
    )

    # Train Random Forest
    print("\nTraining Random Forest...")
    random_forest_model = train_random_forest(
        X_train, y_train, preprocessor
    )

    # Evaluate both models
    print("\nEvaluating models...")

    models = {
        "Logistic Regression": logistic_model,
        "Random Forest": random_forest_model,
    }

    results = {}

    for name, model in models.items():
        results[name] = evaluate_model(
            model, X_test, y_test
        )
    print("\nSaving trained models...")

    save_model(logistic_model, "logistic_regression")
    save_model(random_forest_model, "random_forest")
    # Display comparison
    print("\n========== MODEL COMPARISON ==========")

    metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
    ]

    print(f"{'Metric':<15}{'Logistic Regression':<23}{'Random Forest':<18}")

    for metric in metrics:
        print(
            f"{metric:<15}"
            f"{results['Logistic Regression'][metric]:<23.4f}"
            f"{results['Random Forest'][metric]:<18.4f}"
        )


if __name__ == "__main__":
    main()
