
from creditlens.models.training import (
    prepare_training_data,
    train_logistic_regression,
)
from creditlens.models.evaluation import (
    evaluate_model,
    print_evaluation_report,
)


DATASET_PATH = "data/raw/credit_card_default.xls"


def main():
    print("Loading CreditLens dataset...")

    X_train, X_test, y_train, y_test, preprocessor = (
        prepare_training_data(DATASET_PATH)
    )

    print("Training Logistic Regression...")

    model = train_logistic_regression(
        X_train,
        y_train,
        preprocessor,
    )

    print("Evaluating Logistic Regression...")

    results = evaluate_model(model, X_test, y_test)

    print_evaluation_report(results)


if __name__ == "__main__":
    main()
