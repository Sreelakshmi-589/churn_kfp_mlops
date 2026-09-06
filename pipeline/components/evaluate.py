from kfp import dsl
from kfp.dsl import Input, Output, Dataset, Model


@dsl.component(
    base_image="python:3.11",
    packages_to_install=[
        "scikit-learn",
        "joblib",
    ],
)
def evaluate_model(
    model: Input[Model],
    test_data: Input[Dataset],
    metrics: Output[dsl.Metrics],
    threshold: float,
):

    import joblib

    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score as sklearn_f1_score,
    )

    # -------------------------
    # Load trained model
    # -------------------------

    classifier = joblib.load(model.path)

    # -------------------------
    # Load test data
    # -------------------------

    data = joblib.load(test_data.path)

    X_test = data["X"]
    y_test = data["y"]

    print(f"Test samples: {X_test.shape[0]}")
    print(f"Test features: {X_test.shape[1]}")

    # -------------------------
    # Generate predictions
    # -------------------------

    predictions = classifier.predict(X_test)

    # -------------------------
    # Calculate metrics
    # -------------------------

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1 = sklearn_f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    # -------------------------
    # Log metrics
    # -------------------------

    metrics.log_metric(
        "accuracy",
        accuracy,
    )

    metrics.log_metric(
        "precision",
        precision,
    )

    metrics.log_metric(
        "recall",
        recall,
    )

    metrics.log_metric(
        "f1_score",
        f1,
    )


    if f1 < threshold:
       raise ValueError(
           f"MODEL REJECTED: "
           f"F1 score {f1:.4f} is below "
           f"threshold {threshold:.4f}"
       )

    print(
        f"MODEL APPROVED: "
        f"F1 score {f1:.4f} >= "
        f"threshold {threshold:.4f}"
    )


    print("===================================")
    print("MODEL EVALUATION")
    print("===================================")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print("===================================")
