from kfp import dsl
from kfp.dsl import Input, Output, Dataset, Model


@dsl.component(
    base_image="python:3.11",
    packages_to_install=[
        "scikit-learn",
        "joblib",
    ],
)
def train_model(
    train_data: Input[Dataset],
    model: Output[Model],
):

    import joblib
    from sklearn.linear_model import LogisticRegression

    # -------------------------
    # Load training data
    # -------------------------

    data = joblib.load(train_data.path)

    X_train = data["X"]
    y_train = data["y"]

    print(f"Training samples: {X_train.shape[0]}")
    print(f"Training features: {X_train.shape[1]}")

    # -------------------------
    # Create model
    # -------------------------

    classifier = LogisticRegression(
        max_iter=1000,
        random_state=42,
    )

    # -------------------------
    # Train model
    # -------------------------

    classifier.fit(
        X_train,
        y_train,
    )

    # -------------------------
    # Save model
    # -------------------------

    joblib.dump(
        classifier,
        model.path,
    )

    # -------------------------
    # Model metadata
    # -------------------------

    model.metadata["model_type"] = "LogisticRegression"
    model.metadata["training_samples"] = X_train.shape[0]
    model.metadata["features"] = X_train.shape[1]

    print("===================================")
    print("MODEL TRAINING COMPLETED")
    print("===================================")
    print(f"Model: Logistic Regression")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Model saved to: {model.path}")
    print("===================================")
