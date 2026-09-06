from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def preprocess_dataset(input_path: str, output_dir: str):
    """Prepare the churn dataset for model training."""

    input_path = Path(input_path)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = pd.read_csv(input_path)

    print(f"Original shape: {df.shape}")

    # Remove customer identifier
    df = df.drop(columns=["customerID"])

    # Convert target to numerical values
    df["Churn"] = df["Churn"].map({
        "No": 0,
        "Yes": 1
    })

    # TotalCharges can contain blank strings
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # Separate features and target
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    numerical_features = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
    ]

    categorical_features = [
        column
        for column in X.columns
        if column not in numerical_features
    ]

    print(f"Numerical features: {numerical_features}")
    print(f"Categorical features: {categorical_features}")

    # Numerical preprocessing
    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Categorical preprocessing
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            ),
        ]
    )

    # Combine preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numerical_pipeline,
                numerical_features
            ),
            (
                "cat",
                categorical_pipeline,
                categorical_features
            ),
        ]
    )

    # IMPORTANT:
    # Split BEFORE fitting the preprocessor.
    # This prevents data leakage.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Fit only on training data
    X_train_processed = preprocessor.fit_transform(X_train)

    # Transform test data using the already-fitted preprocessor
    X_test_processed = preprocessor.transform(X_test)

    print(f"Training shape: {X_train_processed.shape}")
    print(f"Test shape: {X_test_processed.shape}")

    # Save processed datasets
    train_data = {
        "X": X_train_processed,
        "y": y_train.to_numpy(),
    }

    test_data = {
        "X": X_test_processed,
        "y": y_test.to_numpy(),
    }

    joblib.dump(
        train_data,
        output_dir / "train.pkl"
    )

    joblib.dump(
        test_data,
        output_dir / "test.pkl"
    )

    # Save preprocessing pipeline
    joblib.dump(
        preprocessor,
        output_dir / "preprocessor.pkl"
    )

    print("Preprocessing completed successfully.")

    return output_dir


if __name__ == "__main__":
    preprocess_dataset(
        "data/churn.csv",
        "model/processed"
    )
