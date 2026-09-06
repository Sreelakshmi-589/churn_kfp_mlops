from kfp import dsl
from kfp.dsl import Input, Output, Dataset, Model


@dsl.component(
    base_image="python:3.11",
    packages_to_install=[
        "pandas",
        "scikit-learn",
        "joblib",
    ],
)
def preprocess_data(
    input_data: Input[Dataset],
    train_data: Output[Dataset],
    test_data: Output[Dataset],
    preprocessor: Output[Model],
):

    import joblib
    import pandas as pd

    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    # -------------------------
    # Load data
    # -------------------------

    df = pd.read_csv(input_data.path)

    print(f"Original shape: {df.shape}")

    # -------------------------
    # Basic transformations
    # -------------------------

    df = df.drop(columns=["customerID"])

    df["Churn"] = df["Churn"].map({
        "No": 0,
        "Yes": 1,
    })

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce",
    )

    # -------------------------
    # Features / target
    # -------------------------

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

    print(
        f"Numerical features: {numerical_features}"
    )

    print(
        f"Categorical features: {categorical_features}"
    )

    # -------------------------
    # Numerical pipeline
    # -------------------------

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    # -------------------------
    # Categorical pipeline
    # -------------------------

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    # -------------------------
    # Combined preprocessor
    # -------------------------

    preprocessor_pipeline = ColumnTransformer(
        transformers=[
            (
                "num",
                numerical_pipeline,
                numerical_features,
            ),
            (
                "cat",
                categorical_pipeline,
                categorical_features,
            ),
        ]
    )

    # -------------------------
    # Train/test split
    # -------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # -------------------------
    # Fit ONLY on training data
    # -------------------------

    X_train_processed = (
        preprocessor_pipeline.fit_transform(X_train)
    )

    X_test_processed = (
        preprocessor_pipeline.transform(X_test)
    )

    print(
        f"Training shape: {X_train_processed.shape}"
    )

    print(
        f"Test shape: {X_test_processed.shape}"
    )

    # -------------------------
    # Save training dataset
    # -------------------------

    joblib.dump(
        {
            "X": X_train_processed,
            "y": y_train.to_numpy(),
        },
        train_data.path,
    )

    # -------------------------
    # Save test dataset
    # -------------------------

    joblib.dump(
        {
            "X": X_test_processed,
            "y": y_test.to_numpy(),
        },
        test_data.path,
    )

    # -------------------------
    # Save preprocessor
    # -------------------------

    joblib.dump(
        preprocessor_pipeline,
        preprocessor.path,
    )

    print("Preprocessing completed successfully.")
