from kfp import dsl
from kfp.dsl import Input, Dataset


@dsl.component(
    base_image="python:3.11",
    packages_to_install=["pandas"],
)
def validate_data(
    input_data: Input[Dataset],
) -> str:

    from pathlib import Path
    import pandas as pd

    input_path = Path(input_data.path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {input_path}"
        )

    df = pd.read_csv(input_path)

    if df.empty:
        raise ValueError("Dataset is empty")

    expected_columns = [
        "customerID",
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
        "MonthlyCharges",
        "TotalCharges",
        "Churn",
    ]

    missing_columns = (
        set(expected_columns) - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    missing_values = df[expected_columns].isnull().sum()

    columns_with_missing_values = (
        missing_values[missing_values > 0]
    )

    if not columns_with_missing_values.empty:
        raise ValueError(
            f"Missing values found:\n"
            f"{columns_with_missing_values}"
        )

    duplicate_customers = (
        df["customerID"].duplicated().sum()
    )

    if duplicate_customers > 0:
        raise ValueError(
            f"Found {duplicate_customers} "
            f"duplicate customer IDs"
        )

    valid_target_values = {"Yes", "No"}

    actual_target_values = set(
        df["Churn"].unique()
    )

    invalid_target_values = (
        actual_target_values - valid_target_values
    )

    if invalid_target_values:
        raise ValueError(
            f"Invalid Churn values found: "
            f"{invalid_target_values}"
        )

    print("===================================")
    print("DATA VALIDATION PASSED")
    print("===================================")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(
        f"Duplicate customers: "
        f"{duplicate_customers}"
    )
    print(
        f"Missing values: "
        f"{df[expected_columns].isnull().sum().sum()}"
    )
    print("Churn distribution:")
    print(df["Churn"].value_counts())
    print("===================================")

    return "PASS"
