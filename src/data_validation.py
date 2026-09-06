from pathlib import Path
import pandas as pd


EXPECTED_COLUMNS = [
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


def validate_dataset(input_path: str) -> str:
    """Validate the Telco Customer Churn dataset."""

    path = Path(input_path)

    # Check file exists
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {input_path}"
        )

    # Load dataset
    df = pd.read_csv(path)

    # Check dataset isn't empty
    if df.empty:
        raise ValueError("Dataset is empty")

    # Check required columns
    missing_columns = set(EXPECTED_COLUMNS) - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    # Check missing values
    missing_values = df[EXPECTED_COLUMNS].isnull().sum()

    columns_with_missing_values = missing_values[
        missing_values > 0
    ]

    if not columns_with_missing_values.empty:
        raise ValueError(
            f"Missing values found:\n{columns_with_missing_values}"
        )

    # Check duplicate customer IDs
    duplicate_customers = df["customerID"].duplicated().sum()

    if duplicate_customers > 0:
        raise ValueError(
            f"Found {duplicate_customers} duplicate customer IDs"
        )

    # Check target values
    valid_target_values = {"Yes", "No"}

    actual_target_values = set(df["Churn"].unique())

    invalid_target_values = actual_target_values - valid_target_values

    if invalid_target_values:
        raise ValueError(
            f"Invalid Churn values found: {invalid_target_values}"
        )

    print("===================================")
    print("DATA VALIDATION PASSED")
    print("===================================")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Duplicate customers: {duplicate_customers}")
    print(
        f"Missing values: "
        f"{df[EXPECTED_COLUMNS].isnull().sum().sum()}"
    )
    print("Churn distribution:")
    print(df["Churn"].value_counts())
    print("===================================")

    return "PASS"


if __name__ == "__main__":
    result = validate_dataset("data/churn.csv")
    print(f"Validation result: {result}")
