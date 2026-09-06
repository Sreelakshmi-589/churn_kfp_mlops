from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_root():
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"


def test_prediction():
    customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 50.0,
        "TotalCharges": 600.0,
    }

    response = client.post(
        "/predict",
        json=customer,
    )

    assert response.status_code == 200

    data = response.json()

    assert "churn_prediction" in data
    assert "churn" in data
    assert "churn_probability" in data

    assert data["churn_prediction"] in [0, 1]
    assert data["churn"] in ["Yes", "No"]
    assert 0.0 <= data["churn_probability"] <= 1.0


def test_invalid_request():
    customer = {
        "gender": "Female"
    }

    response = client.post(
        "/predict",
        json=customer,
    )

    assert response.status_code == 422

def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["preprocessor_loaded"] is True


def test_model_info():
    response = client.get("/model-info")

    assert response.status_code == 200

    data = response.json()

    assert data["model_type"] == "LogisticRegression"
    assert data["preprocessor_type"] == "ColumnTransformer"
