from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import os

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "model" / "model.pkl"
PREPROCESSOR_PATH = BASE_DIR / "model" / "preprocessor.pkl"


model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)


app = FastAPI(
    title="Customer Churn Prediction API",
    version="1.0.0",
)


class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


class PredictionResponse(BaseModel):
    churn_prediction: int
    churn: str
    churn_probability: float


@app.get("/")
def root():
    return {
        "message": "Customer Churn Prediction API",
        "status": "healthy",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None,
        "pod": os.getenv("HOSTNAME", "local"),
    }


@app.get("/model-info")
def model_info():
    return {
        "model_type": type(model).__name__,
        "preprocessor_type": type(preprocessor).__name__,
        "model_path": str(MODEL_PATH),
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(customer: CustomerData):

    data = pd.DataFrame([customer.model_dump()])

    transformed_data = preprocessor.transform(data)

    prediction = model.predict(transformed_data)[0]

    probability = model.predict_proba(
        transformed_data
    )[0][1]

    return {
        "churn_prediction": int(prediction),
        "churn": "Yes" if prediction == 1 else "No",
        "churn_probability": round(
            float(probability),
            4,
        ),
    }
