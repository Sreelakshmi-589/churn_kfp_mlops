# Customer Churn Prediction — End-to-End MLOps Project

An end-to-end MLOps project for predicting customer churn using machine learning, Kubeflow Pipelines, FastAPI, Docker, Kubernetes, GitHub Actions, and GitHub Container Registry.

The goal of this project is not only to train a machine learning model, but to build a complete workflow around it, from data validation and preprocessing to testing, containerisation, model serving, and Kubernetes deployment.

---

## 1. Project Overview

This project uses the **IBM Telco Customer Churn dataset** to predict whether a customer is likely to leave a telecom service.

The project follows this workflow:

```text
Customer Churn Dataset
        ↓
Data Validation
        ↓
Data Preprocessing
        ↓
Model Training
        ↓
Model Evaluation
        ↓
Quality Gate
        ↓
FastAPI
        ↓
Docker
        ↓
GitHub Actions
        ↓
GitHub Container Registry
        ↓
Kubernetes / Minikube
```

The project is designed as a practical MLOps learning project, combining machine learning with software engineering, CI/CD, containerisation, and Kubernetes.

---

# 2. What We Built

The project currently contains the following major components:

### Machine Learning

* Customer churn prediction
* Logistic Regression model
* Numerical and categorical preprocessing
* Train/test split
* Feature scaling
* One-hot encoding
* Missing-value handling
* Model evaluation
* F1-score quality gate

### MLOps

* Kubeflow Pipelines (KFP)
* Pipeline components
* Data validation
* Reproducible preprocessing
* Model artifacts
* Evaluation metrics
* Quality gate

### API

* FastAPI model-serving API
* `/`
* `/health`
* `/model-info`
* `/predict`

### Containerisation

* Docker
* Docker image containing the API and model artifacts
* `.dockerignore`

### CI/CD

* GitHub Actions
* Automated tests
* Docker image build
* GitHub Container Registry (GHCR)
* Versioned Docker images using Git commit SHA
* `latest` Docker image tag

### Kubernetes

* Kubernetes Deployment
* Two API replicas
* Kubernetes Service
* Readiness probe
* Resource requests and limits
* Private GHCR image pull using Kubernetes Secret
* Service load balancing
* Pod self-healing

---

# 3. Dataset

The project uses the **IBM Telco Customer Churn dataset**.

The dataset contains:

* **7,043 customers**
* **21 columns**
* Target column: `Churn`

The target contains:

```text
No  → 5174
Yes → 1869
```

Approximately 26.5% of customers in the dataset churned.

Important columns include:

```text
customerID
gender
SeniorCitizen
Partner
Dependents
tenure
PhoneService
MultipleLines
InternetService
OnlineSecurity
OnlineBackup
DeviceProtection
TechSupport
StreamingTV
StreamingMovies
Contract
PaperlessBilling
PaymentMethod
MonthlyCharges
TotalCharges
Churn
```

---

# 4. Machine Learning Pipeline

The machine learning workflow is implemented using **Kubeflow Pipelines**.

The pipeline contains the following stages:

```text
Load Data
    ↓
Validate Data
    ↓
Preprocess Data
    ↓
Train Model
    ↓
Evaluate Model
    ↓
Quality Check
```

## Load Data

The pipeline loads the churn dataset and passes it to the following components.

## Data Validation

The validation component checks:

* Expected columns exist
* Missing values
* Duplicate customer IDs
* Valid churn values

This helps prevent invalid data from reaching the training stage.

## Preprocessing

The preprocessing stage:

1. Removes `customerID`
2. Converts `Churn` from `Yes/No` to `1/0`
3. Converts `TotalCharges` to numeric values
4. Handles missing numerical values
5. Handles missing categorical values
6. Scales numerical features
7. One-hot encodes categorical features
8. Splits the data into training and testing sets

The preprocessing pipeline uses:

```text
Numerical features
    ↓
Median Imputation
    ↓
StandardScaler

Categorical features
    ↓
Most Frequent Imputation
    ↓
OneHotEncoder
```

The preprocessing transformer is saved as an artifact so that the same transformation can be used when serving predictions.

---

# 5. Preventing Data Leakage

One important decision in the preprocessing pipeline was to split the dataset into training and testing data **before fitting the preprocessing transformer**.

The preprocessor is fitted only on the training data.

```text
Raw Data
   ↓
Train/Test Split
   │
   ├── Training Data → Fit Preprocessor
   │
   └── Test Data → Transform using fitted Preprocessor
```

This prevents information from the test dataset from influencing the training process.

This is an important machine learning engineering practice because preprocessing the complete dataset before splitting can introduce data leakage.

---

# 6. Model Training

The current model is:

```text
Logistic Regression
```

Configuration:

```python
LogisticRegression(
    max_iter=1000,
    random_state=42
)
```

Training data:

```text
Training samples: 5634
Features: 46
```

The trained model is saved as:

```text
model/model.pkl
```

The preprocessing transformer is saved as:

```text
model/preprocessor.pkl
```

---

# 7. Model Evaluation

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score

The successful evaluation produced:

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 0.8055 |
| Precision | 0.6572 |
| Recall    | 0.5588 |
| F1 Score  | 0.6040 |

The project uses **F1 score** as the main quality metric.

The quality threshold is:

```text
F1 >= 0.60
```

The current model achieved:

```text
F1 = 0.6040
```

Therefore:

```text
Quality Gate → PASSED
```

The purpose of the quality gate is to prevent a model that does not meet the minimum performance requirement from progressing through the pipeline.

---

# 8. Kubeflow Pipelines

The project uses:

```text
kfp==2.17.0
```

The pipeline is defined in:

```text
pipeline/pipeline.py
```

Components are located in:

```text
pipeline/components/
```

The components are:

```text
load_data.py
validate.py
preprocess.py
train.py
evaluate.py
quality_gate.py
```

The pipeline was initially explored using Docker-based local execution.

However, this caused problems with Windows paths and Docker volume mounting.

---

# 9. Why We Used WSL

The development machine uses Windows 11.

The project also uses:

* Docker
* Kubernetes
* Minikube
* Linux-based containers
* Kubeflow Pipelines

Running the project directly from the Windows filesystem caused problems with Linux container paths and Docker volume mounts.

For example, local KFP execution encountered Windows path/volume issues.

Instead of fighting the Windows filesystem behaviour, the project was moved into the Linux filesystem provided by **WSL2 Ubuntu**.

The project is now located under:

This gives us a Linux-native development environment while still using the Windows laptop.

The architecture is:

```text
Windows 11
    │
    └── WSL2
         │
         └── Ubuntu
              │
              ├── Project
              ├── Docker
              ├── kubectl
              └── Minikube
```

This made the development environment much closer to the Linux environments commonly used in cloud and Kubernetes infrastructure.

---

# 10. Local KFP Execution

The initial local execution approach used Docker-based execution.

However, KFP's local Docker runner caused filesystem and volume-mount problems when working across the Windows/WSL/Docker boundary.

To make local development reliable, the project switched to:

```python
local.SubprocessRunner()
```

This means the pipeline components run as local subprocesses inside the WSL environment.

The pipeline successfully completed locally.

Example successful metrics:

```text
Accuracy  : 0.8055
Precision : 0.6572
Recall    : 0.5588
F1 Score  : 0.6040
```

The resulting model artifacts were copied into:

```text
model/model.pkl
model/preprocessor.pkl
```

---

# 11. FastAPI Model Serving

The trained model is exposed through a FastAPI application.

Location:

```text
api/main.py
```

The API loads:

```text
model/model.pkl
model/preprocessor.pkl
```

when the application starts.

### Root endpoint

```http
GET /
```

Returns:

```json
{
  "message": "Customer Churn Prediction API",
  "status": "healthy"
}
```

### Health endpoint

```http
GET /health
```

Returns information about:

* API health
* Model availability
* Preprocessor availability
* Pod hostname

Example:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "preprocessor_loaded": true,
  "pod": "churn-api-..."
}
```

The hostname is particularly useful when running multiple Kubernetes replicas because it allows us to see which pod handled a request.

### Model information

```http
GET /model-info
```

Returns information about the loaded model and preprocessor.

### Prediction endpoint

```http
POST /predict
```

The endpoint accepts customer information and returns:

```json
{
  "churn_prediction": 0,
  "churn": "No",
  "churn_probability": 0.3128
}
```

---

# 12. API Testing

Automated API tests are located in:

```text
tests/test_api.py
```

The tests cover the FastAPI application and prediction functionality.

The local test suite successfully produced:

```text
5 passed
```

These tests are also executed automatically by GitHub Actions.

---

# 13. Docker

The FastAPI application is containerised using Docker.

The Dockerfile uses:

```dockerfile
FROM python:3.11-slim
```

The container contains:

```text
api/
model/
requirements.txt
```

The application runs using Uvicorn:

```text
0.0.0.0:8000
```

The Docker image contains the trained model and preprocessing artifact, so the API can serve predictions without retraining the model.

---

# 14. GitHub Actions CI

The project uses GitHub Actions for continuous integration.

Workflow:

```text
Git Push
   ↓
GitHub Actions
   ↓
Install Python 3.11
   ↓
Install dependencies
   ↓
Run pytest
   ↓
Build Docker image
   ↓
Push image to GHCR
```

The workflow file is:

```text
.github/workflows/ci.yml
```

The workflow runs on:

```text
push → main
pull request → main
```

---

# 15. Continuous Integration Results

GitHub Actions successfully runs the test suite.

Current result:

```text
5 passed
```

There is a non-failing deprecation warning from a dependency:

```text
DeprecationWarning:
anyio.abc.BlockingPortal alias is deprecated
```

This does not affect the application or test result.

---

# 16. GitHub Container Registry

Docker images are published to **GitHub Container Registry (GHCR)**.

Image:

```text
ghcr.io/sreelakshmi-589/churn-api
```

Two tags are created:

```text
latest
<Git commit SHA>
```

For example:

```text
ghcr.io/sreelakshmi-589/churn-api:<commit-sha>
```

Using the Git commit SHA gives every image a unique version.

This is better than relying only on:

```text
latest
```

because a specific deployment can always be traced back to the exact source-code commit that produced the image.

The `latest` tag is also maintained for convenience.

---

# 17. Private GHCR Image

The GHCR package is private.

GitHub Actions uses:

```text
GITHUB_TOKEN
```

to authenticate and publish the image.

No separate GHCR password is required for the GitHub Actions workflow.

For Kubernetes, a GitHub Personal Access Token with package read permission was used to create a Kubernetes image-pull secret:

```text
ghcr-secret
```

The secret allows Kubernetes to pull the private Docker image from GHCR.

The token itself is not stored in the repository.

---

# 18. Kubernetes Deployment

The application was deployed to Kubernetes using **Minikube**.

Minikube version:

```text
v1.39.0
```

Kubernetes version:

```text
v1.37.0
```

The cluster uses the Docker driver.

The Kubernetes configuration is located in:

```text
k8s/
├── deployment.yaml
└── service.yaml
```

---

# 19. Kubernetes Deployment

The application runs with:

```text
2 replicas
```

Architecture:

```text
             Kubernetes Service
                    │
           ┌────────┴────────┐
           ▼                 ▼
      churn-api Pod     churn-api Pod
```

The Deployment includes:

### Resource requests

```text
CPU:    100m
Memory: 256Mi
```

### Resource limits

```text
CPU:    500m
Memory: 512Mi
```

### Readiness probe

The application exposes:

```text
/health
```

Kubernetes uses this endpoint to determine whether a pod is ready to receive traffic.

---

# 20. Kubernetes Service

The API is exposed using a Kubernetes:

```text
NodePort
```

Service.

The service listens on:

```text
8000
```

and forwards traffic to the API container on:

```text
8000
```

The service provides a stable endpoint in front of the two pods.

---

# 21. Kubernetes Load Balancing Test

The `/health` endpoint was called repeatedly through the Kubernetes Service.

The requests were observed reaching both API pods.

This confirmed that the Kubernetes Service was distributing traffic between the replicas.

This demonstrated:

```text
Service
   ↓
Pod 1
Pod 2
```

rather than sending every request to a single pod.

---

# 22. Kubernetes Self-Healing

Kubernetes self-healing was also tested.

A running pod was deleted manually.

Kubernetes automatically created a replacement pod to maintain the desired replica count.

For example:

```text
Desired replicas: 2

Pod 1 → deleted
Pod 2 → running

Kubernetes
    ↓
Creates replacement Pod

Final state:
Pod 1 → running
Pod 2 → running
```

This demonstrates one of the core benefits of Kubernetes: maintaining the desired state of the application.

---

# 23. Docker and Kubernetes Deployment Challenges

Several practical problems were encountered during development.

## Challenge 1 — Windows filesystem and KFP/Docker mounts

Local KFP Docker execution encountered Windows path and Docker volume-mount issues.

### Solution

Move the project into the WSL Linux filesystem:

```text
~/projects/churn_kfp_mlops
```

and use:

```python
local.SubprocessRunner()
```

for local pipeline execution.

---

## Challenge 2 — `TotalCharges` contained blank values

The dataset reported no traditional `NaN` values in the column, but some values were blank strings.

### Solution

Convert the column explicitly:

```python
pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)
```

The resulting missing values are handled by the preprocessing pipeline.

---

## Challenge 3 — Feature mismatch during model serving

An earlier version of the API produced a feature mismatch because the API input did not match the feature structure expected by the trained model.

The model expected:

```text
46 features
```

while the API was initially providing a different structure.

### Solution

The same saved preprocessing pipeline used during training is now loaded by the API.

```text
API input
    ↓
Saved preprocessor
    ↓
46 model features
    ↓
Logistic Regression
    ↓
Prediction
```

This keeps training and inference preprocessing consistent.

---

## Challenge 4 — GitHub Actions dependency failure

The CI pipeline initially failed because a required dependency for FastAPI testing was missing.

### Solution

The dependency was added to:

```text
requirements.txt
```

After the change:

```text
5 tests passed
```

in GitHub Actions.

---

## Challenge 5 — GHCR image naming

The first GHCR publishing attempt failed because the Docker image repository name generated from the GitHub owner contained uppercase characters.

Docker registry naming requires lowercase repository names.

### Solution

The GHCR path was changed to:

```text
ghcr.io/sreelakshmi-589/churn-api
```

The image then built and pushed successfully.

---

# 24. Current Project Architecture

The current completed architecture is:

```text
                    GitHub Repository
                           │
                           ▼
                  GitHub Actions
                           │
                    ┌──────┴──────┐
                    ▼             ▼
                  pytest      Docker Build
                                  │
                                  ▼
                                GHCR
                                  │
                                  ▼
                              Minikube
                                  │
                         Kubernetes Service
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
              churn-api Pod               churn-api Pod
                    │                           │
                    └─────────────┬─────────────┘
                                  ▼
                              FastAPI
                                  │
                                  ▼
                           ML Model + Preprocessor
```

---

# 25. Project Structure

```text
churn_kfp_mlops/
│
├── data/
│   └── churn.csv
│
├── pipeline/
│   ├── pipeline.py
│   └── components/
│       ├── load_data.py
│       ├── validate.py
│       ├── preprocess.py
│       ├── train.py
│       ├── evaluate.py
│       └── quality_gate.py
│
├── src/
│   ├── data_validation.py
│   └── preprocessing.py
│
├── api/
│   ├── __init__.py
│   └── main.py
│
├── model/
│   ├── model.pkl
│   ├── preprocessor.pkl
│   └── processed/
│       ├── preprocessor.pkl
│       ├── train.pkl
│       └── test.pkl
│
├── tests/
│   ├── __init__.py
│   └── test_api.py
│
├── k8s/
│   ├── deployment.yaml
│   └── service.yaml
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── .dockerignore
├── .gitignore
├── Dockerfile
└── requirements.txt
```

---

# 26. Technologies Used

### Machine Learning

* Python
* Pandas
* Scikit-learn
* Logistic Regression
* Joblib

### MLOps

* Kubeflow Pipelines
* Model artifacts
* Data validation
* Model evaluation
* Quality gates

### API

* FastAPI
* Uvicorn
* Pydantic

### Containers

* Docker
* Docker Desktop

### Kubernetes

* Kubernetes
* Minikube
* kubectl

### CI/CD

* Git
* GitHub
* GitHub Actions

### Container Registry

* GitHub Container Registry (GHCR)

### Development Environment

* Windows 11
* WSL2
* Ubuntu

---

# 27. Why This Project Matters

A machine learning model by itself is not enough for a production ML system.

This project demonstrates how a model can move through multiple stages:

```text
Data
 ↓
Validation
 ↓
Preprocessing
 ↓
Training
 ↓
Evaluation
 ↓
Quality Gate
 ↓
Model Artifact
 ↓
API
 ↓
Docker
 ↓
CI
 ↓
Container Registry
 ↓
Kubernetes
```

This gives practical experience with both **machine learning engineering and platform/MLOps engineering**.

---

# 28. Next Steps

The next stage is to connect the existing CI/CD pipeline to Kubernetes automatically.

Currently:

```text
Git push
   ↓
GitHub Actions
   ↓
Tests
   ↓
Docker build
   ↓
GHCR
```

The planned next stage is:

```text
Git push
   ↓
GitHub Actions
   ↓
Tests
   ↓
Docker build
   ↓
GHCR
   ↓
Kubernetes deployment
   ↓
Rolling update
   ↓
Application
```

A self-hosted GitHub Actions runner was considered because the current Kubernetes cluster is running locally in Minikube.

The self-hosted runner would run inside the WSL Ubuntu environment on the development laptop, allowing GitHub Actions to communicate with the local Minikube cluster.

However, this has **not yet been completed**.

---

# 29. Future Cloud Deployment

The longer-term goal is to move the Kubernetes deployment from Minikube to **Amazon EKS**.

The intended architecture is:

```text
GitHub
   ↓
GitHub Actions
   ↓
Tests
   ↓
Docker Build
   ↓
GHCR
   ↓
AWS IAM / OIDC
   ↓
Amazon EKS
   ↓
Kubernetes Deployment
   ↓
churn-api
```

EKS was not created because the current AWS account/free-tier situation does not currently provide a suitable environment for safely creating an EKS cluster.

The local Minikube implementation therefore provides the Kubernetes environment for development and learning until cloud deployment is possible.

---

# 30. Lessons Learned

This project has provided practical experience in:

* Designing an ML pipeline
* Building reusable pipeline components
* Data validation
* Preventing data leakage
* Model evaluation
* Model quality gates
* Saving and loading ML artifacts
* Serving ML models with FastAPI
* Docker containerisation
* GitHub Actions CI/CD
* GitHub Container Registry
* Kubernetes Deployments
* Kubernetes Services
* Kubernetes probes
* Kubernetes resource management
* Kubernetes replicas
* Kubernetes self-healing
* Private container registries
* Linux development using WSL
* Troubleshooting Docker/Kubernetes integration

The project also demonstrates an important engineering lesson:

> MLOps is not only about training a model. It is about building a reliable system around the model.
