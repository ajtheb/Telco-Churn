from fastapi.testclient import TestClient

from api.main import app

VALID_PAYLOAD = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 24,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "No",
    "DeviceProtection": "Yes",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "One year",
    "PaperlessBilling": "No",
    "PaymentMethod": "Mailed check",
    "MonthlyCharges": 42.30,
    "TotalCharges": 1840.75,
}


def test_health_ok_once_artifacts_loaded(trained_artifacts):
    app.state.artifacts_loaded = True
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_valid_payload_returns_200(trained_artifacts):
    app.state.artifacts_loaded = True
    client = TestClient(app)

    response = client.post("/predict", json=VALID_PAYLOAD)

    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {"churn_prediction", "churn_probability"}
    assert body["churn_prediction"] in (0, 1)


def test_predict_invalid_payload_returns_422(trained_artifacts):
    app.state.artifacts_loaded = True
    client = TestClient(app)

    invalid_payload = {**VALID_PAYLOAD, "gender": "Alien"}
    response = client.post("/predict", json=invalid_payload)

    assert response.status_code == 422
