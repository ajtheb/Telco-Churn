import model_inference

SAMPLE_INPUT = {
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


def test_predict_churn_returns_stable_prediction(trained_artifacts):
    result = model_inference.predict_churn(SAMPLE_INPUT)

    assert set(result.keys()) == {"churn_prediction", "churn_probability"}
    assert result["churn_prediction"] in (0, 1)
    assert 0.0 <= result["churn_probability"] <= 1.0

    # deterministic: same input -> same output
    assert model_inference.predict_churn(SAMPLE_INPUT) == result
