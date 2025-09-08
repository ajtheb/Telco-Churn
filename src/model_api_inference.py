import mlflow
import pandas as pd

# Input your single example as a dictionary matching the model's feature names
single_input = {
    'gender': 1,
    'SeniorCitizen': 0,
    'Partner': 1,
    'Dependents': 0,
    'tenure': 24,
    'PhoneService': 1,
    'MultipleLines': 0,
    'InternetService': 1,
    'OnlineSecurity': 0,
    'OnlineBackup': 1,
    'DeviceProtection': 1,
    'TechSupport': 0,
    'StreamingTV': 1,
    'StreamingMovies': 1,
    'Contract': 1,
    'PaperlessBilling': 1,
    'PaymentMethod': 2,
    'MonthlyCharges': 70.0,
    'TotalCharges': 1680.0,
    'tenure_group': 1,
    'monthly_total_ratio': 0.04
}
test_path = 'data/processed/featured_telco.csv'
# Convert to single-row DataFrame
# input_df = pd.DataFrame([single_input])

# Load test data to ensure feature consistency
input_df = pd.read_csv(test_path)
input_df = input_df.drop(columns=['Churn'])
print(input_df.columns)
# Load the model from MLflow registry (replace <model_id> as needed)
model = mlflow.pyfunc.load_model("runs:/5a2a565e3f8e49e5bce59e503886a708/model")

# Predict for the single input
prediction = model.predict(input_df)

print("Prediction:", prediction)
