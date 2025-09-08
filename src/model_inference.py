import pandas as pd
import joblib

def predict_churn(input_dict):
    clf = joblib.load('models/churn_rf.pkl')
    # Example: preprocess input_dict to match trained features
    input_df = pd.DataFrame([input_dict])
    # Apply same feature engineering and scaling as training...
    # For demonstration, assuming input_df matches feature format
    prediction = clf.predict(input_df)
    probability = clf.predict_proba(input_df)[0, 1]
    return {'churn_prediction': int(prediction), 'churn_probability': float(probability)}

# Example usage:
if __name__ == "__main__":
    sample_input = {'gender': 1, 'SeniorCitizen': 0, 'Partner': 1, 'Dependents': 0, 'tenure': 24,
                    'PhoneService': 1, 'MultipleLines': 0, 'InternetService': 1, 'OnlineSecurity': 0,
                    'OnlineBackup': 1, 'DeviceProtection': 1, 'TechSupport': 0, 'StreamingTV': 1,
                    'StreamingMovies': 1, 'Contract': 1, 'PaperlessBilling': 1, 'PaymentMethod': 2,
                    'MonthlyCharges': 70.0, 'TotalCharges': 1680.0, 'tenure_group': 1, 'monthly_total_ratio': 0.04}
    print(predict_churn(sample_input))
