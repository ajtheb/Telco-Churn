import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_clean_data(input_path='data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv'):
    df = pd.read_csv(input_path)
    df = df.dropna(how="all")
    df = df[~df.duplicated()]
    # Remove rows where TotalCharges is blank and fix type
    df = df[df['TotalCharges'] != ' ']
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])
    df.drop(['customerID'], axis=1, inplace=True)
    return df

def encode_categorical(df):
    for col in df.select_dtypes(include='object').columns:
        if col != 'Churn':
            df[col] = LabelEncoder().fit_transform(df[col])
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    return df

if __name__ == "__main__":
    df = load_and_clean_data()
    df = encode_categorical(df)
    df.to_csv('data/processed/cleaned_telco.csv', index=False)
