import pandas as pd
from sklearn.preprocessing import StandardScaler

def add_features(df):
    # Tenure buckets
    df['tenure_group'] = pd.cut(df['tenure'], bins=[0, 12, 24, 36, 48, 60, 72], labels=False)
    # Monthly/Total Ratio
    df['monthly_total_ratio'] = df['MonthlyCharges'] / (df['TotalCharges'] + 1)
    return df

def scale_numerical(df):
    scaler = StandardScaler()
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'monthly_total_ratio']
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df

if __name__ == "__main__":
    df = pd.read_csv('data/processed/cleaned_telco.csv')
    df = add_features(df)
    df = scale_numerical(df)
    df.to_csv('data/processed/featured_telco.csv', index=False)
