import pandas as pd
from sklearn.preprocessing import StandardScaler


def add_features(df1):
    # Tenure buckets
    df1["tenure_group"] = pd.cut(
        df1["tenure"], bins=[0, 12, 24, 36, 48, 60, 72], labels=False
    )
    # Monthly/Total Ratio
    df1["monthly_total_ratio"] = df1["MonthlyCharges"] / (df1["TotalCharges"] + 1)
    return df1


def scale_numerical(df1):
    scaler = StandardScaler()
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges", "monthly_total_ratio"]
    df1[num_cols] = scaler.fit_transform(df1[num_cols])
    return df1


if __name__ == "__main__":
    df = pd.read_csv("data/processed/cleaned_telco.csv")
    df = add_features(df)
    df = scale_numerical(df)
    df.to_csv("data/processed/featured_telco.csv", index=False)
