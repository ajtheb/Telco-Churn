import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_and_clean_data(
        input_path="data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
        ):
    df1 = pd.read_csv(input_path)
    df1 = df1.dropna(how="all")
    df1 = df1[~df1.duplicated()]
    # Remove rows where TotalCharges is blank and fix type
    df1 = df1[df1["TotalCharges"] != " "]
    df1["TotalCharges"] = pd.to_numeric(df1["TotalCharges"])
    df1.drop(["customerID"], axis=1, inplace=True)
    return df1


def encode_categorical(df1):
    for col in df1.select_dtypes(include="object").columns:
        if col != "Churn":
            df1[col] = LabelEncoder().fit_transform(df1[col])
    df1["Churn"] = df1["Churn"].map({"Yes": 1, "No": 0})
    return df1


if __name__ == "__main__":
    df = load_and_clean_data()
    df = encode_categorical(df)
    df.to_csv("data/processed/cleaned_telco.csv", index=False)
