import pandas as pd

from data_preprocessing import encode_categorical, load_and_clean_data


def test_load_and_clean_data_drops_duplicates_and_blanks(sample_raw_df, tmp_path):
    csv_path = tmp_path / "raw.csv"
    sample_raw_df.to_csv(csv_path, index=False)

    df = load_and_clean_data(input_path=str(csv_path))

    # 5 raw rows -> 1 blank-TotalCharges row dropped, 1 exact duplicate collapsed
    assert len(df) == 3
    assert "customerID" not in df.columns
    assert pd.api.types.is_numeric_dtype(df["TotalCharges"])


def test_encode_categorical_fits_and_reuses_encoders(sample_raw_df, tmp_path):
    csv_path = tmp_path / "raw.csv"
    sample_raw_df.to_csv(csv_path, index=False)
    df = load_and_clean_data(input_path=str(csv_path))

    encoded_df, encoders = encode_categorical(df.copy())

    assert set(encoders.keys()) == set(
        df.select_dtypes(include="object").columns
    ) - {"Churn"}
    assert encoded_df["Churn"].isin([0, 1]).all()
    for col in encoders:
        assert pd.api.types.is_integer_dtype(encoded_df[col])

    # reusing fitted encoders (inference path) must reproduce identical encodings,
    # not refit new ones
    reencoded_df, _ = encode_categorical(df.copy(), encoders=encoders)
    for col in encoders:
        assert (reencoded_df[col] == encoded_df[col]).all()
