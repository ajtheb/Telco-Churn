import pandas as pd
import joblib
import pytest

import config
from model_training import train_model


@pytest.mark.integration
def test_training_pipeline():
    clf = train_model()
    assert hasattr(clf, "predict"), "Model does not have predict method"


@pytest.mark.integration
def test_model_output_shape():
    x_test = pd.read_csv(config.X_TEST_PATH)
    clf = joblib.load(config.MODEL_PATH)
    preds = clf.predict(x_test)
    assert len(preds) == len(x_test), "Prediction shape mismatch"
