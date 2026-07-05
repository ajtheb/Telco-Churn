import logging

import pandas as pd
import joblib
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

import config

logger = logging.getLogger(__name__)


def evaluate_model():
    x_test = pd.read_csv(config.X_TEST_PATH)
    y_test = pd.read_csv(config.Y_TEST_PATH)
    clf = joblib.load(config.MODEL_PATH)
    y_pred = clf.predict(x_test)
    logger.info("Confusion matrix:\n%s", confusion_matrix(y_test, y_pred))
    logger.info("Classification report:\n%s", classification_report(y_test, y_pred))
    y_proba = clf.predict_proba(x_test)[:, 1]
    logger.info("ROC-AUC Score: %s", roc_auc_score(y_test, y_proba))


if __name__ == "__main__":
    evaluate_model()
