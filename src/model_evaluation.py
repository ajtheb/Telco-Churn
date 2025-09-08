import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

def evaluate_model():
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_test = pd.read_csv('data/processed/y_test.csv')
    clf = joblib.load('models/churn_rf.pkl')
    y_pred = clf.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    y_proba = clf.predict_proba(X_test)[:, 1]
    print("ROC-AUC Score:", roc_auc_score(y_test, y_proba))

if __name__ == "__main__":
    evaluate_model()
