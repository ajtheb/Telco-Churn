# Churn Prediction

A machine learning pipeline and API for predicting customer churn on the Telco customer churn dataset. The project covers the full lifecycle from raw data to a served model: cleaning and feature engineering, model training and evaluation with MLflow experiment tracking, and a FastAPI service that serves real-time churn predictions.

## Features

- **Data pipeline** ([src/data_preprocessing.py](src/data_preprocessing.py), [src/feature_engineering.py](src/feature_engineering.py)) — cleans raw customer data, encodes categorical fields, and derives numerical features (e.g. `monthly_total_ratio`), scaling them with a persisted `StandardScaler`.
- **Model training & evaluation** ([src/model_training.py](src/model_training.py), [src/model_evaluation.py](src/model_evaluation.py)) — trains a `RandomForestClassifier`, logs parameters/metrics/artifacts to MLflow, and reports accuracy, F1, and ROC-AUC.
- **Model registry** ([src/model_register.py](src/model_register.py)) — registers a trained run's model into the MLflow Model Registry.
- **Inference** ([src/model_inference.py](src/model_inference.py)) — loads the trained model plus its scaler/encoders/feature columns and produces churn predictions for new customer records.
- **REST API** ([src/api/main.py](src/api/main.py)) — a FastAPI app exposing:
  - `POST /predict` — returns `churn_prediction` and `churn_probability` for a customer profile (see [src/api/schemas.py](src/api/schemas.py) for the request schema).
  - `GET /health` — reports whether model artifacts loaded successfully.
  - `GET /metrics` — Prometheus metrics (request counts, latency, prediction distribution).
- **Containerized deployment** — [Dockerfile](Dockerfile) and [render.yaml](render.yaml) for running the API as a Docker service (e.g. on Render).

## Project layout

```
src/
  data_preprocessing.py   # load + clean raw data, encode categoricals
  feature_engineering.py  # derive + scale numerical features
  model_training.py       # train RandomForest, log to MLflow
  model_evaluation.py     # evaluate trained model on holdout set
  model_register.py       # register a run's model in MLflow
  model_inference.py      # load artifacts and predict on new data
  config.py               # paths, MLflow settings, env-driven config
  api/
    main.py               # FastAPI app (/predict, /health, /metrics)
    schemas.py             # request/response pydantic models
data/                     # raw and processed datasets
models/                   # trained model, scaler, encoders, feature columns
mlruns/                   # MLflow tracking data
notebooks/                # exploratory analysis
tests/                    # pytest test suite
```

## Getting started

```bash
pip install -r requirements.txt
```

Run the pipeline stages (from the repo root):

```bash
python src/data_preprocessing.py
python src/feature_engineering.py
python src/model_training.py
python src/model_evaluation.py
```

Serve the API locally:

```bash
uvicorn src.api.main:app --reload
```

Run tests:

```bash
pytest
```
