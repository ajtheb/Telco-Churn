import logging
import os
import sys
import time

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, make_asgi_app

# src/ holds bare, non-package-qualified modules (config, model_inference, ...)
# imported the same way the rest of this codebase already does (see
# tests/test_model_pipeline.py), so put it on sys.path here too.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model_inference import load_artifacts, predict_churn  # noqa: E402

from api.schemas import ChurnPredictionRequest, ChurnPredictionResponse  # noqa: E402

logger = logging.getLogger(__name__)

app = FastAPI(title="Churn Prediction API")

REQUEST_COUNT = Counter(
    "churn_api_requests_total", "Total prediction requests", ["status"]
)
REQUEST_LATENCY = Histogram(
    "churn_api_request_latency_seconds", "Prediction request latency in seconds"
)
PREDICTION_COUNT = Counter(
    "churn_api_predictions_total", "Predicted churn outcomes", ["prediction"]
)

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


@app.on_event("startup")
def startup():
    try:
        load_artifacts()
        app.state.artifacts_loaded = True
    except Exception:
        logger.exception("Failed to load model artifacts at startup")
        app.state.artifacts_loaded = False


@app.get("/health")
def health():
    if getattr(app.state, "artifacts_loaded", False):
        return {"status": "ok"}
    return JSONResponse(status_code=503, content={"status": "model not loaded"})


@app.post("/predict", response_model=ChurnPredictionResponse)
def predict(request: ChurnPredictionRequest):
    start = time.time()
    try:
        result = predict_churn(request.dict())
    except Exception:
        REQUEST_COUNT.labels(status="error").inc()
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed")

    latency = time.time() - start
    REQUEST_LATENCY.observe(latency)
    REQUEST_COUNT.labels(status="success").inc()
    PREDICTION_COUNT.labels(prediction=str(result["churn_prediction"])).inc()
    logger.info(
        "Prediction served: tenure=%s contract=%s prediction=%s probability=%.4f "
        "latency=%.4fs",
        request.tenure,
        request.Contract,
        result["churn_prediction"],
        result["churn_probability"],
        latency,
    )
    return result
