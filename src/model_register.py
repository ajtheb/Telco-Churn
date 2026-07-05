import logging
import os

import mlflow

logger = logging.getLogger(__name__)

run_id = os.environ["RUN_ID"]
model_name = "CustomerChurnModel"
result = mlflow.register_model(f"runs:/{run_id}/model", model_name)
logger.info("Model registered with name: %s", result.name)
