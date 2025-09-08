import mlflow
import os

# You might want to get the latest run_id;
#  here we show hardcoded for simplicity
run_id = os.environ.get("MLFLOW_RUN_ID", "last_run_id_here")
model_name = os.environ.get("MLFLOW_MODEL_NAME", "CustomerChurnModel")

result = mlflow.register_model(f"runs:/{run_id}/model", model_name)
print("Model registered with name:", result.name)
