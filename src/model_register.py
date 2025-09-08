import mlflow
import os

run_id = os.environ["RUN_ID"]
model_name = "CustomerChurnModel"
result = mlflow.register_model(f"runs:/{run_id}/model", model_name)
print("Model registered with name:", result.name)
