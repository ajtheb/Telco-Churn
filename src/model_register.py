# Example: Registering model after training
import mlflow

run_id = "5a2a565e3f8e49e5bce59e503886a708"
result = mlflow.register_model(f"runs:/{run_id}/model", "CustomerChurnModel")
print("Model registered with name:", result.name)

client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name="CustomerChurnModel",
    version=1,  # Replace with your actual version number
    stage="Production",
)
