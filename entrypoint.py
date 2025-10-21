import os
import mlflow
from vllm import LLM, SamplingParams
from flask import Flask, request, jsonify

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server-0.mlflow-model.svc.cluster.local:5000")
MODEL_NAMES = os.getenv("MODEL_NAMES", "invoice-model,po-model").split(",")
LOCAL_BASE = "/models"

# Configure MLflow to use your tracking server and MinIO
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
os.makedirs(LOCAL_BASE, exist_ok=True)

print("🎯 Using MLflow server:", MLFLOW_TRACKING_URI)
print("📦 Models to load:", MODEL_NAMES)

# Download all models registered under MLflow Registry prefix
from mlflow import MlflowClient
client = MlflowClient()

loaded_models = {}

for name in MODEL_NAMES:
    registry_name = f"vllm-{name.strip()}"
    try:
        # Get latest version from registry
        versions = client.search_model_versions(f"name='{registry_name}'")
        latest = sorted(versions, key=lambda v: int(v.version))[-1]
        run_id = latest.run_id
        print(f"⬇️  Downloading {registry_name} (run_id={run_id}) ...")
        path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=".", dst_path=f"{LOCAL_BASE}/{name}")
        print(f"✅ Downloaded {name} → {path}")
        loaded_models[name] = LLM(model=path)
    except Exception as e:
        print(f"⚠️ Failed to load {registry_name}: {e}")

# Flask inference API
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    model_name = data.get("model")
    prompt = data.get("prompt", "")
    if model_name not in loaded_models:
        return jsonify({"error": f"Model '{model_name}' not found", "available": list(loaded_models.keys())}), 404

    llm = loaded_models[model_name]
    params = SamplingParams(max_tokens=256)
    result = llm.generate(prompt, sampling_params=params)
    return jsonify({"model": model_name, "response": result[0].outputs[0].text})

@app.route("/models", methods=["GET"])
def list_models():
    return jsonify({"available_models": list(loaded_models.keys())})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9000)
