### MLFLOW setup with juju
```
kubectl port-forward -n mlflow-model mlflow-server-0 5000:5000

kubectl patch svc mlflow-minio -n mlflow-model -p '{
  "spec": {
    "type": "NodePort",
    "ports": [
      {"port":9000,"targetPort":9000,"nodePort":32000},
      {"port":9001,"targetPort":9001,"nodePort":32001}
    ]
  }
}'




juju config mlflow-minio access-key
juju config mlflow-minio secret-key

export MLFLOW_TRACKING_URI="http://localhost:5000"
export MLFLOW_S3_ENDPOINT_URL="http://localhost:32000"
export AWS_ACCESS_KEY_ID="minio"
export AWS_SECRET_ACCESS_KEY="minioadmin"


http://localhost:32001/object-browser/mlflow/0/34f2da5ba4794d6e91ca18fa56ab1d40/artifacts
http://localhost:5000/

```