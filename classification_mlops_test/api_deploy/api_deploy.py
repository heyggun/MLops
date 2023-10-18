import os
import mlflow
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# blob storage credentialapp
os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = f"http://61.80.148.154:30001"

app = FastAPI()

MODEL_PATH = os.environ.get("MODEL_PATH", "/home/yeoai/ai_data/ai_pn_classification_data")


class Artifact(BaseModel):
    artifact_uri: str | None
    run_id: str | None
    artifact_path: str | None
    tracking_uri: str | None


@app.post("/download_artifact")
async def get_artifact(artifact_uri: Artifact):
    data = artifact_uri.dict()

    if not data["artifact_uri"] and not (data["run_id"] and data["artifact_path"]):
        return -1

    try:
        mlflow.artifacts.download_artifacts(
            artifact_uri=data["artifact_uri"],
            dst_path=MODEL_PATH,
        )

    except Exception as e:
        print(e)

    return 0

#
if __name__ == '__main__':
    uvicorn.run("api_deploy:app", host= "0.0.0.0", port=8888)
