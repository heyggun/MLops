import mlflow
from fastapi.testclient import TestClient

from api_deploy import app, get_artifact

client = TestClient(app)


def test_get_artifact():
    # todo: write test
    response = client.post(
        "/download_artifact",
        headers={"X-Token": "coneofsilence"},
        json={
            "artifact_uri": "s3://mlflow/mlflow/artifacts/10/ca63521e13444d4e8b174758f988499b/artifacts/electra-fs",
            "run_id": "",
            "artifact_path": "",
            "tracking_uri": "",
        },
    )

    res_code = response.status_code
    res = response.json()

    assert res_code == 200
    assert res == 0
