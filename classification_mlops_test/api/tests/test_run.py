import pytest
import mlflow
from os.path import join, abspath, dirname
from fastapi.testclient import TestClient

from src.run import app

client = TestClient(app)


def test_health_check():
    response = client.get("/_service_health_check")

    assert response.status_code == 200
    assert response.json() == {"msg": "Health Check"}


# todo: find the way how to load model from blob, currently it is unabled
def test_conts_filter():
    response = client.post(
        "/conts_filter",
        headers={"X-Token": "coneofsilence"},
        json={
            "UserInfo": [
                {
                    "autoNo": 11234,
                    "memNo": 2259011,
                    "mateConts": "안녕하세여안녕010안녕하1234세여안1234녕하세여안녕하세하세여안녕하세여안녕하세여안녕하세여안녕하세여5678여안녕하세여안녕하세여안녕하세여안녕하세여안녕하세여안녕하세여안녕하세여안녕하세여안녕하세여안녕하세여안녕하세여안녕하세여안녕하세여안녕하세여안녕하세여안녕하세여안녕하세여안녕하세여안녕하세여안녕하세여안녕하세여안녕하세여",
                    "familyConts": "안녕하세여안녕하세여안녕하세여안녕하세여안녕하세여안녕하세여안녕하세여안녕하세여안녕하세여안녕하세여안녕하세여안녕하세여안녕하세여안녕하세여안녕하세여안녕하세여안녕하세여안녕하세여안녕하세여안녕하세여안녕하세여안녕하세여안녕하세여안녕하세여안녕하세여안녕하세여안녕하세여안녕하세여안녕하세여안녕하세여안녕하세여안녕하세여안녕하세여안녕하세여안녕하세여안녕하세여",
                },
                {
                    "autoNo": 10290,
                    "memNo": 9999,
                    "mateConts": "오르텅스블루열심이 일해서 연인을 만나도 될거같은 스펙 만들었습니다. ",
                    "familyConts": "아빠, 엄마, 나, 동생 이렇게 4인 가족입니다. 다알아서 잘살고 있습니다. 동생은 장가가서 잘살고 있고 엄마빠가 이제 너만 가면 더이상 바랄게 없다고 하십니다. ",
                },
            ],
        },
    )

    res_code = response.status_code
    res = response.json()
    expected_output = {
        "autoNo": [11234, 10290],
        "memNo": [2259011, 9999],
        "matePred": [[1, 1], [0, 0]],
        "familyPred": [[1, 0], [0, 0]],
    }

    assert res_code == 200
    assert isinstance(res, list)
    for i in range(len(res)):
        assert isinstance(res[i], dict)
        assert res[i]["autoNo"] == expected_output["autoNo"][i]
        assert res[i]["memNo"] == expected_output["memNo"][i]
        assert res[i]["matePred"] == expected_output["matePred"][i]
        assert res[i]["familyPred"] == expected_output["familyPred"][i]
