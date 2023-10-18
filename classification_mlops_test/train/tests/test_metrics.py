import numpy as np

from src.utils.metrics import compute_metrics


def test_compute_metrics():
    class data:
        def __init__(self):
            self.label_ids = np.array([0, 0, 0, 0, 0, 0])
            self.predictions = np.array(
                [
                    [1],
                    [1],
                    [1],
                    [1],
                    [1],
                    [1],
                ]
            )

    res = compute_metrics(data())
    print(res["accuracy"], res["f1"], res["precision"], res["recall"])

    assert res["accuracy"] == 1.0
