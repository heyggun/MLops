import os
import pytest
from os.path import join, abspath, dirname

from src.utils.data_loading import download_data, load_data, data_split

resource_dir = join(dirname(abspath(__file__)), "resources")


@pytest.mark.xfail(reason="no way of currently testing this externally")
def test_download_data(tmpdir):
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://61.80.148.154:30001"
    os.environ["AWS_ACCESS_KEY_ID"] = "minio"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"

    tmp = join(tmpdir, "test.csv")
    download_data("test", "op/test.csv", tmp)


def test_data_split():
    # todo: write test code
    pass


def test_load_data():
    # todo: write test code
    xlsx_file = join(resource_dir, "test_xlsx.xlsx")
    csv_file = join(resource_dir, "test_csv.csv")
    pkl_file = join(resource_dir, "test_pkl.pkl")
