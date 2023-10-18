import os
import torch
import logging
import pandas as pd
from os import listdir
from minio import Minio
from torch.utils.data import Dataset
from os.path import join, isfile, splitext, dirname, abspath


def download_data(bucket_name, object_name, output_dir):
    mino_endpoint = os.environ.get(
        "MLFLOW_S3_ENDPOINT_URL", "http://61.80.148.154:30001"
    ).split("//")[1]
    access_key = os.environ["AWS_ACCESS_KEY_ID"]
    secret_key = os.environ["AWS_SECRET_ACCESS_KEY"]

    logging.info(f"Downloading data from {mino_endpoint}:{bucket_name}/{object_name}")

    try:
        client = Minio(
            mino_endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=False,
        )
        client.fget_object(
            bucket_name=bucket_name,
            object_name=object_name,
            file_path=output_dir,
        )
    except Exception as e:
        logging.error(f"{e}, trying with external address")

        client = Minio(
            "61.80.148.154:30001",
            access_key="minio",
            secret_key="minio123",
            secure=False,
        )

        client.fget_object(
            bucket_name=bucket_name,
            object_name=object_name,
            file_path=output_dir,
        )


def data_split(train_data_path, n_val):
    data_dir = dirname(abspath(train_data_path))
    data = pd.concat(
        [
            load_data(join(data_dir, file))
            for file in listdir(data_dir)
            if isfile(join(data_dir, file)) and not file.startswith(".")
        ],
    ).reset_index(drop=True)

    train_data = data.sample(frac=1 - n_val).reset_index(drop=True)
    val_data = data.drop(train_data.index).reset_index(drop=True)

    return train_data, val_data


def load_data(file_path, is_large=False):
    logging.info(f"Loading data: {file_path}")
    ext = splitext(file_path)[1]

    supports = [
        ".xlsx",
        ".csv",
        ".pkl",
    ]
    if not is_large:
        if ext == supports[0]:
            return pd.read_excel(file_path)
        elif ext == supports[1]:
            return pd.read_csv(file_path)
        elif ext == supports[2]:
            return pd.read_pickle(file_path)
        else:
            raise TypeError(
                f"{ext} is unsupported file type, Supports only {' '.join(supports)}"
            )

    else:
        # todo : Add object that reads line by line from largefile
        raise TypeError("'is_large' should remain 'False' due to incomplete function")


class BasicDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.token = self.tokenize(list(data.iloc[:, 0]), tokenizer)
        self.label = data.iloc[:, 1].values

    @staticmethod
    def tokenize(text_list, tokenizer):
        tokenized = tokenizer(
            text_list,
            max_length=128,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=True,
        )

        return tokenized

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        res = {k: torch.tensor(v[idx]) for k, v in self.token.items()}
        res["label"] = torch.tensor(self.label[idx])
        return res
