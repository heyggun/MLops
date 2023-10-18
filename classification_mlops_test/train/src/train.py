import os
import torch
import mlflow
import logging
import argparse
from os.path import dirname, abspath, join

from transformers import (
    Trainer,
    AutoTokenizer,
    TrainingArguments,
    AutoModelForSequenceClassification,
)

from utils.data_loading import BasicDataset, data_split, download_data
from utils.metrics import compute_metrics

# todo: write mlflow, minio connection error handling exception
# MLFLOW_URI = os.environ["MLFLOW_URI"]
# MLFLOW_URI = f"http://mlflow-server-service.mlflow-system.svc.cluster.local:5000"
MLFLOW_URI = "http://61.80.148.154:30002"
# Set mlflow tracking server
mlflow.set_tracking_uri(MLFLOW_URI)

# Set minio credential
os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://61.80.148.154:30001"
# os.environ[
#     "MLFLOW_S3_ENDPOINT_URL"
# ] = f"http://minio-service.kubeflow.svc.cluster.local:9000"


def train_model(
    device,
    experiment,
    bucket_name,
    object_name,
    data_version: str = "No version",
    tag: str = "No tag",
    data_name: str = "train.csv",
    model_name: str = "beomi/KcELECTRA-base-v2022",
    epochs: int = 5,
    batch_size: int = 1,
    learning_rate: float = 1e-5,
    val_percent: float = 0.2,
    save_checkpoint: bool = False,
    weight_decay: float = 1e-8,
    momentum: float = 0.999,
    gradient_clipping: float = 1.0,
    amp: bool = False,
    stage: str = "build",
):
    # 0. Load pretrained model and data
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Define the components of the model in a dictionary
    artifact = {"model": model, "tokenizer": tokenizer}
    # Load data from storage
    train_data_path = join(dirname(abspath(__file__)), "data", data_name)
    download_data(bucket_name, object_name, train_data_path)

    # 1. Split into train / validation
    # Data should fit in the memory
    splited = data_split(train_data_path, val_percent)

    # 2. Create dataset
    train_set = BasicDataset(splited[0], artifact["tokenizer"])
    val_set = BasicDataset(splited[1], artifact["tokenizer"])

    logging.info(
        f"""Starting training:
            Epochs:          {epochs}
            Batch size:      {batch_size}
            Learning rate:   {learning_rate}
            Training size:   {len(train_set)}
            Validation size: {len(val_set)}
            Checkpoints:     {save_checkpoint}
            Device:          {device.type}
            Mixed Precision: {amp}
        """
    )

    mlflow.set_experiment(experiment)
    with mlflow.start_run():
        mlflow.set_tags(
            {
                "tag": tag,
                "purpose": "Spam Classification",
                "data.version": data_version,
                "stage": stage,
            }
        )
        mlflow.transformers.autolog()
        # 5. Begin training
        trainer = Trainer(
            model=artifact["model"],
            train_dataset=train_set,
            eval_dataset=val_set,
            compute_metrics=compute_metrics,
            # 4. Set up the optimizer, the loss, the learning rate scheduler
            args=TrainingArguments(
                output_dir="./",
                no_cuda=True if device == torch.device("cpu") else False,
                use_mps_device=True if device == torch.device("mps") else False,
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=(len(train_set) // (5 * batch_size)),
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                save_strategy="yes" if save_checkpoint else "no",
                # lr_scheduler_type="reduce_lr_on_plateau",
                dataloader_drop_last=True,
                dataloader_num_workers=os.cpu_count(),
                evaluation_strategy="steps",
            ),
        )
        trainer.train()

        try:
            model_info = mlflow.transformers.log_model(
                transformers_model=artifact,
                task="text-classification",
                registered_model_name="ElectraClassification",
                artifact_path=experiment,
            )
            logging.info(f"Model logged successfully {model_info}")

        except Exception as e:
            # todo: save models(vectorizer and electra) at minio
            logging.error(
                f"{e}, Failed to log model in mlflow. Saving models locally..."
            )
        finally:
            # Remove downloaded data
            os.remove(train_data_path)

        logging.info("Finishing an experiment...")


def get_args():
    parser = argparse.ArgumentParser(description="Train the kc-electra model")

    parser.add_argument(
        "experiment", type=str, default=False, help="Set experiment name"
    )
    parser.add_argument(
        "bucket_name", type=str, default=False, help="Storage bucket name"
    )
    parser.add_argument(
        "object_name", type=str, default=False, help="A object name to load data"
    )
    parser.add_argument("--epochs", "-e", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch-size", "-b", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--learning-rate",
        "-l",
        metavar="LR",
        type=float,
        default=1e-5,
        help="Learning rate",
        dest="lr",
    )
    parser.add_argument(
        "--validation",
        "-v",
        dest="val",
        type=float,
        default=20.0,
        help="Percent of the data that is used as validation (0-100)",
    )
    parser.add_argument(
        "--model-name",
        "-m",
        dest="mn",
        type=str,
        default="beomi/KcELECTRA-base-v2022",
        help="Pretrained model name to load",
    )
    parser.add_argument(
        "--is-mps", type=int, default=0, help="1 if you are currently using mps"
    )
    parser.add_argument(
        "--tag", "-t", type=str, default="No tag", help="Setting a tag for mlflow"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.is_mps:
        device = torch.device("mps")

    logging.info(f"Using device {device}")

    try:
        train_model(
            device=device,
            experiment=args.experiment,
            bucket_name=args.bucket_name,
            object_name=args.object_name,
            model_name=args.mn,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            val_percent=args.val / 100,
            tag=args.tag,
        )
    except torch.cuda.OutOfMemoryError:
        logging.error(
            "Detected OutOfMemoryError! "
            "Enabling checkpointing to reduce memory usage, but this slows down training. "
            "Consider enabling AMP (--amp) for fast and memory efficient training"
        )
        torch.cuda.empty_cache()
        train_model(
            device=torch.device("cpu"),
            experiment=args.experiment,
            bucket_name=args.bucket_name,
            object_name=args.object_name,
            model_name=args.mn,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            val_percent=args.val / 100,
            tag=args.tag,
        )
