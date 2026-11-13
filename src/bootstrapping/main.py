"""Bootstrap the stack: MinIO artifact bucket, AI Gateway models, MLflow and Phoenix experiments/datasets."""

import logging

from dotenv import load_dotenv

from bootstrapping.common.configuration import load_config
from bootstrapping.resources.mlflow.ai_gateway import bootstrap_mlflow_ai_gateway
from bootstrapping.resources.mlflow.experiments import bootstrap_mlflow_experiments
from bootstrapping.resources.phoenix.datasets import bootstrap_phoenix
from bootstrapping.resources.phoenix.settings import bootstrap_phoenix_settings
from bootstrapping.resources.s3 import bootstrap_s3


def main() -> None:
    """Provision the stack's resources from config."""
    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    config = load_config("main")
    bootstrap_s3(config)
    # The gateway's artifact root is s3://mlflow-artifacts, so the bucket must exist first.
    bootstrap_mlflow_ai_gateway(config)
    bootstrap_mlflow_experiments(config)
    bootstrap_phoenix(config)
    # Phoenix providers point at the MLflow AI Gateway, so they must be set up after MLflow.
    bootstrap_phoenix_settings(config)


if __name__ == "__main__":
    main()
