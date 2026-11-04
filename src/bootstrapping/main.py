"""Bootstrap the MLflow stack: MinIO artifact bucket, AI Gateway models, experiments and datasets."""

import logging

from dotenv import load_dotenv

from bootstrapping.common.configuration import load_config
from bootstrapping.resources.ai_gateway import bootstrap_ai_gateway
from bootstrapping.resources.mlflow_experiments import bootstrap_mlflow_experiments
from bootstrapping.resources.s3 import bootstrap_s3

load_dotenv()


def main() -> None:
    """Provision the MLflow stack from config."""
    logging.basicConfig(level=logging.INFO)
    config = load_config("main")
    bootstrap_s3(config)
    bootstrap_ai_gateway(config)
    bootstrap_mlflow_experiments(config)


if __name__ == "__main__":
    main()
