"""Config models. All classes here are named XxxConfig."""

from pathlib import Path

import yaml
from pydantic import BaseModel

CONFIG_DIR = Path("config") / "evaluations" / "mlflow"


class MlflowEvaluationConfig(BaseModel):
    """Evaluation settings."""

    experiment_name: str
    dataset_name: str


def load_config(name: str) -> MlflowEvaluationConfig:
    """Load the evaluation config at config/evaluations/mlflow/<name>.yaml."""
    data = yaml.safe_load((CONFIG_DIR / f"{name}.yaml").read_text())
    return MlflowEvaluationConfig.model_validate(data)
