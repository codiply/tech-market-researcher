"""Evaluation-wide constants."""

from enum import Enum

DEFAULT_AGENT_CONFIG_NAME = "dev"
DEFAULT_EVALS_CONFIG_NAME = "main"


class EvalsPlatform(str, Enum):
    """Supported evaluation platforms."""

    ARIZE = "arize"
    MLLFLOW = "mlflow"


DEFAULT_EVALS_PLATFORM = EvalsPlatform.MLLFLOW
