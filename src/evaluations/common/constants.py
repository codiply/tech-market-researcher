"""Evaluation-wide constants."""

from enum import Enum

DEFAULT_AGENT_CONFIG_NAME = "dev"
DEFAULT_EVALS_CONFIG_NAME = "main"

EVALS_USER_ID = "evals"
EVALS_ENVIRONMENT = "evals"
EVALS_SESSION_ID_PREFIX = "eval-"
EVALS_RUN_NAME_PREFIX = "evals-"


class EvalsPlatform(str, Enum):
    """Supported evaluation platforms."""

    ARIZE = "arize"
    MLLFLOW = "mlflow"


DEFAULT_EVALS_PLATFORM = EvalsPlatform.MLLFLOW
