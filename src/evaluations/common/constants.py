"""Evaluation-wide constants."""

from enum import Enum

DEFAULT_AGENT_CONFIG_NAME = "dev"
DEFAULT_EVALS_CONFIG_NAME = "main"

EVALS_USER_ID = "evals"
EVALS_ENVIRONMENT = "evals"
# Names of the agent config mixins applied when running evals, from config/agent/mixins/. The
# shared evals mixin is always applied; the platform mixin adds the platform's own backend.
EVALS_AGENT_MIXIN = "evals"
EVALS_MLFLOW_AGENT_MIXIN = "evals-mlflow"
EVALS_PHOENIX_AGENT_MIXIN = "evals-phoenix"
EVALS_SESSION_ID_PREFIX = "eval-"
EVALS_RUN_NAME_PREFIX = "evals-"


class EvalsPlatform(str, Enum):
    """Supported evaluation platforms."""

    PHOENIX = "phoenix"
    MLLFLOW = "mlflow"


DEFAULT_EVALS_PLATFORM = EvalsPlatform.MLLFLOW
