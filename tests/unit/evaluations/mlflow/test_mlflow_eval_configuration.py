"""Unit tests for the MLflow evaluation configuration loading."""

from evaluations.mlflow.configuration import get_config_loader


def test_validate_all_mlflow_eval_configs() -> None:
    """Validate that every MLflow eval config in config/evaluations/mlflow/ is well-formed.

    The eval runner reads its config at startup, so a broken config (missing file,
    malformed YAML, or a value that fails MlflowEvaluationConfig validation) would only
    be discovered when someone runs the evals with that config. This test loads all
    committed configs in CI/pre-commit so such breakage is caught at commit time, not
    at runtime.
    """
    get_config_loader().validate_all_configs()
