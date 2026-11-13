"""Unit tests for the Phoenix evaluation configuration loading."""

from evaluations.phoenix.configuration import get_config_loader


def test_validate_all_phoenix_eval_configs() -> None:
    """Validate that every Phoenix eval config in config/evaluations/phoenix/ is well-formed.

    The eval runner reads its config at startup, so a broken config (missing file,
    malformed YAML, or a value that fails PhoenixEvaluationConfig validation) would only
    be discovered when someone runs the evals with that config. This test loads all
    committed configs in CI/pre-commit so such breakage is caught at commit time, not
    at runtime.
    """
    get_config_loader().validate_all_configs()
