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


def test_validate_all_phoenix_eval_mixins() -> None:
    """Validate that every Phoenix eval mixin in config/evaluations/phoenix/mixins/ is well-formed.

    Each mixin is applied on top of the main eval config, so a broken mixin (missing file,
    malformed YAML, or a value that fails PhoenixEvaluationConfig validation) would only be
    discovered when someone runs the evals with that mixin. This test loads all committed
    mixins in CI/pre-commit so such breakage is caught at commit time, not at runtime.
    """
    get_config_loader().validate_all_mixins()
