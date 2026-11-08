"""Unit tests for the agent configuration loading."""

from tech_market_researcher.core.configuration import get_config_loader


def test_validate_all_agent_configs() -> None:
    """Validate that every agent config in config/agent/ is well-formed.

    The agent reads its config at startup, so a broken config (missing file, malformed
    YAML, or a value that fails ApplicationConfig validation) would only be discovered
    when someone boots the stack with that config. This test loads all committed configs
    in CI/pre-commit so such breakage is caught at commit time, not at runtime.
    """
    get_config_loader().validate_all_configs()
