"""Unit tests for the bootstrapping configuration loading."""

from bootstrapping.common.configuration import get_config_loader


def test_validate_all_bootstrap_configs() -> None:
    """Validate that every bootstrap config in config/bootstrapping/ is well-formed.

    The bootstrap script reads its config at startup, so a broken config (missing file,
    malformed YAML, or a value that fails BootstrapConfig validation) would only be
    discovered when someone boots the stack with that config. This test loads all
    committed configs in CI/pre-commit so such breakage is caught at commit time, not
    at runtime.
    """
    get_config_loader().validate_all_configs()


def test_validate_all_bootstrap_mixins() -> None:
    """Validate that every bootstrap mixin in config/bootstrapping/mixins/ is well-formed.

    Each mixin is applied on top of the main bootstrap config, so a broken mixin (missing file,
    malformed YAML, or a value that fails BootstrapConfig validation) would only be discovered
    when someone boots the stack with that mixin. This test loads all committed mixins in
    CI/pre-commit so such breakage is caught at commit time, not at runtime.
    """
    get_config_loader().validate_all_mixins()
