"""Generic config loading."""

from pathlib import Path
from typing import Any, Generic, TypeVar

import yaml
from pydantic import BaseModel, field_validator

ConfigT = TypeVar("ConfigT", bound=BaseModel)


class ConfigLoaderParams(BaseModel):
    """Optional behavior settings for ConfigLoader."""

    common_config_name: str = "_common"
    mixins_folder_name: str = "mixins"
    local_override_filename_infix: str = "local"
    mixins_field: str = "_mixins"
    config_file_extension: str = "yaml"

    @field_validator("config_file_extension")
    @classmethod
    def _check_config_file_extension(cls, value: str) -> str:
        if value not in ("yaml", "yml"):
            raise ValueError("config_file_extension must be 'yaml' or 'yml'.")
        return value


def deep_merge(configs: list[dict[str, Any]]) -> dict[str, Any]:
    """Deep-merge configs in order, with later entries winning over earlier ones. Arrays are overwritten, not merged."""
    merged: dict[str, Any] = {}
    for config in configs:
        for key, value in config.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = deep_merge([merged[key], value])
            else:
                merged[key] = value
    return merged


class ConfigLoader(Generic[ConfigT]):
    """Loads a config into a pydantic model."""

    def __init__(
        self,
        model_class: type[ConfigT],
        base_path: Path,
        params: ConfigLoaderParams = ConfigLoaderParams(),
    ) -> None:
        self._model_class = model_class
        self._base_path = base_path
        self._params = params

    def _load_with_mixins(self, path: Path) -> dict[str, Any]:
        """Load the config at path, deep-merged with its declared mixins in declaration order."""
        config = yaml.safe_load(path.read_text())
        mixins: list[dict[str, Any]] = []
        for mixin_name in config.pop(self._params.mixins_field, []):
            mixin = yaml.safe_load(
                (
                    self._base_path
                    / self._params.mixins_folder_name
                    / f"{mixin_name}.{self._params.config_file_extension}"
                ).read_text()
            )
            mixin.pop(self._params.mixins_field, None)
            mixins.append(mixin)
        return deep_merge([config, *mixins])

    def validate_all_configs(self) -> None:
        """Load every config in the base path, skipping the common config and local overrides."""
        for path in sorted(self._base_path.glob(f"*.{self._params.config_file_extension}")):
            config_name = path.name[: -len(f".{self._params.config_file_extension}")]
            if config_name == self._params.common_config_name:
                continue
            if config_name.endswith(f".{self._params.local_override_filename_infix}"):
                continue
            self.load(config_name, include_local_overrides=False)

    def load(self, config_name: str, include_local_overrides: bool = True) -> ConfigT:
        """Load and validate the config.

        Args:
            config_name: Name of the config to load.
            include_local_overrides: When False, skip the <config_name>.<infix>.yaml override file.
        """
        if config_name == self._params.common_config_name:
            raise ValueError(
                f"Cannot load '{config_name}': the common config is not a standalone config that can be loaded."
            )
        configs: list[dict[str, Any]] = []
        common_path = self._base_path / f"{self._params.common_config_name}.{self._params.config_file_extension}"
        if common_path.exists():
            configs.append(self._load_with_mixins(common_path))
        named_path = self._base_path / f"{config_name}.{self._params.config_file_extension}"
        if not named_path.exists():
            raise FileNotFoundError(f"Config file not found: {named_path}")
        configs.append(self._load_with_mixins(named_path))
        local_path = (
            self._base_path
            / f"{config_name}.{self._params.local_override_filename_infix}.{self._params.config_file_extension}"
        )
        if include_local_overrides and local_path.exists():
            configs.append(self._load_with_mixins(local_path))
        merged = deep_merge(configs)
        return self._model_class.model_validate(merged)
