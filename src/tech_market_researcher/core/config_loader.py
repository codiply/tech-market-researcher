"""Generic config loading."""

import os
import shutil
from pathlib import Path
from typing import Any, Generic, TypeVar

import yaml
from pydantic import BaseModel, field_validator

ConfigT = TypeVar("ConfigT", bound=BaseModel)

# True when the DEBUG_DUMP_RENDERED_CONFIGS env var is set to True, true, or 1.
DUMP_RENDERED_CONFIGS = os.environ.get("DEBUG_DUMP_RENDERED_CONFIGS", "").lower() in {"true", "1"}

# Debug dump folder names.
WITH_MIXINS_FOLDER = "with_mixins"
MERGED_FOLDER = "merged"


class ConfigLoaderParams(BaseModel):
    """Optional behavior settings for ConfigLoader."""

    common_config_name: str = "_common"
    mixins_folder_name: str = "mixins"
    local_override_filename_infix: str = "local"
    mixins_field: str = "_mixins"
    config_file_extension: str = "yaml"
    debug_dir_path: Path | None = None

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

    def _read_yaml(self, path: Path) -> dict[str, Any]:
        """Read a YAML config file. An empty file (e.g. fully commented out) is treated as an empty config."""
        data = yaml.safe_load(path.read_text())
        if data is None:
            return {}
        if not isinstance(data, dict):
            raise ValueError(f"Expected a mapping at the top level of {path}, got {type(data).__name__}.")
        return data

    def _load_with_mixins(self, path: Path, debug_dir: Path | None) -> dict[str, Any]:
        """Load the config at path, deep-merged with its declared mixins in declaration order."""
        config = self._read_yaml(path)
        mixins: list[dict[str, Any]] = []
        for mixin_name in config.pop(self._params.mixins_field, []):
            mixin = self._read_yaml(
                self._base_path / self._params.mixins_folder_name / f"{mixin_name}.{self._params.config_file_extension}"
            )
            mixin.pop(self._params.mixins_field, None)
            mixins.append(mixin)
        rendered = deep_merge([config, *mixins])
        self._debug_dump_rendered_file_if_needed(debug_dir, f"{WITH_MIXINS_FOLDER}/{path.name}", rendered)
        return rendered

    def _prepare_debug_directory_if_needed(self, config_name: str) -> Path | None:
        """Return the effective debug dir (debug_dir_path/config_name), deleting any existing files there.

        Returns None when dumping is disabled.
        """
        base = self._params.debug_dir_path
        if base is None or not DUMP_RENDERED_CONFIGS:
            return None
        debug_dir = base / config_name
        if debug_dir.exists():
            shutil.rmtree(debug_dir)
        return debug_dir

    def _debug_dump_rendered_file_if_needed(
        self, debug_dir: Path | None, relative_path: str, config: dict[str, Any]
    ) -> None:
        """Write a rendered config under the debug dir, creating folders as needed, when dumping is enabled."""
        if debug_dir is None or not DUMP_RENDERED_CONFIGS:
            return
        path = debug_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(config, sort_keys=False))

    def _dump_merged_config_if_needed(self, config_name: str, merged: dict[str, Any], debug_dir: Path | None) -> None:
        """Dump the final merged config under the effective debug dir, when set."""
        self._debug_dump_rendered_file_if_needed(
            debug_dir, f"{MERGED_FOLDER}/{config_name}.{self._params.config_file_extension}", merged
        )

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
        ext = self._params.config_file_extension

        debug_dir = self._prepare_debug_directory_if_needed(config_name)

        configs: list[dict[str, Any]] = []

        # Load the common config if it exists
        common_path = self._base_path / f"{self._params.common_config_name}.{ext}"
        if common_path.exists():
            configs.append(self._load_with_mixins(common_path, debug_dir))
        # Load the config named when calling this method
        named_path = self._base_path / f"{config_name}.{ext}"
        if not named_path.exists():
            raise FileNotFoundError(f"Config file not found: {named_path}")
        configs.append(self._load_with_mixins(named_path, debug_dir))
        # Load the local override config if it exists
        local_path = self._base_path / f"{config_name}.{self._params.local_override_filename_infix}.{ext}"
        if include_local_overrides and local_path.exists():
            configs.append(self._load_with_mixins(local_path, debug_dir))

        merged = deep_merge(configs)

        self._dump_merged_config_if_needed(config_name, merged, debug_dir)

        return self._model_class.model_validate(merged)
