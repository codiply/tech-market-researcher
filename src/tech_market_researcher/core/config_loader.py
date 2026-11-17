"""Generic config loading."""

import logging
import shutil
from pathlib import Path
from typing import Any, Generic, TypeVar

import yaml
from pydantic import BaseModel, field_validator

from tech_market_researcher.core.collection_utils import find_duplicates

ConfigT = TypeVar("ConfigT", bound=BaseModel)

logger = logging.getLogger(__name__)

# Debug dump folder names, numbered in the order the layers are applied.
MIXINS_FOLDER = "0_mixins"
RAW_FOLDER = "1_raw"
CONFIGS_WITH_MIXINS_FOLDER = "2_configs_with_mixins"
MERGED_FOLDER = "3_merged"
FINAL_FOLDER = "4_final"
# Debug dump file listing the extra mixins in the order they were passed.
EXTRA_MIXINS_LIST_FILENAME = "extra_mixins_list.yaml"


class ConfigLoaderParams(BaseModel):
    """Optional behavior settings for ConfigLoader."""

    common_config_name: str = "_common"
    mixins_folder_name: str = "mixins"
    local_override_filename_infix: str = "local"
    mixins_field: str = "_mixins"
    config_file_extension: str = "yaml"
    # When set, rendered configs are dumped under this path; None disables the dump.
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

    def _read_mixin(self, mixin_name: str, debug_dir: Path | None) -> dict[str, Any]:
        """Read a mixin by name from the mixins folder, dropping the mixins field."""
        path = self._base_path / self._params.mixins_folder_name / f"{mixin_name}.{self._params.config_file_extension}"
        if not path.exists():
            raise FileNotFoundError(f"Mixin file not found: {path}")
        self._debug_copy_file_if_needed(debug_dir, f"{MIXINS_FOLDER}/{path.name}", path)
        mixin = self._read_yaml(path)
        mixin.pop(self._params.mixins_field, None)
        return mixin

    def _load_with_mixins(self, path: Path, debug_dir: Path | None) -> dict[str, Any]:
        """Load the config at path, deep-merged with its declared mixins in declaration order."""
        self._debug_copy_file_if_needed(debug_dir, f"{RAW_FOLDER}/{path.name}", path)
        config = self._read_yaml(path)
        mixins = [self._read_mixin(mixin_name, debug_dir) for mixin_name in config.pop(self._params.mixins_field, [])]
        rendered = deep_merge([config, *mixins])
        self._debug_dump_yaml_if_needed(debug_dir, f"{CONFIGS_WITH_MIXINS_FOLDER}/{path.name}", rendered)
        return rendered

    def _load_extra_mixins(self, extra_mixins: list[str] | None, debug_dir: Path | None) -> list[dict[str, Any]]:
        """Load extra mixins from the mixins folder, in the given order, to be applied on top of everything else."""
        mixin_names = extra_mixins or []
        duplicates = find_duplicates(mixin_names)
        if duplicates:
            raise ValueError(f"Extra mixins must not contain duplicates: {', '.join(duplicates)}.")
        mixins = [self._read_mixin(mixin_name, debug_dir) for mixin_name in mixin_names]
        self._debug_dump_yaml_if_needed(debug_dir, EXTRA_MIXINS_LIST_FILENAME, mixin_names)
        if mixins:
            logger.info("Applying %d extra mixin(s): %s", len(mixins), ", ".join(mixin_names))
        return mixins

    def _prepare_debug_directory_if_needed(self) -> Path | None:
        """Return the effective debug dir, deleting any existing files there so it reflects only the latest run.

        Returns None when no debug dir was configured.
        """
        debug_dir = self._params.debug_dir_path
        if debug_dir is None:
            return None
        if debug_dir.exists():
            shutil.rmtree(debug_dir)
        logger.info("Dumping config debug files to %s", debug_dir)
        return debug_dir

    def _debug_copy_file_if_needed(self, debug_dir: Path | None, relative_path: str, source_path: Path) -> None:
        """Copy a file verbatim under the debug dir, creating folders as needed, when set."""
        if debug_dir is None:
            return
        path = debug_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, path)

    def _debug_dump_yaml_if_needed(
        self, debug_dir: Path | None, relative_path: str, data: dict[str, Any] | list[str]
    ) -> None:
        """Write a rendered config (or list) under the debug dir, creating folders as needed, when set."""
        if debug_dir is None:
            return
        path = debug_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(data, sort_keys=False))

    def _dump_merged_config_if_needed(
        self, config_name: str, merged: dict[str, Any], folder: str, debug_dir: Path | None
    ) -> None:
        """Dump a merged config under the given folder of the effective debug dir, when set."""
        self._debug_dump_yaml_if_needed(
            debug_dir, f"{folder}/{config_name}.{self._params.config_file_extension}", merged
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

    def load(
        self, config_name: str, extra_mixins: list[str] | None = None, include_local_overrides: bool = True
    ) -> ConfigT:
        """Load and validate the config.

        Args:
            config_name: Name of the config to load.
            extra_mixins: Names of mixins in the mixins folder, applied on top of all other layers, in order.
            include_local_overrides: When False, skip the <config_name>.<infix>.yaml override file.
        """
        if config_name == self._params.common_config_name:
            raise ValueError(
                f"Cannot load '{config_name}': the common config is not a standalone config that can be loaded."
            )
        ext = self._params.config_file_extension

        debug_dir = self._prepare_debug_directory_if_needed()

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
        # Apply the extra mixins programmatically, on top of everything else
        extra_mixins_loaded = self._load_extra_mixins(extra_mixins, debug_dir)

        merged = deep_merge(configs)
        self._dump_merged_config_if_needed(config_name, merged, MERGED_FOLDER, debug_dir)

        merged_with_extra = deep_merge([merged, *extra_mixins_loaded])
        self._dump_merged_config_if_needed(config_name, merged_with_extra, FINAL_FOLDER, debug_dir)

        return self._model_class.model_validate(merged_with_extra)
