"""Generic config loading."""

import logging
import shutil
import tempfile
import threading
import zipfile
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Generic, Self, TypeVar

import yaml
from pydantic import BaseModel, field_validator, model_validator

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


class ListMergeBehaviour(str, Enum):
    """How list fields are combined when merging config layers."""

    OVERWRITE = "overwrite"
    EXTEND = "extend"


class ConfigLoaderParams(BaseModel):
    """Optional behavior settings for ConfigLoader."""

    # How list fields are combined when merging config layers: overwrite replaces the value,
    # extend appends the later layer's items to the earlier layer's items.
    list_merge_behaviour: ListMergeBehaviour = ListMergeBehaviour.OVERWRITE
    common_config_name: str = "_common"
    # The config loaded when load() is called without an explicit config name.
    default_config_name: str = "default"
    mixins_folder_name: str = "mixins"
    local_override_filename_infix: str = "local"
    mixins_field: str = "_mixins"
    config_file_extension: str = "yaml"
    # When set, rendered configs are dumped under this path; None disables the dump.
    debug_dir_path: Path | None = None
    # When set, the rendered debug dump is zipped into a timestamped archive in this folder after each load.
    archive_dir_path: Path | None = None
    # When set, at most this many archives are kept in the archive dir; older ones are pruned after each archive.
    archive_max_count: int | None = None

    @field_validator("config_file_extension")
    @classmethod
    def _check_config_file_extension(cls, value: str) -> str:
        if value not in ("yaml", "yml"):
            raise ValueError("config_file_extension must be 'yaml' or 'yml'.")
        return value

    @model_validator(mode="after")
    def _check_debug_and_archive_dirs_disjoint(self) -> Self:
        if (
            self.debug_dir_path is not None
            and self.archive_dir_path is not None
            and (
                self.debug_dir_path.is_relative_to(self.archive_dir_path)
                or self.archive_dir_path.is_relative_to(self.debug_dir_path)
            )
        ):
            raise ValueError("debug_dir_path and archive_dir_path must not be the same or nested within each other.")
        if self.archive_max_count is not None and self.archive_max_count < 1:
            raise ValueError("archive_max_count must be a positive integer.")
        return self


def deep_merge(configs: list[dict[str, Any]], list_merge_behaviour: ListMergeBehaviour) -> dict[str, Any]:
    """Deep-merge configs in order, with later entries winning over earlier ones.

    List fields are combined according to list_merge_behaviour: OVERWRITE replaces the earlier
    value, EXTEND appends the later layer's items to the earlier layer's items.
    """
    merged: dict[str, Any] = {}
    for config in configs:
        for key, value in config.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = deep_merge([merged[key], value], list_merge_behaviour)
            elif (
                isinstance(value, list)
                and isinstance(merged.get(key), list)
                and list_merge_behaviour is ListMergeBehaviour.EXTEND
            ):
                merged[key] = merged[key] + value
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
        # Serializes the rmtree + replace of the shared debug dir; the dumps themselves happen in private staging dirs.
        self._swap_lock = threading.Lock()

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
        rendered = deep_merge([config, *mixins], self._params.list_merge_behaviour)
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

    def _archive_debug_dir_if_needed(self, debug_dir: Path | None, config_name: str) -> None:
        """Zip the debug dir contents into a timestamped archive in the archive dir, when set.

        Prunes the oldest archives first when archive_max_count is set.
        """
        if debug_dir is None or self._params.archive_dir_path is None:
            return
        archive_dir = self._params.archive_dir_path
        archive_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        archive_path = archive_dir / f"config_{config_name}_{timestamp}.zip"
        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(debug_dir.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(debug_dir))
        logger.info("Archived config debug files to %s", archive_path)
        self._prune_archives_if_needed(archive_dir)

    def _prune_archives_if_needed(self, archive_dir: Path) -> None:
        """Keep only the newest archive_max_count archives (by modification time), when the cap is set."""
        max_count = self._params.archive_max_count
        if max_count is None:
            return
        archives = sorted(archive_dir.glob("*.zip"), key=lambda path: path.stat().st_mtime, reverse=True)
        for stale in archives[max_count:]:
            stale.unlink(missing_ok=True)
            logger.info("Pruned stale config archive: %s", stale)

    def _swap_debug_dir_if_needed(self, staging_dir: Path) -> None:
        """Replace the contents of the configured debug dir with the staged one, when set.

        The target is emptied and the staged contents are copied in. The copy is not atomic, so a
        reader may briefly see an empty or partially populated debug dir; the swap is serialized per
        instance, so concurrent loads on the same loader cannot interleave.
        """
        debug_dir = self._params.debug_dir_path
        if debug_dir is None:
            return
        with self._swap_lock:
            if debug_dir.exists():
                shutil.rmtree(debug_dir)
            shutil.copytree(staging_dir, debug_dir)
        logger.info("Replaced the debug dir with the latest dump: %s", debug_dir)

    def list_config_names(self) -> list[str]:
        """List every config name in the base path, excluding the common config and local overrides."""
        names = []
        for path in sorted(self._base_path.glob(f"*.{self._params.config_file_extension}")):
            config_name = path.name[: -len(f".{self._params.config_file_extension}")]
            if config_name == self._params.common_config_name:
                continue
            if config_name.endswith(f".{self._params.local_override_filename_infix}"):
                continue
            names.append(config_name)
        return names

    def validate_all_configs(self) -> None:
        """Load every config in the base path, skipping the common config and local overrides."""
        for config_name in self.list_config_names():
            self.load(config_name, include_local_overrides=False)

    def list_mixin_names(self) -> list[str]:
        """List every mixin name in the mixins folder, sorted; empty when the folder doesn't exist."""
        mixins_dir = self._base_path / self._params.mixins_folder_name
        if not mixins_dir.is_dir():
            return []
        return [
            path.name[: -len(f".{self._params.config_file_extension}")]
            for path in sorted(mixins_dir.glob(f"*.{self._params.config_file_extension}"))
        ]

    def validate_all_mixins(self) -> None:
        """Load the default config with every mixin in the mixins folder, to check each mixin is well-formed."""
        mixin_names = self.list_mixin_names()
        if not mixin_names:
            logger.info("No mixins found in %s, skipping mixin validation.", self._params.mixins_folder_name)
            return
        for mixin_name in mixin_names:
            self.load(self._params.default_config_name, extra_mixins=[mixin_name], include_local_overrides=False)

    def load(
        self,
        config_name: str | None = None,
        extra_mixins: list[str] | None = None,
        include_local_overrides: bool = True,
    ) -> ConfigT:
        """Load and validate the config.

        When a debug dir and/or archive dir is configured, the rendered layers are dumped into a private
        staging dir, archived, and the staged dir then replaces the latest debug dir (swaps are serialized
        per instance). On failure the staging dir is discarded and no archive is written.

        Args:
            config_name: Name of the config to load, or None to load the default config.
            extra_mixins: Names of mixins in the mixins folder, applied on top of all other layers, in order.
            include_local_overrides: When False, skip the <config_name>.<infix>.yaml override file.
        """
        if config_name is None:
            config_name = self._params.default_config_name
        if config_name == self._params.common_config_name:
            raise ValueError(
                f"Cannot load '{config_name}': the common config is not a standalone config that can be loaded."
            )
        if self._params.debug_dir_path is None and self._params.archive_dir_path is None:
            return self._load_config(config_name, extra_mixins, include_local_overrides, debug_dir=None)
        with tempfile.TemporaryDirectory(prefix="config_debug_") as staging_name:
            staging_dir = Path(staging_name)
            where = self._params.debug_dir_path if self._params.debug_dir_path is not None else "the archive dir"
            logger.info("Preparing config debug dump for %s in staging dir: %s", where, staging_dir)
            result = self._load_config(config_name, extra_mixins, include_local_overrides, debug_dir=staging_dir)
            self._archive_debug_dir_if_needed(staging_dir, config_name)
            self._swap_debug_dir_if_needed(staging_dir)
            return result

    def _load_config(
        self,
        config_name: str,
        extra_mixins: list[str] | None,
        include_local_overrides: bool,
        debug_dir: Path | None,
    ) -> ConfigT:
        """Merge all config layers, dump them under the debug dir when set, and validate into the model."""
        ext = self._params.config_file_extension

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

        merged = deep_merge(configs, self._params.list_merge_behaviour)
        self._dump_merged_config_if_needed(config_name, merged, MERGED_FOLDER, debug_dir)

        merged_with_extra = deep_merge([merged, *extra_mixins_loaded], self._params.list_merge_behaviour)
        self._dump_merged_config_if_needed(config_name, merged_with_extra, FINAL_FOLDER, debug_dir)

        return self._model_class.model_validate(merged_with_extra)
