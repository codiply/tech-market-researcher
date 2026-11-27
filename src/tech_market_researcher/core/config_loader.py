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


class MergeOperation(str, Enum):
    """Per-node merge behaviour forced by a merge directive YAML tag."""

    OVERWRITE = "overwrite"
    EXTEND = "extend"
    DELETE = "delete"


class MergeDirective:
    """A config value wrapped by a merge directive YAML tag (``!overwrite``, ``!extend`` or ``!delete``).

    The directive only affects how the value is combined with the previous config layer in
    deep_merge, overriding the loader's list merge behaviour for that node (a !delete directive
    removes the key entirely); it is unwrapped or dropped before validation, so it never reaches the model.
    """

    __slots__ = ("value", "operation")

    def __init__(self, value: Any, operation: MergeOperation) -> None:
        self.value = value
        self.operation = operation

    def __eq__(self, other: object) -> bool:
        return isinstance(other, MergeDirective) and self.value == other.value and self.operation == other.operation

    def __repr__(self) -> str:
        return f"MergeDirective({self.operation.value!r}, {self.value!r})"


# Module-level resolver used to determine the underlying type of scalars wrapped in a merge directive tag.
_scalar_resolver = yaml.resolver.Resolver()


def _construct_merge_directive(loader: yaml.SafeLoader, node: yaml.Node, operation: MergeOperation) -> MergeDirective:
    if isinstance(node, yaml.MappingNode):
        loader.flatten_mapping(node)
        value: Any = loader.construct_mapping(node)
    elif isinstance(node, yaml.SequenceNode):
        value = loader.construct_sequence(node)
    elif isinstance(node, yaml.ScalarNode):
        # implicit is PyYAML's (plain, non_plain) resolver-allowance tuple: a plain (unquoted) scalar may
        # use plain-implicit resolvers (int, bool, float, ...), while a quoted scalar always resolves as a string.
        implicit = (True, False) if node.style is None else (False, False)
        tag = _scalar_resolver.resolve(yaml.ScalarNode, node.value, implicit)
        value = loader.yaml_constructors[tag](loader, node)
    else:
        raise ValueError(f"Cannot apply a merge directive tag to {type(node).__name__} node.")
    return MergeDirective(value, operation)


def _construct_overwrite(loader: yaml.SafeLoader, node: yaml.Node) -> MergeDirective:
    return _construct_merge_directive(loader, node, MergeOperation.OVERWRITE)


def _construct_extend(loader: yaml.SafeLoader, node: yaml.Node) -> MergeDirective:
    return _construct_merge_directive(loader, node, MergeOperation.EXTEND)


def _construct_delete(
    loader: yaml.SafeLoader,  # pylint: disable=unused-argument
    node: yaml.Node,  # pylint: disable=unused-argument
) -> MergeDirective:
    """Construct a !delete sentinel; the tagged value (usually absent) is ignored."""
    return MergeDirective(None, MergeOperation.DELETE)


class ConfigSafeLoader(yaml.SafeLoader):  # pylint: disable=too-many-ancestors
    """yaml.SafeLoader that additionally understands the !overwrite, !extend and !delete merge directive tags.

    !overwrite forces the value to replace the existing value at merge time (instead of merging dicts
    or extending lists); !extend forces the value's list items to be appended to the existing list;
    !delete removes the key from the merged result entirely. Both !overwrite and !extend override the
    loader's configured list merge behaviour for the tagged node only.
    """


ConfigSafeLoader.add_constructor("!" + MergeOperation.OVERWRITE.value, _construct_overwrite)
ConfigSafeLoader.add_constructor("!" + MergeOperation.EXTEND.value, _construct_extend)
ConfigSafeLoader.add_constructor("!" + MergeOperation.DELETE.value, _construct_delete)


def _represent_merge_directive(dumper: yaml.SafeDumper, directive: MergeDirective) -> yaml.Node:
    """Render a MergeDirective as its plain wrapped value, so debug dumps stay plain YAML."""
    return dumper.represent_data(directive.value)


yaml.SafeDumper.add_representer(MergeDirective, _represent_merge_directive)


def _unwrap_merge_directives(value: Any) -> Any:
    """Recursively replace MergeDirective values with their wrapped values, so none reach the model."""
    if isinstance(value, MergeDirective):
        return value.value
    if isinstance(value, dict):
        return {key: _unwrap_merge_directives(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_unwrap_merge_directives(item) for item in value]
    return value


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
    value, EXTEND appends the later layer's items to the earlier layer's items. A value wrapped in
    a MergeDirective forces its own behaviour for that node, overriding list_merge_behaviour; a
    !extend directive whose earlier value is not a list sets the value, i.e. extends an empty list;
    a !delete directive removes the key from the merged result, whether or not an earlier layer set it.
    """
    merged: dict[str, Any] = {}
    for config in configs:
        for key, value in config.items():
            existing = merged.get(key)
            if isinstance(value, MergeDirective):
                if value.operation is MergeOperation.DELETE:
                    merged.pop(key, None)
                    continue
                if (
                    value.operation is MergeOperation.EXTEND
                    and isinstance(existing, list)
                    and isinstance(value.value, list)
                ):
                    merged[key] = existing + value.value
                    continue
                if value.operation is MergeOperation.OVERWRITE:
                    merged[key] = value.value
                    continue
                value = value.value  # !extend on a non-list: fall back to the default behaviour
            if isinstance(value, dict) and isinstance(existing, dict):
                merged[key] = deep_merge([existing, value], list_merge_behaviour)
            elif isinstance(value, dict) and existing is None:
                # Recurse into a wholly new subtree so merge directives inside it resolve.
                merged[key] = deep_merge([value], list_merge_behaviour)
            elif (
                isinstance(value, list)
                and isinstance(existing, list)
                and list_merge_behaviour is ListMergeBehaviour.EXTEND
            ):
                merged[key] = existing + value
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
        """Read a YAML config file. An empty file (e.g. fully commented out) is treated as an empty config.

        Parsed with ConfigSafeLoader, which additionally understands the !overwrite / !extend /
        !delete merge directive tags.
        """
        data = yaml.load(path.read_text(), Loader=ConfigSafeLoader)
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

    def _load_layer(self, path: Path, debug_dir: Path | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Load the config at path and its declared mixins, dropping the mixins field.

        Returns the file config and its mixins in declaration order, so the caller can apply them as
        separate merge layers (a file's merge directive tags must resolve against the previous files,
        not be pre-consumed within this file).
        """
        self._debug_copy_file_if_needed(debug_dir, f"{RAW_FOLDER}/{path.name}", path)
        config = self._read_yaml(path)
        mixins = [self._read_mixin(mixin_name, debug_dir) for mixin_name in config.pop(self._params.mixins_field, [])]
        rendered = deep_merge([config, *mixins], self._params.list_merge_behaviour)
        self._debug_dump_yaml_if_needed(debug_dir, f"{CONFIGS_WITH_MIXINS_FOLDER}/{path.name}", rendered)
        return config, mixins

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
            config, mixins = self._load_layer(common_path, debug_dir)
            configs.extend([config, *mixins])
        # Load the config named when calling this method
        named_path = self._base_path / f"{config_name}.{ext}"
        if not named_path.exists():
            raise FileNotFoundError(f"Config file not found: {named_path}")
        config, mixins = self._load_layer(named_path, debug_dir)
        configs.extend([config, *mixins])
        # Load the local override config if it exists
        local_path = self._base_path / f"{config_name}.{self._params.local_override_filename_infix}.{ext}"
        if include_local_overrides and local_path.exists():
            config, mixins = self._load_layer(local_path, debug_dir)
            configs.extend([config, *mixins])
        # Apply the extra mixins programmatically, on top of everything else
        extra_mixins_loaded = self._load_extra_mixins(extra_mixins, debug_dir)

        merged = deep_merge(configs, self._params.list_merge_behaviour)
        self._dump_merged_config_if_needed(config_name, merged, MERGED_FOLDER, debug_dir)

        merged_with_extra = deep_merge([merged, *extra_mixins_loaded], self._params.list_merge_behaviour)
        self._dump_merged_config_if_needed(config_name, merged_with_extra, FINAL_FOLDER, debug_dir)

        return self._model_class.model_validate(_unwrap_merge_directives(merged_with_extra))
