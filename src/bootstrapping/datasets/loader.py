"""Loading dataset records from YAML files under assets/datasets/.

Files may be plain ``.yaml`` or ``.yaml.jinja``; the latter is rendered as a
Jinja template that must yield valid YAML.
"""

import logging
from pathlib import Path

import jinja2
import yaml

from bootstrapping.common.configuration import DatasetFiltersConfig
from bootstrapping.common.exceptions import BootstrapError
from bootstrapping.datasets.filtering import apply_filters
from bootstrapping.datasets.model import DatasetRecord

logger = logging.getLogger(__name__)

DATASETS_DIR = Path("assets") / "datasets"
DEBUG_DIR = Path(".debug") / "datasets"
JINJA_ENV = jinja2.Environment(undefined=jinja2.StrictUndefined, keep_trailing_newline=True)


def _dataset_files(directory: Path) -> list[Path]:
    """Return the dataset files in a directory.

    Raises BootstrapError if two files share the same base name, e.g.
    ``a.yaml`` and ``a.yaml.jinja``.
    """
    files = sorted({*directory.glob("*.yaml"), *directory.glob("*.yaml.jinja")})
    seen: dict[str, Path] = {}
    for path in files:
        base = path.name.removesuffix(".jinja")
        if base in seen:
            raise BootstrapError(
                f"Dataset files '{seen[base]}' and '{path}' share the base name '{base}'; "
                "a base name may be used by at most one file."
            )
        seen[base] = path
    return files


def load_datasets(names: list[str], dump_rendered: bool = False) -> list[DatasetRecord]:
    """Load every record from the named dataset folders.

    Names are folder names relative to assets/datasets/; each file in a folder
    may hold a list of records. Disabled records are skipped. With
    dump_rendered, rendered Jinja templates are written to .debug/datasets/.
    """
    records: list[DatasetRecord] = []
    for name in names:
        directory = DATASETS_DIR / name
        if not directory.is_dir():
            raise BootstrapError(f"Dataset directory '{directory}' does not exist.")
        for path in _dataset_files(directory):
            text = path.read_text(encoding="utf-8")
            if path.name.endswith(".yaml.jinja"):
                try:
                    text = JINJA_ENV.from_string(text).render()
                except jinja2.TemplateError as error:
                    raise BootstrapError(
                        f"Dataset file '{path}' failed to render as a Jinja template: {error}"
                    ) from error
                if dump_rendered:
                    dump_path = DEBUG_DIR / name / path.name.removesuffix(".jinja")
                    dump_path.parent.mkdir(parents=True, exist_ok=True)
                    dump_path.write_text(text, encoding="utf-8")
                    logger.info("Dumped rendered dataset template to '%s'.", dump_path)
            raw = yaml.safe_load(text)
            if not isinstance(raw, list):
                raise BootstrapError(f"Dataset file '{path}' must contain a list of records.")
            stem = path.name.removesuffix(".jinja").removesuffix(".yaml")
            width = len(str(len(raw)))
            for index, entry in enumerate(raw, start=1):
                entry["unique_id"] = f"{name}/{stem}/{index:0{width}d}"
                record = DatasetRecord.model_validate(entry)
                if not record.disabled:
                    records.append(record)
    return records


def load_filtered_datasets(
    names: list[str], filters: DatasetFiltersConfig, dump_rendered: bool = False
) -> list[DatasetRecord]:
    """Load every record from the named dataset folders, then apply the filters."""
    return apply_filters(load_datasets(names, dump_rendered), filters)
