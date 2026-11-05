"""Loading dataset records from the local asset folders."""

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel

from bootstrapping.common.configuration import DatasetSourceConfig
from bootstrapping.common.exceptions import BootstrapError

DATASETS_DIR = Path("assets") / "datasets"


def record_input_hash(inputs: dict) -> str:
    """Stable hash of a record's inputs; MLflow uses it as its record dedup key."""
    return hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()


class DatasetRecord(BaseModel):
    """One record loaded from a local JSON file."""

    source: str  # path relative to assets/datasets/, extension stripped
    raw: dict  # record exactly as stored in the file

    @property
    def inputs(self) -> dict:
        """The record's inputs."""
        return self.raw["inputs"]


def load_records(dataset_config: DatasetSourceConfig) -> list[DatasetRecord]:
    """Load one record per JSON file across the configured local directories.

    The directory names are relative to assets/datasets/.
    """
    records: list[DatasetRecord] = []
    for name in dataset_config.local_directories:
        directory = DATASETS_DIR / name
        if not directory.is_dir():
            raise BootstrapError(f"Dataset directory '{directory}' does not exist.")
        for path in sorted(directory.glob("*.json")):
            records.append(
                DatasetRecord(
                    source=path.relative_to(DATASETS_DIR).with_suffix("").as_posix(),
                    raw=json.loads(path.read_text(encoding="utf-8")),
                )
            )
    if not records:
        raise BootstrapError(f"No records found for dataset '{dataset_config.dataset_name}'.")
    return records
