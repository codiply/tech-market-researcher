"""Loading dataset records from YAML files under assets/datasets/."""

from pathlib import Path

import yaml

from bootstrapping.common.exceptions import BootstrapError
from bootstrapping.datasets.model import DatasetRecord

DATASETS_DIR = Path("assets") / "datasets"


def load_datasets(names: list[str]) -> list[DatasetRecord]:
    """Load every record from the named dataset folders.

    Names are folder names relative to assets/datasets/; each YAML file in
    a folder may hold a list of records. Disabled records are skipped.
    """
    records: list[DatasetRecord] = []
    for name in names:
        directory = DATASETS_DIR / name
        if not directory.is_dir():
            raise BootstrapError(f"Dataset directory '{directory}' does not exist.")
        for path in sorted(directory.glob("*.yaml")):
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
            if not isinstance(raw, list):
                raise BootstrapError(f"Dataset file '{path}' must contain a list of records.")
            width = len(str(len(raw)))
            for index, entry in enumerate(raw, start=1):
                entry["unique_id"] = f"{name}/{path.stem}/{index:0{width}d}"
                record = DatasetRecord.model_validate(entry)
                if not record.disabled:
                    records.append(record)
    return records
