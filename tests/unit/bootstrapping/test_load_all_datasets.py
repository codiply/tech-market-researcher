"""Unit tests for loading the committed evaluation datasets."""

import pytest

from bootstrapping.datasets.loader import DATASETS_DIR, load_datasets

DATASET_NAMES = sorted(path.name for path in DATASETS_DIR.iterdir() if path.is_dir())


@pytest.mark.parametrize("name", DATASET_NAMES)
def test_load_dataset(name: str) -> None:
    """Load a single dataset folder without exceptions.

    A broken dataset file (malformed YAML, an invalid Jinja template, or a record
    that fails DatasetRecord validation) would only be discovered when someone
    uploads the datasets to the evaluators. This test loads all committed datasets
    in CI/pre-commit so such breakage is caught at commit time, not at runtime.
    The test is parametrized per dataset so a failure points at the exact folder.
    """
    load_datasets([name])
