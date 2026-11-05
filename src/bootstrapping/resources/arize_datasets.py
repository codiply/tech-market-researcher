"""Arize (Phoenix) dataset bootstrapping via the Phoenix REST API."""

import logging
import os

import httpx

from bootstrapping.common.configuration import ArizeDatasetConfig, BootstrapConfig
from bootstrapping.common.datasets import load_records
from bootstrapping.common.exceptions import BootstrapError

logger = logging.getLogger(__name__)


def _find_dataset(client: httpx.Client, name: str) -> str | None:
    """Return the id of the dataset with the given name, or None."""
    response = client.get("/v1/datasets", params={"name": name, "limit": 10})
    response.raise_for_status()
    for dataset in response.json().get("data", []):
        if dataset["name"] == name:
            return dataset["id"]
    return None


def _example_metadata(record: dict) -> dict:
    """Map the MLflow record's extra fields (expectations, tags) onto Phoenix metadata."""
    return {key: record[key] for key in ("expectations", "tags") if key in record}


def _ensure_dataset(client: httpx.Client, dataset_config: ArizeDatasetConfig) -> None:
    """Create or update the dataset so its examples match the configured records.

    Examples are upserted by the local file path relative to assets/datasets/ (directory plus
    file name, human-readable in the Arize UI), so re-runs do not create duplicates. Records no
    longer present in the folders are not pruned: Phoenix exposes no REST endpoint for deleting
    examples. Each mutating upload creates a new immutable dataset version, which experiments
    pin to for reproducible evaluations.
    """
    records = load_records(dataset_config)
    payload: dict = {
        "name": dataset_config.dataset_name,
        "inputs": [record.inputs for record in records],
        "outputs": [record.raw.get("outputs") or {} for record in records],
        "metadata": [_example_metadata(record.raw) for record in records],
        "example_ids": [record.source for record in records],
    }
    if _find_dataset(client, dataset_config.dataset_name) is None:
        payload["action"] = "create"
        payload["description"] = dataset_config.description or (
            f"Provisioned by the bootstrap from assets/datasets/{', '.join(dataset_config.local_directories)}"
        )
        logger.info("Creating dataset '%s' in Arize.", dataset_config.dataset_name)
    else:
        payload["action"] = "update"
        logger.info("Dataset '%s' already exists in Arize - updating.", dataset_config.dataset_name)
    response = client.post("/v1/datasets/upload", params={"sync": "true"}, json=payload)
    if not response.is_success:
        raise BootstrapError(f"Failed to sync dataset '{dataset_config.dataset_name}' with Arize: {response.text}")
    data = response.json()["data"]
    logger.info(
        "Upserted %d examples to dataset '%s' (%d created, %d updated, %d deleted).",
        len(records),
        dataset_config.dataset_name,
        data["num_created_examples"],
        data["num_updated_examples"],
        data["num_deleted_examples"],
    )


def bootstrap_arize(config: BootstrapConfig) -> None:
    """Provision Arize (Phoenix) datasets from the configured local directories."""
    base_url = os.environ["ARIZE_SERVER_URI"].rstrip("/")
    with httpx.Client(base_url=base_url, timeout=60.0) as client:
        for dataset_config in config.arize.datasets:
            _ensure_dataset(client, dataset_config)
