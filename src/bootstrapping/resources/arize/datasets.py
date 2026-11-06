"""Arize (Phoenix) dataset bootstrapping via the Phoenix REST API."""

import logging
import os

import httpx

from bootstrapping.common.configuration import ArizeDatasetConfig, BootstrapConfig
from bootstrapping.common.exceptions import BootstrapError
from bootstrapping.datasets.arize import ArizeExample, to_arize_example
from bootstrapping.datasets.loader import load_datasets

logger = logging.getLogger(__name__)


def _find_dataset(client: httpx.Client, name: str) -> str | None:
    """Return the id of the dataset with the given name, or None."""
    response = client.get("/v1/datasets", params={"name": name, "limit": 10})
    response.raise_for_status()
    for dataset in response.json().get("data", []):
        if dataset["name"] == name:
            return dataset["id"]
    return None


def _ensure_dataset(client: httpx.Client, dataset_config: ArizeDatasetConfig, dump_rendered: bool) -> None:
    """Create or update the dataset so its examples match the configured records.

    Examples are upserted by their unique id (folder/file name/index under assets/datasets/,
    human-readable in the Arize UI), so re-runs do not create duplicates. Records no longer
    present in the folders are not pruned: Phoenix exposes no REST endpoint for deleting
    examples. Each mutating upload creates a new immutable dataset version, which experiments
    pin to for reproducible evaluations.
    """
    examples = [to_arize_example(r) for r in load_datasets(dataset_config.local_directories, dump_rendered)]
    payload = ArizeExample.to_payload(dataset_config.dataset_name, examples)
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
        len(examples),
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
            _ensure_dataset(client, dataset_config, config.debugging.dump_rendered_datasets)
