"""MLflow experiment and dataset bootstrapping."""

import hashlib
import json
import logging
import os
from pathlib import Path

import mlflow
from mlflow.exceptions import MlflowException

from bootstrapping.common.configuration import BootstrapConfig, MlflowDatasetConfig
from bootstrapping.common.exceptions import BootstrapError

logger = logging.getLogger(__name__)

DATASETS_DIR = Path("assets") / "datasets"


def _input_hash(inputs: dict) -> str:
    """Hash of the record inputs, matching the server's dedup key."""
    return hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()


def _load_records(dataset_config: MlflowDatasetConfig) -> list[dict]:
    """Load one record per JSON file across the configured local directories.

    The directory names are relative to assets/datasets/.
    """
    records: list[dict] = []
    for name in dataset_config.local_directories:
        directory = DATASETS_DIR / name
        if not directory.is_dir():
            raise BootstrapError(f"Dataset directory '{directory}' does not exist.")
        for path in sorted(directory.glob("*.json")):
            records.append(json.loads(path.read_text(encoding="utf-8")))
    if not records:
        raise BootstrapError(f"No records found for dataset '{dataset_config.dataset_name}'.")
    return records


def _ensure_experiment(client: mlflow.MlflowClient, name: str) -> str:
    """Return the experiment id, creating the experiment if it does not exist."""
    experiment = client.get_experiment_by_name(name)
    if experiment is not None:
        logger.info("Experiment '%s' already exists - reusing.", name)
        return experiment.experiment_id
    experiment_id = client.create_experiment(name)
    logger.info("Created experiment '%s'.", name)
    return experiment_id


def _find_dataset(
    client: mlflow.MlflowClient, experiment_id: str, name: str
) -> mlflow.entities.EvaluationDataset | None:
    """Return the dataset with the given name in the experiment, or None."""
    for dataset in client.search_datasets(experiment_ids=[experiment_id]):
        if dataset.name == name:
            return dataset
    return None


def _ensure_dataset(client: mlflow.MlflowClient, dataset_config: MlflowDatasetConfig, experiment_id: str) -> None:
    """Create the dataset if missing, then sync its records with the configured ones.

    Records are upserted by the server (keyed on the inputs hash), so re-runs do not
    create duplicates. Records no longer present in the folders are pruned; pruning
    breaks any run/evaluation links to those record ids.
    """
    records = _load_records(dataset_config)
    dataset = _find_dataset(client, experiment_id, dataset_config.dataset_name)
    if dataset is None:
        dataset = client.create_dataset(dataset_config.dataset_name, experiment_id)
        logger.info("Created dataset '%s' in experiment '%s'.", dataset_config.dataset_name, dataset_config.experiment)
    else:
        logger.info("Dataset '%s' already exists - reusing.", dataset_config.dataset_name)
    dataset.merge_records(records)
    logger.info("Synced %d records to dataset '%s'.", len(records), dataset_config.dataset_name)
    expected = {_input_hash(record["inputs"]) for record in records}
    stale_ids = [r.dataset_record_id for r in dataset.records if _input_hash(r.inputs) not in expected]
    if stale_ids:
        deleted = dataset.delete_records(stale_ids)
        logger.info(
            "Pruned %d records from dataset '%s' no longer present in the folders.",
            deleted,
            dataset_config.dataset_name,
        )


def bootstrap_mlflow_experiments(config: BootstrapConfig) -> None:
    """Provision MLflow experiments and their evaluation datasets."""
    mlflow.set_tracking_uri(os.environ["MLFLOW_SERVER_URI"].rstrip("/"))
    client = mlflow.MlflowClient()
    try:
        for dataset_config in config.mlflow.datasets:
            experiment_id = _ensure_experiment(client, dataset_config.experiment)
            _ensure_dataset(client, dataset_config, experiment_id)
    except MlflowException as e:
        raise BootstrapError(str(e)) from e
