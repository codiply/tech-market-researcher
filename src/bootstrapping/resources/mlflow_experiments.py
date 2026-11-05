"""MLflow experiment and dataset bootstrapping."""

import logging
import os

import mlflow
from mlflow.exceptions import MlflowException

from bootstrapping.common.configuration import BootstrapConfig, MlflowDatasetConfig
from bootstrapping.common.datasets import load_records, record_input_hash
from bootstrapping.common.exceptions import BootstrapError

logger = logging.getLogger(__name__)


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
    records = load_records(dataset_config)
    dataset = _find_dataset(client, experiment_id, dataset_config.dataset_name)
    if dataset is None:
        dataset = client.create_dataset(dataset_config.dataset_name, experiment_id)
        logger.info("Created dataset '%s' in experiment '%s'.", dataset_config.dataset_name, dataset_config.experiment)
    else:
        logger.info("Dataset '%s' already exists - reusing.", dataset_config.dataset_name)
    dataset.merge_records([record.raw for record in records])
    logger.info("Synced %d records to dataset '%s'.", len(records), dataset_config.dataset_name)
    expected = {record_input_hash(record.inputs) for record in records}
    stale_ids = [r.dataset_record_id for r in dataset.records if record_input_hash(r.inputs) not in expected]
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
