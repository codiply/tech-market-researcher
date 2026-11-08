"""MLflow evaluation runner."""

import os
from datetime import datetime, timezone

import mlflow
from mlflow.genai.scorers import Scorer

from evaluations.common.constants import EVALS_RUN_NAME_PREFIX
from evaluations.common.runtime_context import build_runtime_context
from evaluations.mlflow.configuration import MlflowEvaluationConfig
from evaluations.mlflow.scorers import build_scorers_list
from tech_market_researcher.agents.main import TechMarketResearcherAgent


def _find_dataset(client: mlflow.MlflowClient, experiment_id: str, name: str):
    """Return the dataset with the given name in the experiment."""
    for dataset in client.search_datasets(experiment_ids=[experiment_id]):
        if dataset.name == name:
            return dataset
    raise ValueError(f"Dataset '{name}' not found in experiment with id {experiment_id}.")


def _predict_fn_for(agent: TechMarketResearcherAgent, messages: list[dict[str, str]], context: dict[str, str]) -> str:
    """Run one agent turn for a dataset record and return the final response text."""
    runtime_context = build_runtime_context(context)
    result = agent.invoke(messages, runtime_context)
    return result["messages"][-1].content


def _run_name() -> str:
    """Build the MLflow run name for an evaluation, e.g. evals-20261108-103015."""
    return f"{EVALS_RUN_NAME_PREFIX}{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"


def mlflow_evaluate(config: MlflowEvaluationConfig, agent: TechMarketResearcherAgent) -> None:
    """Evaluate the agent against the MLflow dataset named by the config using its guidelines."""
    mlflow.set_tracking_uri(os.environ["MLFLOW_SERVER_URI"].rstrip("/"))
    experiment = mlflow.set_experiment(config.experiment_name)
    # The run is started explicitly so its name is ours; genai.evaluate reuses the active run.
    with mlflow.start_run(run_name=_run_name()):
        dataset = _find_dataset(mlflow.MlflowClient(), experiment.experiment_id, config.dataset_name)

        data = [
            {"inputs": record.inputs, "expectations": record.expectations}
            for record in dataset.records
            if record.expectations
        ]

        def predict_fn(messages: list[dict[str, str]], context: dict[str, str]) -> str:
            """Run one agent turn for a dataset record and return the final response text."""
            return _predict_fn_for(agent, messages, context)

        # The judge runs against the AI Gateway endpoint of the active tracking server.
        scorers: list[Scorer] = build_scorers_list(config)
        mlflow.genai.evaluate(data=data, predict_fn=predict_fn, scorers=scorers)
