"""MLflow evaluation runner."""

import os
from typing import Any

import mlflow
from langgraph.graph.state import CompiledStateGraph
from mlflow.genai.scorers import ExpectationsGuidelines, Scorer

from evaluations.common.runtime_context import build_runtime_context
from evaluations.mlflow.configuration import MlflowEvaluationConfig

Agent = CompiledStateGraph[Any, Any]


def _find_dataset(client: mlflow.MlflowClient, experiment_id: str, name: str):
    """Return the dataset with the given name in the experiment."""
    for dataset in client.search_datasets(experiment_ids=[experiment_id]):
        if dataset.name == name:
            return dataset
    raise ValueError(f"Dataset '{name}' not found in experiment with id {experiment_id}.")


def _predict_fn_for(agent: Agent, messages: list[dict[str, str]], context: dict[str, str]) -> str:
    """Run one agent turn for a dataset record and return the final response text."""
    runtime_context = build_runtime_context(context)
    result = agent.invoke({"messages": messages}, context=runtime_context)
    return result["messages"][-1].content


def mlflow_evaluate(config: MlflowEvaluationConfig, agent: Agent) -> None:
    """Evaluate the agent against the MLflow dataset named by the config using its guidelines."""
    mlflow.set_tracking_uri(os.environ["MLFLOW_SERVER_URI"].rstrip("/"))
    experiment = mlflow.set_experiment(config.experiment_name)
    dataset = _find_dataset(mlflow.MlflowClient(), experiment.experiment_id, config.dataset_name)

    data = [
        {"inputs": record.inputs, "expectations": record.expectations}
        for record in dataset.records
        if record.expectations and record.expectations.get("guidelines")
    ]

    def predict_fn(messages: list[dict[str, str]], context: dict[str, str]) -> str:
        """Run one agent turn for a dataset record and return the final response text."""
        return _predict_fn_for(agent, messages, context)

    # The judge runs against the AI Gateway endpoint of the active tracking server.
    scorers: list[Scorer] = [ExpectationsGuidelines(model=f"gateway:/{config.judge_model_name}")]
    mlflow.genai.evaluate(data=data, predict_fn=predict_fn, scorers=scorers)
