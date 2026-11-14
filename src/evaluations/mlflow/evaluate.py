"""MLflow evaluation runner."""

import asyncio
import os
import threading
from datetime import datetime, timezone

import mlflow
from mlflow.genai.scorers import Scorer

from evaluations.common.constants import EVALS_RUN_NAME_PREFIX
from evaluations.common.runtime_context import build_runtime_context
from evaluations.mlflow.configuration import HarnessConfig, MlflowEvaluationConfig
from evaluations.mlflow.scorers import build_scorers_list
from tech_market_researcher.agents.main import TechMarketResearcherAgent


def _find_dataset(client: mlflow.MlflowClient, experiment_id: str, name: str):
    """Return the dataset with the given name in the experiment."""
    for dataset in client.search_datasets(experiment_ids=[experiment_id]):
        if dataset.name == name:
            return dataset
    raise ValueError(f"Dataset '{name}' not found in experiment with id {experiment_id}.")


async def _predict_fn_for(
    agent: TechMarketResearcherAgent, messages: list[dict[str, str]], context: dict[str, str]
) -> str:
    """Run one agent turn for a dataset record and return the final response text."""
    runtime_context = build_runtime_context(context)
    result = await agent.ainvoke(messages, runtime_context)
    return result["messages"][-1].content


def _run_name() -> str:
    """Build the MLflow run name for an evaluation, e.g. evals-20261108-103015."""
    return f"{EVALS_RUN_NAME_PREFIX}{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"


def _apply_harness_env(harness: HarnessConfig) -> None:
    """Export MLflow evaluation-harness env vars (MLFLOW_GENAI_EVAL_*) from the config.

    Unset (None) values are left to the MLflow defaults.
    """
    values = {
        "MLFLOW_GENAI_EVAL_MAX_WORKERS": harness.max_workers,
        "MLFLOW_GENAI_EVAL_MAX_SCORER_WORKERS": harness.max_scorer_workers,
        "MLFLOW_GENAI_EVAL_PREDICT_RATE_LIMIT": harness.predict_rate_limit,
        "MLFLOW_GENAI_EVAL_SCORER_RATE_LIMIT": harness.scorer_rate_limit,
        "MLFLOW_GENAI_EVAL_MAX_RETRIES": harness.max_retries,
    }
    for name, value in values.items():
        if value is not None:
            os.environ[name] = str(value)


def mlflow_evaluate(config: MlflowEvaluationConfig, agent: TechMarketResearcherAgent) -> None:
    """Evaluate the agent against the MLflow dataset named by the config using its guidelines."""
    _apply_harness_env(config.harness)
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

        # MLflow's evaluation harness runs each record on its own short-lived event loop, but
        # the agent's async model client caches a connection pool bound to the loop it first
        # runs on; reusing that pool from a closed loop raises `RuntimeError: Event loop is
        # closed`. So every agent call is driven by one dedicated, long-lived loop instead.
        agent_loop = asyncio.new_event_loop()
        threading.Thread(target=agent_loop.run_forever, daemon=True).start()

        def predict_fn(messages: list[dict[str, str]], context: dict[str, str]) -> str:
            """Run one agent turn for a dataset record and return the final response text."""
            return asyncio.run_coroutine_threadsafe(_predict_fn_for(agent, messages, context), agent_loop).result()

        # The judge runs against the AI Gateway endpoint of the active tracking server.
        scorers: list[Scorer] = build_scorers_list(config)
        mlflow.genai.evaluate(data=data, predict_fn=predict_fn, scorers=scorers)
