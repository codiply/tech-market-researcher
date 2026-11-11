"""Arize (Phoenix) evaluation runner."""

import asyncio
import os
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

from phoenix.client import AsyncClient

from evaluations.arize.configuration import ArizeEvaluationConfig
from evaluations.arize.evaluators import build_evaluators_list
from evaluations.common.constants import EVALS_RUN_NAME_PREFIX
from evaluations.common.runtime_context import build_runtime_context
from tech_market_researcher.agents.main import TechMarketResearcherAgent

# The Phoenix REST API base; the UI/collector ports are documented in .env.
ARIZE_SERVER_URI_ENV = "ARIZE_SERVER_URI"


def _predict_fn_for(agent: TechMarketResearcherAgent, record_input: Mapping[str, Any]) -> str:
    """Run one agent turn for a dataset record and return the final response text."""
    messages = record_input["messages"]
    context = record_input.get("context", {})
    runtime_context = build_runtime_context(context)
    result = agent.invoke(messages, runtime_context)
    return result["messages"][-1].content


def _run_name() -> str:
    """Build the experiment name for an evaluation, e.g. evals-20261108-103015."""
    return f"{EVALS_RUN_NAME_PREFIX}{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"


def arize_evaluate(config: ArizeEvaluationConfig, agent: TechMarketResearcherAgent) -> None:
    """Evaluate the agent against the Arize dataset named by the config using its evaluators."""
    evaluators = build_evaluators_list(config)

    def task(record_input: Mapping[str, Any]) -> str:
        """Run one agent turn for a dataset record and return the final response text."""
        return _predict_fn_for(agent, record_input)

    # Phoenix's sync API runs serially; the async API's `concurrency` parameter controls how
    # many tasks and evaluations run in parallel. Unset/0 leaves the Phoenix default (3).
    extra: dict[str, Any] = {}
    if config.harness.max_concurrent:
        extra["concurrency"] = config.harness.max_concurrent

    async def run() -> None:
        client = AsyncClient(base_url=os.environ[ARIZE_SERVER_URI_ENV].rstrip("/"))
        dataset = await client.datasets.get_dataset(dataset=config.dataset_name)
        await client.experiments.run_experiment(
            dataset=dataset,
            task=task,
            evaluators=evaluators,
            experiment_name=_run_name(),
            **extra,
        )

    asyncio.run(run())
