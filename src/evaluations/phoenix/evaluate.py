"""Phoenix evaluation runner."""

import asyncio
import os
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from typing import Any

from phoenix.client import AsyncClient

from evaluations.common.constants import EVALS_RUN_NAME_PREFIX
from evaluations.common.runtime_context import build_runtime_context
from evaluations.phoenix.configuration import PhoenixEvaluationConfig
from evaluations.phoenix.evaluators import build_evaluators_list
from tech_market_researcher.agents.main import TechMarketResearcherAgent, create_agent
from tech_market_researcher.core.arguments import AgentCommandLineArguments
from tech_market_researcher.core.configuration import ApplicationConfig

# The Phoenix REST API base; the UI/collector ports are documented in .env.
PHOENIX_SERVER_URI_ENV = "PHOENIX_SERVER_URI"


async def _predict_fn_for(agent: TechMarketResearcherAgent, record_input: Mapping[str, Any]) -> str:
    """Run one agent turn for a dataset record and return the final response text."""
    messages = record_input["messages"]
    context = record_input.get("context", {})
    runtime_context = build_runtime_context(context)
    result = await agent.ainvoke(messages, runtime_context)
    return result["messages"][-1].content


def _run_name() -> str:
    """Build the experiment name for an evaluation, e.g. evals-20261108-103015."""
    return f"{EVALS_RUN_NAME_PREFIX}{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"


def phoenix_evaluate(
    config: PhoenixEvaluationConfig,
    agent_args: AgentCommandLineArguments,
    config_mutations_hooks: list[Callable[[ApplicationConfig], None]],
) -> None:
    """Evaluate the agent against the Phoenix dataset named by the config using its evaluators."""
    evaluators = build_evaluators_list(config)

    # Phoenix's sync API runs serially; the async API's `concurrency` parameter controls how
    # many tasks and evaluations run in parallel. Unset/0 leaves the Phoenix default (3).
    # `timeout` is the per-task execution timeout in seconds; unset leaves the Phoenix
    # default (60), which cancels and re-queues agent turns that take longer than that.
    extra: dict[str, Any] = {}
    if config.harness.max_concurrent:
        extra["concurrency"] = config.harness.max_concurrent
    if config.harness.task_timeout:
        extra["timeout"] = config.harness.task_timeout

    async def run() -> None:
        # The checkpointer binds to the loop it is built on, so build the agent on the same loop.
        agent = await create_agent(agent_args, config_mutations_hooks)

        async def task(record_input: Mapping[str, Any]) -> str:
            """Run one agent turn for a dataset record and return the final response text."""
            return await _predict_fn_for(agent, record_input)

        client = AsyncClient(base_url=os.environ[PHOENIX_SERVER_URI_ENV].rstrip("/"))
        dataset = await client.datasets.get_dataset(dataset=config.dataset_name)
        async with agent:
            await client.experiments.run_experiment(
                dataset=dataset,
                task=task,
                evaluators=evaluators,
                experiment_name=_run_name(),
                **extra,
            )

    asyncio.run(run())
