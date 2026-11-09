"""Builds the list of Arize evaluators from the evaluation config."""

import os
from collections.abc import Mapping
from typing import Any, cast

from opentelemetry.trace import Tracer
from phoenix.client.resources.experiments.types import EvalsEvaluator
from phoenix.evals.evaluators import ClassificationEvaluator
from phoenix.evals.llm import LLM, PromptLike
from phoenix.evals.llm.types import ObjectGenerationMethod
from phoenix.evals.metrics import (
    ConcisenessEvaluator,
    CorrectnessEvaluator,
    HallucinationEvaluator,
    RefusalEvaluator,
)

from evaluations.arize.configuration import ArizeEvaluationConfig, BaseEvaluatorConfig

# The judge talks to the OpenAI-compatible MLflow AI Gateway endpoint. Its base URL is derived
# from the tracking server URI (see the agent's model_providers in config/agent/_common.yaml).
GATEWAY_BASE_PATH = "/gateway/mlflow/v1"


class JudgeLLM(LLM):
    """Gateway judge LLM that generates objects via tool calling, never structured output.

    Gateway models such as Bedrock's Nova ignore the strict json_schema response_format and
    return JSON wrapped in markdown fences. The adapter's AUTO discovery only falls back to
    tool calling on a BadRequestError, which never happens, so its json.loads crashes; tool
    calling is respected by all gateway models.
    """

    def generate_object(
        self, prompt: PromptLike, schema: dict[str, Any], tracer: Tracer | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        kwargs.pop("method", None)
        return super().generate_object(
            prompt, schema, tracer=tracer, method=ObjectGenerationMethod.TOOL_CALLING, **kwargs
        )


def _gateway_base_url() -> str:
    """Return the OpenAI-compatible gateway base URL as seen from this host."""
    return os.environ["MLFLOW_SERVER_URI"].rstrip("/") + GATEWAY_BASE_PATH


def _judge_llm(config: ArizeEvaluationConfig, evaluator_config: BaseEvaluatorConfig) -> LLM:
    """Return the judge LLM for an evaluator, honoring any per-evaluator model override."""
    return JudgeLLM(
        provider="openai",
        model=evaluator_config.model_name_override or config.judge_model_name,
        base_url=_gateway_base_url(),
        api_key="dummy",
    )


def _user_message(input: Mapping[str, Any]) -> str:  # pylint: disable=redefined-builtin
    """Extract the last user message from a dataset record's input.

    The Phoenix built-in evaluators expect a plain string query, but our records carry the full
    message list under input["messages"], so the mapping unwraps it into a single query string.
    """
    messages = input.get("messages", [])
    user_messages = [message["content"] for message in messages if message.get("role") == "user"]
    return user_messages[-1] if user_messages else ""


def _map_input(evaluator: ClassificationEvaluator) -> EvalsEvaluator:
    """Bind an evaluator's `input` field to the record's user query, unwrapping the message list.

    Phoenix's built-in evaluators satisfy the experiment evaluator protocol at runtime, but mypy
    does not infer it structurally, so the result is cast.
    """
    evaluator.bind({"input": _user_message})
    return cast("EvalsEvaluator", evaluator)


# The built-in LLM evaluators are defined in Phoenix here:
# https://github.com/Arize-ai/phoenix/blob/main/packages/phoenix-evals/src/phoenix/evals/metrics/
def build_evaluators_list(config: ArizeEvaluationConfig) -> list[EvalsEvaluator]:
    """Build the evaluators enabled in the config; the judge goes through the AI Gateway."""
    evaluators: list[EvalsEvaluator] = []
    # Keep these in alphabetical order.
    if config.evaluators.conciseness.enabled:
        evaluators.append(_map_input(ConcisenessEvaluator(llm=_judge_llm(config, config.evaluators.conciseness))))
    if config.evaluators.correctness.enabled:
        evaluators.append(_map_input(CorrectnessEvaluator(llm=_judge_llm(config, config.evaluators.correctness))))
    if config.evaluators.hallucination.enabled:
        evaluators.append(_map_input(HallucinationEvaluator(llm=_judge_llm(config, config.evaluators.hallucination))))
    if config.evaluators.refusal.enabled:
        evaluators.append(_map_input(RefusalEvaluator(llm=_judge_llm(config, config.evaluators.refusal))))
    if not evaluators:
        raise ValueError("No evaluators are enabled in the evaluation config.")
    return evaluators
