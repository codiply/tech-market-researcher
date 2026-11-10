"""Builds the list of Arize evaluators from the evaluation config."""

import json
import os
from collections.abc import Mapping
from enum import Enum
from typing import Any, Callable, cast

from opentelemetry.trace import Tracer
from phoenix.client.resources.experiments.types import EvalsEvaluator
from phoenix.evals.evaluators import ClassificationEvaluator, Evaluator, Score
from phoenix.evals.llm import LLM, PromptLike
from phoenix.evals.llm.types import ObjectGenerationMethod
from phoenix.evals.metrics import (
    CompletenessEvaluator,
    ConcisenessEvaluator,
    CorrectnessEvaluator,
    DocumentRelevanceEvaluator,
    FaithfulnessEvaluator,
    HallucinationEvaluator,
    MatchesRegex,
    PiiDetectionEvaluator,
    PrecisionRecallFScore,
    RefusalEvaluator,
    RetrievalRelevanceEvaluator,
    ToolInvocationEvaluator,
    ToolResponseHandlingEvaluator,
    ToolSelectionEvaluator,
    ToxicityEvaluator,
    UserFrictionEvaluator,
)
from pydantic import BaseModel, Field

from evaluations.arize.configuration import ArizeEvaluationConfig, BaseEvaluatorConfig

# The judge talks to the OpenAI-compatible MLflow AI Gateway endpoint. Its base URL is derived
# from the tracking server URI (see the agent's model_providers in config/agent/_common.yaml).
GATEWAY_BASE_PATH = "/gateway/mlflow/v1"

# The dataset example fields the evaluators read from, kept in sync with to_arize_example in
# bootstrapping/datasets/arize.py:
#   example output:   expected_response, expected_facts, expected_tool_calls
#   example metadata: guidelines, tags, source_documents, available_tools, tool_calls,
#                     tool_results
# Not all of these are populated yet; the extractors below degrade to empty values so
# evaluators stay runnable until the data is added to the datasets.


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


# The input/metadata extractors below are Phoenix input-mapping functions: their parameters
# are bound by name to the experiment's per-record values (input, output, expected, metadata).


def _user_message(input: Mapping[str, Any]) -> str:  # pylint: disable=redefined-builtin
    """Return the last user message of the record's conversation as a plain query string."""
    messages = input.get("messages", [])
    user_messages = [message["content"] for message in messages if message.get("role") == "user"]
    return user_messages[-1] if user_messages else ""


def _record_transcript(input: Mapping[str, Any]) -> str:  # pylint: disable=redefined-builtin
    """Render the record's messages as a plain-text transcript (without the answer)."""
    return "\n".join(f"{message['role']}: {message['content']}" for message in input.get("messages", []))


def _conversation(input: Mapping[str, Any], output: Any = "") -> str:  # pylint: disable=redefined-builtin
    """Render the record's conversation plus the agent's answer as a plain-text transcript."""
    return f"{_record_transcript(input)}\nassistant: {output}"


def _source_documents(metadata: Mapping[str, Any] | None = None) -> str:
    """Join the record's source documents (retrieved/search content) for grounding evaluators."""
    return "\n\n".join(str(document) for document in (metadata or {}).get("source_documents") or [])


def _guidelines(metadata: Mapping[str, Any] | None = None) -> str:
    """Return the record's guidelines as a bullet list."""
    return "\n".join(f"- {guideline}" for guideline in (metadata or {}).get("guidelines") or [])


def _available_tools(metadata: Mapping[str, Any] | None = None) -> str:
    """Return the record's available tool schemas as JSON."""
    return json.dumps((metadata or {}).get("available_tools") or [])


def _tool_calls(metadata: Mapping[str, Any] | None = None) -> str:
    """Return the agent's tool calls for the record as JSON."""
    return json.dumps((metadata or {}).get("tool_calls") or [])


def _tool_results(metadata: Mapping[str, Any] | None = None) -> str:
    """Return the agent's tool results for the record as JSON."""
    return json.dumps((metadata or {}).get("tool_results") or [])


def _expected_response(expected: Mapping[str, Any] | None = None) -> str:
    """Return the record's expected response as a string."""
    return str((expected or {}).get("expected_response") or "")


def _expected_facts(expected: Mapping[str, Any] | None = None) -> str:
    """Return the record's expected facts as a bullet list."""
    return "\n".join(f"- {fact}" for fact in (expected or {}).get("expected_facts") or [])


def _expected_tool_call_names(expected: Mapping[str, Any] | None = None) -> list[str]:
    """Return the names of the record's expected tool calls."""
    return [str(call["name"]) for call in (expected or {}).get("expected_tool_calls") or []]


def _tool_call_names(metadata: Mapping[str, Any] | None = None) -> list[str]:
    """Return the names of the agent's tool calls for the record."""
    return [str(call["name"]) for call in (metadata or {}).get("tool_calls") or []]


class ExactMatchEvaluator(Evaluator):
    """The exact_match code evaluator, scoped to its two input fields.

    Phoenix's built-in `exact_match` is a function evaluator: the experiment client remaps its
    inputs but passes through the extra experiment keys (input, metadata, example, ...), and
    the raw `exact_match(output, expected)` function raises on them. The same 1.0/0.0 semantics
    are therefore implemented here with a schema that only exposes `output` and `expected`.
    """

    class InputSchema(BaseModel):
        """The evaluator's input fields: the agent's answer and the record's expected response."""

        output: str = Field(description="The agent's answer.")
        expected: str = Field(description="The record's expected response.")

    def __init__(self) -> None:
        super().__init__(name="exact_match", kind="code", input_schema=self.InputSchema, direction="maximize")

    def _evaluate(self, eval_input: Mapping[str, Any]) -> list[Score]:
        return [Score(name=self.name, score=1.0 if eval_input["output"] == eval_input["expected"] else 0.0)]


# Custom judge evaluators (not part of Phoenix's built-ins): the Arize mirror of the MLflow
# scorers that consume a record's expectations, which none of the built-in LLM evaluators do.
EXPECTED_FACTS_PROMPT = """You are an expert evaluator checking whether the response covers every expected fact.

<rubric>
COVERED - Every expected fact is present in the response (rephrasing is fine), and the response
does not contradict any of them. If no expected facts are provided, answer covered.

NOT_COVERED - At least one expected fact is missing from the response or contradicted by it.
</rubric>

<data>
<query>
{{input}}
</query>

<expected_facts>
{{expected_facts}}
</expected_facts>

<response>
{{output}}
</response>
</data>

Does the response cover every expected fact without contradicting them (covered), or is at
least one expected fact missing or contradicted (not_covered)?"""

GUIDELINES_PROMPT = """You are an expert evaluator checking whether the response follows the guidelines it was given.

<rubric>
FOLLOWED - The response satisfies every provided guideline. If no guidelines are provided, answer followed.

NOT_FOLLOWED - The response violates at least one provided guideline.
</rubric>

<data>
<query>
{{input}}
</query>

<guidelines>
{{guidelines}}
</guidelines>

<response>
{{output}}
</response>
</data>

Does the response follow every provided guideline (followed), or does it violate at least one (not_followed)?"""


def _custom_evaluator(
    config: ArizeEvaluationConfig,
    name: str,
    prompt: str,
    choices: dict,
    evaluator_config: BaseEvaluatorConfig,
) -> ClassificationEvaluator:
    """Build a custom judge evaluator with the given prompt, fed through the AI Gateway."""
    return ClassificationEvaluator(
        name=name,
        llm=_judge_llm(config, evaluator_config),
        prompt_template=prompt,
        choices=choices,
        direction="maximize",
    )


EvaluatorBuilder = Callable[[ArizeEvaluationConfig], list[EvalsEvaluator]]


def _bind(evaluator: Evaluator, mapping: dict) -> EvalsEvaluator:
    """Bind an evaluator's prompt fields to record extractors and cast it to the protocol."""
    evaluator.bind(mapping)
    return cast("EvalsEvaluator", evaluator)


def _build_completeness_evaluators(config: ArizeEvaluationConfig) -> list[EvalsEvaluator]:
    """Build the `completeness` evaluator(s), if enabled."""
    if not config.evaluators.completeness.enabled:
        return []
    evaluator = CompletenessEvaluator(llm=_judge_llm(config, config.evaluators.completeness))
    return [_bind(evaluator, {"conversation": _conversation})]


def _build_conciseness_evaluators(config: ArizeEvaluationConfig) -> list[EvalsEvaluator]:
    """Build the `conciseness` evaluator(s), if enabled."""
    if not config.evaluators.conciseness.enabled:
        return []
    evaluator = ConcisenessEvaluator(llm=_judge_llm(config, config.evaluators.conciseness))
    return [_bind(evaluator, {"input": _user_message})]


def _build_correctness_evaluators(config: ArizeEvaluationConfig) -> list[EvalsEvaluator]:
    """Build the `correctness` evaluator(s), if enabled."""
    if not config.evaluators.correctness.enabled:
        return []
    evaluator = CorrectnessEvaluator(llm=_judge_llm(config, config.evaluators.correctness))
    return [_bind(evaluator, {"input": _user_message})]


def _build_document_relevance_evaluators(config: ArizeEvaluationConfig) -> list[EvalsEvaluator]:
    """Build the `document_relevance` evaluator(s), if enabled."""
    if not config.evaluators.document_relevance.enabled:
        return []
    evaluator = DocumentRelevanceEvaluator(llm=_judge_llm(config, config.evaluators.document_relevance))
    return [_bind(evaluator, {"input": _user_message, "document_text": _source_documents})]


def _build_expected_facts_evaluators(config: ArizeEvaluationConfig) -> list[EvalsEvaluator]:
    """Build the custom `expected_facts` evaluator(s), if enabled."""
    if not config.evaluators.expected_facts.enabled:
        return []
    evaluator = _custom_evaluator(
        config,
        "expected_facts",
        EXPECTED_FACTS_PROMPT,
        {"covered": 1.0, "not_covered": 0.0},
        config.evaluators.expected_facts,
    )
    return [_bind(evaluator, {"input": _user_message, "expected_facts": _expected_facts})]


def _build_exact_match_evaluators(config: ArizeEvaluationConfig) -> list[EvalsEvaluator]:
    """Build the `exact_match` code evaluator(s), if enabled. No judge LLM is needed."""
    if not config.evaluators.exact_match.enabled:
        return []
    return [_bind(ExactMatchEvaluator(), {"expected": _expected_response})]


def _build_faithfulness_evaluators(config: ArizeEvaluationConfig) -> list[EvalsEvaluator]:
    """Build the `faithfulness` evaluator(s), if enabled."""
    if not config.evaluators.faithfulness.enabled:
        return []
    evaluator = FaithfulnessEvaluator(llm=_judge_llm(config, config.evaluators.faithfulness))
    return [_bind(evaluator, {"input": _user_message, "context": _source_documents})]


def _build_guidelines_evaluators(config: ArizeEvaluationConfig) -> list[EvalsEvaluator]:
    """Build the custom `guidelines` evaluator(s), if enabled."""
    if not config.evaluators.guidelines.enabled:
        return []
    evaluator = _custom_evaluator(
        config,
        "guidelines",
        GUIDELINES_PROMPT,
        {"followed": 1.0, "not_followed": 0.0},
        config.evaluators.guidelines,
    )
    return [_bind(evaluator, {"input": _user_message, "guidelines": _guidelines})]


def _build_hallucination_evaluators(config: ArizeEvaluationConfig) -> list[EvalsEvaluator]:
    """Build the `hallucination` evaluator(s), if enabled."""
    if not config.evaluators.hallucination.enabled:
        return []
    evaluator = HallucinationEvaluator(llm=_judge_llm(config, config.evaluators.hallucination))
    # The prompt treats `input` as the full record the assistant had access to; the record's
    # messages are the best stand-in until run-time tool calls can be attached to the task output.
    return [_bind(evaluator, {"input": _record_transcript})]


def _build_matches_regex_evaluators(config: ArizeEvaluationConfig) -> list[EvalsEvaluator]:
    """Build the `matches_regex` code evaluator(s), if enabled. No judge LLM is needed."""
    evaluator_config = config.evaluators.matches_regex
    if not evaluator_config.enabled:
        return []
    if not evaluator_config.pattern:
        raise ValueError("The `matches_regex` evaluator is enabled but has no `pattern`.")
    return [cast("EvalsEvaluator", MatchesRegex(pattern=evaluator_config.pattern))]


def _build_pii_detection_evaluators(config: ArizeEvaluationConfig) -> list[EvalsEvaluator]:
    """Build the `pii_detection` evaluator(s), if enabled."""
    if not config.evaluators.pii_detection.enabled:
        return []
    evaluator = PiiDetectionEvaluator(llm=_judge_llm(config, config.evaluators.pii_detection))
    return [_bind(evaluator, {"conversation": _conversation})]


def _build_precision_recall_evaluators(config: ArizeEvaluationConfig) -> list[EvalsEvaluator]:
    """Build the `precision_recall` code evaluator(s), if enabled. No judge LLM is needed.

    Compares the record's expected tool call names to the agent's; the Phoenix experiment
    client logs only the first produced score (precision), not recall/F.
    """
    evaluator_config = config.evaluators.precision_recall
    if not evaluator_config.enabled:
        return []
    evaluator = PrecisionRecallFScore(
        beta=evaluator_config.beta,
        average=evaluator_config.average,
        positive_label=evaluator_config.positive_label,
        zero_division=evaluator_config.zero_division,
    )
    return [_bind(evaluator, {"expected": _expected_tool_call_names, "output": _tool_call_names})]


def _build_refusal_evaluators(config: ArizeEvaluationConfig) -> list[EvalsEvaluator]:
    """Build the `refusal` evaluator(s), if enabled."""
    if not config.evaluators.refusal.enabled:
        return []
    evaluator = RefusalEvaluator(llm=_judge_llm(config, config.evaluators.refusal))
    return [_bind(evaluator, {"input": _user_message})]


def _build_retrieval_relevance_evaluators(config: ArizeEvaluationConfig) -> list[EvalsEvaluator]:
    """Build the `retrieval_relevance` evaluator(s), if enabled."""
    if not config.evaluators.retrieval_relevance.enabled:
        return []
    evaluator = RetrievalRelevanceEvaluator(llm=_judge_llm(config, config.evaluators.retrieval_relevance))
    return [_bind(evaluator, {"input": _user_message, "context": _source_documents})]


def _build_tool_invocation_evaluators(config: ArizeEvaluationConfig) -> list[EvalsEvaluator]:
    """Build the `tool_invocation` evaluator(s), if enabled."""
    if not config.evaluators.tool_invocation.enabled:
        return []
    evaluator = ToolInvocationEvaluator(llm=_judge_llm(config, config.evaluators.tool_invocation))
    return [
        _bind(
            evaluator,
            {"input": _user_message, "available_tools": _available_tools, "tool_selection": _tool_calls},
        )
    ]


def _build_tool_response_handling_evaluators(config: ArizeEvaluationConfig) -> list[EvalsEvaluator]:
    """Build the `tool_response_handling` evaluator(s), if enabled."""
    if not config.evaluators.tool_response_handling.enabled:
        return []
    evaluator = ToolResponseHandlingEvaluator(llm=_judge_llm(config, config.evaluators.tool_response_handling))
    return [
        _bind(
            evaluator,
            {"input": _user_message, "tool_call": _tool_calls, "tool_result": _tool_results},
        )
    ]


def _build_tool_selection_evaluators(config: ArizeEvaluationConfig) -> list[EvalsEvaluator]:
    """Build the `tool_selection` evaluator(s), if enabled."""
    if not config.evaluators.tool_selection.enabled:
        return []
    evaluator = ToolSelectionEvaluator(llm=_judge_llm(config, config.evaluators.tool_selection))
    return [
        _bind(
            evaluator,
            {"input": _user_message, "available_tools": _available_tools, "tool_selection": _tool_calls},
        )
    ]


def _build_toxicity_evaluators(config: ArizeEvaluationConfig) -> list[EvalsEvaluator]:
    """Build the `toxicity` evaluator(s), if enabled."""
    if not config.evaluators.toxicity.enabled:
        return []
    evaluator = ToxicityEvaluator(llm=_judge_llm(config, config.evaluators.toxicity))
    return [_bind(evaluator, {"text": "output"})]


def _build_user_friction_evaluators(config: ArizeEvaluationConfig) -> list[EvalsEvaluator]:
    """Build the `user_friction` evaluator(s), if enabled."""
    if not config.evaluators.user_friction.enabled:
        return []
    evaluator = UserFrictionEvaluator(llm=_judge_llm(config, config.evaluators.user_friction))
    return [_bind(evaluator, {"conversation": _conversation, "user_message": _user_message})]


class EvaluatorType(str, Enum):
    """The evaluator types; one per entry of the `evaluators` config."""

    COMPLETENESS = "completeness"
    CONCISENESS = "conciseness"
    CORRECTNESS = "correctness"
    DOCUMENT_RELEVANCE = "document_relevance"
    EXPECTED_FACTS = "expected_facts"
    EXACT_MATCH = "exact_match"
    FAITHFULNESS = "faithfulness"
    GUIDELINES = "guidelines"
    HALLUCINATION = "hallucination"
    MATCHES_REGEX = "matches_regex"
    PII_DETECTION = "pii_detection"
    PRECISION_RECALL = "precision_recall"
    REFUSAL = "refusal"
    RETRIEVAL_RELEVANCE = "retrieval_relevance"
    TOOL_INVOCATION = "tool_invocation"
    TOOL_RESPONSE_HANDLING = "tool_response_handling"
    TOOL_SELECTION = "tool_selection"
    TOXICITY = "toxicity"
    USER_FRICTION = "user_friction"


# The evaluator builder per evaluator type, referenced by name. Keep these in alphabetical order.
_EVALUATOR_BUILDERS: dict[EvaluatorType, EvaluatorBuilder] = {
    EvaluatorType.COMPLETENESS: _build_completeness_evaluators,
    EvaluatorType.CONCISENESS: _build_conciseness_evaluators,
    EvaluatorType.CORRECTNESS: _build_correctness_evaluators,
    EvaluatorType.DOCUMENT_RELEVANCE: _build_document_relevance_evaluators,
    EvaluatorType.EXPECTED_FACTS: _build_expected_facts_evaluators,
    EvaluatorType.EXACT_MATCH: _build_exact_match_evaluators,
    EvaluatorType.FAITHFULNESS: _build_faithfulness_evaluators,
    EvaluatorType.GUIDELINES: _build_guidelines_evaluators,
    EvaluatorType.HALLUCINATION: _build_hallucination_evaluators,
    EvaluatorType.MATCHES_REGEX: _build_matches_regex_evaluators,
    EvaluatorType.PII_DETECTION: _build_pii_detection_evaluators,
    EvaluatorType.PRECISION_RECALL: _build_precision_recall_evaluators,
    EvaluatorType.REFUSAL: _build_refusal_evaluators,
    EvaluatorType.RETRIEVAL_RELEVANCE: _build_retrieval_relevance_evaluators,
    EvaluatorType.TOOL_INVOCATION: _build_tool_invocation_evaluators,
    EvaluatorType.TOOL_RESPONSE_HANDLING: _build_tool_response_handling_evaluators,
    EvaluatorType.TOOL_SELECTION: _build_tool_selection_evaluators,
    EvaluatorType.TOXICITY: _build_toxicity_evaluators,
    EvaluatorType.USER_FRICTION: _build_user_friction_evaluators,
}


# Adding an evaluator type without a builder (or vice versa) would silently break the
# enumeration in build_evaluators_list.
if set(_EVALUATOR_BUILDERS) != set(EvaluatorType):
    raise ValueError("_EVALUATOR_BUILDERS must have exactly one builder per EvaluatorType.")


def build_evaluators_list(config: ArizeEvaluationConfig) -> list[EvalsEvaluator]:
    """Build the evaluators enabled in the config; the judge goes through the AI Gateway."""
    evaluators: list[EvalsEvaluator] = []
    for evaluator_type in EvaluatorType:
        evaluators.extend(_EVALUATOR_BUILDERS[evaluator_type](config))
    if not evaluators:
        raise ValueError("No evaluators are enabled in the evaluation config.")
    return evaluators
