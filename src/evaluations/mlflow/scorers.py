"""Builds the list of MLflow scorers from the evaluation config."""

from enum import Enum
from typing import Any, Callable

from mlflow.genai.judges import meets_guidelines
from mlflow.genai.scorers import (
    Completeness,
    Correctness,
    Equivalence,
    ExpectationsGuidelines,
    Fluency,
    Guidelines,
    RelevanceToQuery,
    RetrievalGroundedness,
    RetrievalRelevance,
    RetrievalSufficiency,
    Safety,
    Scorer,
    Summarization,
    ToolCallCorrectness,
    ToolCallEfficiency,
    scorer,
)
from mlflow.genai.utils.trace_utils import parse_inputs_to_str, parse_outputs_to_str

from evaluations.common.fields import get_field, get_field_list
from evaluations.mlflow.configuration import BaseScorerConfig, MlflowEvaluationConfig
from tech_market_researcher.core.common import BasePydanticModel

# Logged per record for scorers a dataset record cannot satisfy; MLflow excludes any
# non-yes/no string from the aggregated run metrics, so the mean covers applicable
# records only.
NOT_APPLICABLE = "not_applicable"

ApplicabilityFunction = Callable[..., bool]  # (inputs=..., outputs=..., expectations=..., trace=...) -> bool

# A scorers builder builds the wrapped scorers for a scorer type from the config, end-to-end
# (enabled check, construction, applicability); it may return more than one (e.g. one
# `Guidelines` scorer per configured instance).
ScorersBuilder = Callable[[MlflowEvaluationConfig], list[Scorer]]

# Per-invocation client-side timeout in seconds for all scorers, enforced by the MLflow client.
# 0 disables it (the gateway timeout is then the backstop).
SCORER_TIMEOUT_SECONDS = 0


class ScorerType(str, Enum):
    """The scorer types; one per entry of the `scorers` config."""

    COMPLETENESS = "completeness"
    CORRECTNESS = "correctness"
    EQUIVALENCE = "equivalence"
    EXPECTATIONS_GUIDELINES = "expectations_guidelines"
    FLUENCY = "fluency"
    GUIDELINES = "guidelines"
    RELEVANCE_TO_QUERY = "relevance_to_query"
    RETRIEVAL_GROUNDEDNESS = "retrieval_groundedness"
    RETRIEVAL_RELEVANCE = "retrieval_relevance"
    RETRIEVAL_SUFFICIENCY = "retrieval_sufficiency"
    SAFETY = "safety"
    SOURCE_DOCUMENT_GROUNDEDNESS = "source_document_groundedness"
    SUMMARIZATION = "summarization"
    TOOL_CALL_CORRECTNESS = "tool_call_correctness"
    TOOL_CALL_EFFICIENCY = "tool_call_efficiency"


def _always_applicable(**_kwargs: Any) -> bool:
    return True


def _has_expectation(expectations: dict[str, Any] | None, field: str) -> bool:
    return bool(get_field(expectations, field))


def _any_expectation(*fields: str) -> ApplicabilityFunction:
    """Return an applicability function that is true when any of the given expectation fields is present."""

    def applicable(expectations=None, **_kwargs: Any) -> bool:
        return any(_has_expectation(expectations, field) for field in fields)

    return applicable


def _all_expectations(*fields: str) -> ApplicabilityFunction:
    """Return an applicability function that is true when all of the given expectation fields are present."""

    def applicable(expectations=None, **_kwargs: Any) -> bool:
        return all(_has_expectation(expectations, field) for field in fields)

    return applicable


def _requires_retriever(trace=None, **_kwargs: Any) -> bool:
    """Applicable only when the trace contains at least one RETRIEVER span."""
    if trace is None:
        return False
    return any(s.attributes.get("mlflow.spanType") == "RETRIEVER" for s in trace.search_spans())


def _judge_applicable(question: str, model: str) -> ApplicabilityFunction:
    """Return an applicability function that asks the judge model whether the record is applicable.

    The judge is asked a yes/no question about the record's request and response via the same
    built-in guidelines judge the other scorers use; a record the judge deems inapplicable is
    scored ``not_applicable``.
    """

    def applicable(inputs=None, outputs=None, **_kwargs: Any) -> bool:
        feedback = meets_guidelines(
            guidelines=[question],
            context={
                "request": parse_inputs_to_str(inputs),
                "response": parse_outputs_to_str(outputs),
            },
            name="applicability",
            model=model,
        )
        return str(feedback.value).strip().lower().startswith("yes")

    return applicable


def wrap(inner: Scorer, applicable: ApplicabilityFunction) -> Scorer:
    """Delegate to ``inner`` unless the record is not applicable.

    A non-applicable record gets a ``NOT_APPLICABLE`` assessment, which MLflow logs
    per record but excludes from the aggregated run metrics; the wrapper keeps the
    inner scorer's name, description, aggregations and timeout so metric keys are
    unchanged.
    """

    @scorer(
        name=inner.name,
        description=inner.description,
        aggregations=inner.aggregations,
        timeout=inner.timeout,
    )
    def conditional_scorer(inputs=None, outputs=None, expectations=None, trace=None):
        if not applicable(inputs=inputs, outputs=outputs, expectations=expectations, trace=trace):
            return NOT_APPLICABLE
        return inner.run(
            inputs=inputs,
            outputs=outputs,
            expectations=expectations,
            trace=trace,
        )

    return conditional_scorer


def _default_judge_model(config: MlflowEvaluationConfig) -> str:
    """Return the config's default judge model, as the AI Gateway model name."""
    return f"gateway:/{config.judge_model_name}"


def _judge_model(config: MlflowEvaluationConfig, scorer_config: BaseScorerConfig) -> str:
    """Return the judge model for a scorer, honoring any per-scorer override."""
    return f"gateway:/{scorer_config.model_name_override or config.judge_model_name}"


class JudgeContext(BasePydanticModel):
    """The record fields a context builder maps to the judge question's variables."""

    expectations: dict[str, Any] | None
    response: str


JudgeContextBuilder = Callable[[JudgeContext], dict[str, str]]


def _custom_judge_scorer(
    name: str,
    description: str,
    question: str,
    context_builder: JudgeContextBuilder,
    model: str,
) -> Scorer:
    """Build a yes/no LLM-judge scorer over the record's fields, like the built-in guidelines judges.

    ``context_builder`` maps the record's expectations and response to the named variables the
    judge question refers to; the scorer returns the judge's yes/no answer.
    """

    @scorer(name=name, description=description, timeout=SCORER_TIMEOUT_SECONDS)
    def judge(_inputs=None, outputs=None, expectations=None, _trace=None):
        # The judge scores only the expectations and the response, so the inputs and the
        # trace are unused by design (the underscore prefix keeps pylint happy).
        context = context_builder(JudgeContext(expectations=expectations, response=parse_outputs_to_str(outputs)))
        feedback = meets_guidelines(guidelines=[question], context=context, name=name, model=model)
        return str(feedback.value)

    return judge


# The built-in scorers are defined in MLflow here:
# https://raw.githubusercontent.com/mlflow/mlflow/refs/heads/master/mlflow/genai/scorers/builtin_scorers.py
# Keep these in alphabetical order.
def _build_completeness_scorers(config: MlflowEvaluationConfig) -> list[Scorer]:
    """Build the `completeness` scorer(s), if enabled."""
    if not config.scorers.completeness.enabled:
        return []
    return [
        wrap(
            Completeness(
                model=_judge_model(config, config.scorers.completeness),
                timeout=SCORER_TIMEOUT_SECONDS,
            ),
            _always_applicable,
        )
    ]


def _build_correctness_scorers(config: MlflowEvaluationConfig) -> list[Scorer]:
    """Build the `correctness` scorer(s), if enabled."""
    if not config.scorers.correctness.enabled:
        return []
    return [
        wrap(
            Correctness(
                model=_judge_model(config, config.scorers.correctness),
                timeout=SCORER_TIMEOUT_SECONDS,
            ),
            _any_expectation("expected_response", "expected_facts"),
        )
    ]


def _build_equivalence_scorers(config: MlflowEvaluationConfig) -> list[Scorer]:
    """Build the `equivalence` scorer(s), if enabled."""
    if not config.scorers.equivalence.enabled:
        return []
    return [
        wrap(
            Equivalence(
                model=_judge_model(config, config.scorers.equivalence),
                inference_params=config.scorers.equivalence.inference_params,
                timeout=SCORER_TIMEOUT_SECONDS,
            ),
            _any_expectation("expected_response"),
        )
    ]


def _build_expectations_guidelines_scorers(config: MlflowEvaluationConfig) -> list[Scorer]:
    """Build the `expectations_guidelines` scorer(s), if enabled."""
    if not config.scorers.expectations_guidelines.enabled:
        return []
    return [
        wrap(
            ExpectationsGuidelines(
                model=_judge_model(config, config.scorers.expectations_guidelines),
                timeout=SCORER_TIMEOUT_SECONDS,
            ),
            _any_expectation("guidelines"),
        )
    ]


def _build_fluency_scorers(config: MlflowEvaluationConfig) -> list[Scorer]:
    """Build the `fluency` scorer(s), if enabled."""
    if not config.scorers.fluency.enabled:
        return []
    # A response that is not prose (e.g. a bare URL the user asked for) cannot be scored for fluency.
    applicable = _judge_applicable(
        "Does the agent response contain prose text (natural-language sentences) that can be "
        "assessed for grammar, fluency and naturalness? Answer no if the response is empty or "
        "consists only of non-prose content such as a URL, a number, or bare identifiers.",
        _default_judge_model(config),
    )
    return [
        wrap(
            Fluency(
                model=_judge_model(config, config.scorers.fluency),
                timeout=SCORER_TIMEOUT_SECONDS,
            ),
            applicable,
        )
    ]


def _build_guidelines_scorers(config: MlflowEvaluationConfig) -> list[Scorer]:
    """Build one `Guidelines` scorer per configured instance, if enabled."""
    guidelines_config = config.scorers.guidelines
    if not guidelines_config.enabled:
        return []
    if not guidelines_config.instances:
        raise ValueError("The `guidelines` scorer is enabled but has no instances.")
    return [
        wrap(
            Guidelines(
                name=instance.name,
                guidelines=instance.guidelines,
                model=_judge_model(config, guidelines_config),
                timeout=SCORER_TIMEOUT_SECONDS,
            ),
            _always_applicable,
        )
        for instance in guidelines_config.instances
    ]


def _build_relevance_to_query_scorers(config: MlflowEvaluationConfig) -> list[Scorer]:
    """Build the `relevance_to_query` scorer(s), if enabled."""
    if not config.scorers.relevance_to_query.enabled:
        return []
    return [
        wrap(
            RelevanceToQuery(
                model=_judge_model(config, config.scorers.relevance_to_query),
                timeout=SCORER_TIMEOUT_SECONDS,
            ),
            _always_applicable,
        )
    ]


def _build_source_document_groundedness_scorers(config: MlflowEvaluationConfig) -> list[Scorer]:
    """Build the `source_document_groundedness` scorer(s), if enabled.

    Judges the response against the record's `source_documents` expectation: every claim in
    the response must be supported by the provided documents.
    """
    scorer_config = config.scorers.source_document_groundedness
    if not scorer_config.enabled:
        return []
    question = (
        "Is the response consistent with the provided source documents, containing no factual "
        "claim that contradicts them? Answer no if any claim in the response contradicts a fact "
        "stated in the source documents."
    )

    def context(ctx: JudgeContext) -> dict[str, str]:
        return {
            "source_documents": "\n\n".join(get_field_list(ctx.expectations, "source_documents")),
            "response": ctx.response,
        }

    return [
        wrap(
            _custom_judge_scorer(
                name="source_document_groundedness",
                description="Scores whether the response contradicts the record's source documents.",
                question=question,
                context_builder=context,
                model=_judge_model(config, scorer_config),
            ),
            _any_expectation("source_documents"),
        )
    ]


def _build_retrieval_groundedness_scorers(config: MlflowEvaluationConfig) -> list[Scorer]:
    """Build the `retrieval_groundedness` scorer(s), if enabled."""
    if not config.scorers.retrieval_groundedness.enabled:
        return []
    return [
        wrap(
            RetrievalGroundedness(
                model=_judge_model(config, config.scorers.retrieval_groundedness),
                timeout=SCORER_TIMEOUT_SECONDS,
            ),
            _requires_retriever,
        )
    ]


def _build_retrieval_relevance_scorers(config: MlflowEvaluationConfig) -> list[Scorer]:
    """Build the `retrieval_relevance` scorer(s), if enabled."""
    if not config.scorers.retrieval_relevance.enabled:
        return []
    return [
        wrap(
            RetrievalRelevance(
                model=_judge_model(config, config.scorers.retrieval_relevance),
                inference_params=config.scorers.retrieval_relevance.inference_params,
                timeout=SCORER_TIMEOUT_SECONDS,
            ),
            _requires_retriever,
        )
    ]


def _build_retrieval_sufficiency_scorers(config: MlflowEvaluationConfig) -> list[Scorer]:
    """Build the `retrieval_sufficiency` scorer(s), if enabled."""
    if not config.scorers.retrieval_sufficiency.enabled:
        return []
    return [
        wrap(
            RetrievalSufficiency(
                model=_judge_model(config, config.scorers.retrieval_sufficiency),
                timeout=SCORER_TIMEOUT_SECONDS,
            ),
            _requires_retriever,
        )
    ]


def _build_safety_scorers(config: MlflowEvaluationConfig) -> list[Scorer]:
    """Build the `safety` scorer(s), if enabled."""
    if not config.scorers.safety.enabled:
        return []
    return [
        wrap(
            Safety(
                model=_judge_model(config, config.scorers.safety),
                timeout=SCORER_TIMEOUT_SECONDS,
            ),
            _always_applicable,
        )
    ]


def _build_summarization_scorers(config: MlflowEvaluationConfig) -> list[Scorer]:
    """Build the `summarization` scorer(s), if enabled."""
    if not config.scorers.summarization.enabled:
        return []
    # The built-in scorer scores the response as a summary of the request's source content, which
    # only applies when the record asks for a summary; whether the response is a good summary
    # is left to the scorer.
    applicable = _judge_applicable(
        "Does the user request ask the agent to summarize source content provided in the "
        "request (e.g. to summarize, condense or give the main points of the provided text)? "
        "Answer no if the request asks a question, a lookup, or any other non-summarization task.",
        _default_judge_model(config),
    )
    return [
        wrap(
            Summarization(
                model=_judge_model(config, config.scorers.summarization),
                timeout=SCORER_TIMEOUT_SECONDS,
            ),
            applicable,
        )
    ]


def _build_tool_call_correctness_scorers(config: MlflowEvaluationConfig) -> list[Scorer]:
    """Build the `tool_call_correctness` scorer(s), if enabled."""
    if not config.scorers.tool_call_correctness.enabled:
        return []
    return [
        wrap(
            ToolCallCorrectness(
                model=_judge_model(config, config.scorers.tool_call_correctness),
                should_exact_match=config.scorers.tool_call_correctness.should_exact_match,
                should_consider_ordering=config.scorers.tool_call_correctness.should_consider_ordering,
                timeout=SCORER_TIMEOUT_SECONDS,
            ),
            _always_applicable,
        )
    ]


def _build_tool_call_efficiency_scorers(config: MlflowEvaluationConfig) -> list[Scorer]:
    """Build the `tool_call_efficiency` scorer(s), if enabled."""
    if not config.scorers.tool_call_efficiency.enabled:
        return []
    return [
        wrap(
            ToolCallEfficiency(
                model=_judge_model(config, config.scorers.tool_call_efficiency),
                timeout=SCORER_TIMEOUT_SECONDS,
            ),
            _always_applicable,
        )
    ]


# The scorers builder per scorer type, referenced by name. Keep these in alphabetical order.
_SCORERS_BUILDERS: dict[ScorerType, ScorersBuilder] = {
    ScorerType.COMPLETENESS: _build_completeness_scorers,
    ScorerType.CORRECTNESS: _build_correctness_scorers,
    ScorerType.EQUIVALENCE: _build_equivalence_scorers,
    ScorerType.EXPECTATIONS_GUIDELINES: _build_expectations_guidelines_scorers,
    ScorerType.FLUENCY: _build_fluency_scorers,
    ScorerType.GUIDELINES: _build_guidelines_scorers,
    ScorerType.RELEVANCE_TO_QUERY: _build_relevance_to_query_scorers,
    ScorerType.RETRIEVAL_GROUNDEDNESS: _build_retrieval_groundedness_scorers,
    ScorerType.RETRIEVAL_RELEVANCE: _build_retrieval_relevance_scorers,
    ScorerType.RETRIEVAL_SUFFICIENCY: _build_retrieval_sufficiency_scorers,
    ScorerType.SAFETY: _build_safety_scorers,
    ScorerType.SOURCE_DOCUMENT_GROUNDEDNESS: _build_source_document_groundedness_scorers,
    ScorerType.SUMMARIZATION: _build_summarization_scorers,
    ScorerType.TOOL_CALL_CORRECTNESS: _build_tool_call_correctness_scorers,
    ScorerType.TOOL_CALL_EFFICIENCY: _build_tool_call_efficiency_scorers,
}


# Adding a scorer type without a builder (or vice versa) would silently break the enumeration
# in build_scorers_list.
if set(_SCORERS_BUILDERS) != set(ScorerType):
    raise ValueError("_SCORERS_BUILDERS must have exactly one builder per ScorerType.")


def build_scorers_list(config: MlflowEvaluationConfig) -> list[Scorer]:
    """Build the scorers enabled in the config; the judge goes through the AI Gateway."""
    scorers: list[Scorer] = []
    for scorer_type in ScorerType:
        scorers.extend(_SCORERS_BUILDERS[scorer_type](config))
    if not scorers:
        raise ValueError("No scorers are enabled in the evaluation config.")
    return scorers
