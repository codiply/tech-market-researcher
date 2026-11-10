"""Builds the list of MLflow scorers from the evaluation config."""

from typing import Any, Callable

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

from evaluations.mlflow.configuration import BaseScorerConfig, MlflowEvaluationConfig

# Logged per record for scorers a dataset record cannot satisfy; MLflow excludes any
# non-yes/no string from the aggregated run metrics, so the mean covers applicable
# records only.
NOT_APPLICABLE = "not_applicable"

Applicability = Callable[..., bool]  # (expectations=..., trace=...) -> bool

# Per-invocation client-side timeout in seconds for all scorers, enforced by the MLflow client.
# 0 disables it (the gateway timeout is then the backstop).
SCORER_TIMEOUT_SECONDS = 0


def _has_expectation(expectations: dict[str, Any] | None, field: str) -> bool:
    return bool((expectations or {}).get(field))


def _any_expectation(*fields: str) -> Applicability:
    """Return an applicability predicate that is true when any of the given expectation fields is present."""

    def applicable(expectations=None, **_kwargs: Any) -> bool:
        return any(_has_expectation(expectations, field) for field in fields)

    return applicable


def _requires_retriever(trace=None, **_kwargs: Any) -> bool:
    """Applicable only when the trace contains at least one RETRIEVER span."""
    if trace is None:
        return False
    return any(s.attributes.get("mlflow.spanType") == "RETRIEVER" for s in trace.search_spans())


def wrap(inner: Scorer, applicable: Applicability) -> Scorer:
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
        if not applicable(expectations=expectations, trace=trace):
            return NOT_APPLICABLE
        return inner.run(
            inputs=inputs,
            outputs=outputs,
            expectations=expectations,
            trace=trace,
        )

    return conditional_scorer


def _judge_model(config: MlflowEvaluationConfig, scorer_config: BaseScorerConfig) -> str:
    """Return the judge model for a scorer, honoring any per-scorer override."""
    return f"gateway:/{scorer_config.model_name_override or config.judge_model_name}"


# The built-in scorers are defined in MLflow here:
# https://raw.githubusercontent.com/mlflow/mlflow/refs/heads/master/mlflow/genai/scorers/builtin_scorers.py
def build_scorers_list(config: MlflowEvaluationConfig) -> list[Scorer]:  # pylint: disable=too-many-branches
    """Build the scorers enabled in the config; the judge goes through the AI Gateway."""
    scorers: list[Scorer] = []
    # Keep these in alphabetical order.
    if config.scorers.completeness.enabled:
        scorers.append(
            Completeness(
                model=_judge_model(config, config.scorers.completeness),
                timeout=SCORER_TIMEOUT_SECONDS,
            )
        )
    if config.scorers.correctness.enabled:
        inner: Scorer = Correctness(
            model=_judge_model(config, config.scorers.correctness),
            timeout=SCORER_TIMEOUT_SECONDS,
        )
        scorers.append(wrap(inner, _any_expectation("expected_response", "expected_facts")))
    if config.scorers.equivalence.enabled:
        scorer_config = config.scorers.equivalence
        inner = Equivalence(
            model=_judge_model(config, scorer_config),
            inference_params=scorer_config.inference_params,
            timeout=SCORER_TIMEOUT_SECONDS,
        )
        scorers.append(wrap(inner, _any_expectation("expected_response")))
    if config.scorers.expectations_guidelines.enabled:
        inner = ExpectationsGuidelines(
            model=_judge_model(config, config.scorers.expectations_guidelines),
            timeout=SCORER_TIMEOUT_SECONDS,
        )
        scorers.append(wrap(inner, _any_expectation("guidelines")))
    if config.scorers.fluency.enabled:
        scorers.append(
            Fluency(
                model=_judge_model(config, config.scorers.fluency),
                timeout=SCORER_TIMEOUT_SECONDS,
            )
        )
    if config.scorers.guidelines.enabled:
        guidelines_config = config.scorers.guidelines
        if not guidelines_config.instances:
            raise ValueError("The `guidelines` scorer is enabled but has no instances.")
        for instance in guidelines_config.instances:
            scorers.append(
                Guidelines(
                    name=instance.name,
                    guidelines=instance.guidelines,
                    model=_judge_model(config, guidelines_config),
                    timeout=SCORER_TIMEOUT_SECONDS,
                )
            )
    if config.scorers.relevance_to_query.enabled:
        scorers.append(
            RelevanceToQuery(
                model=_judge_model(config, config.scorers.relevance_to_query),
                timeout=SCORER_TIMEOUT_SECONDS,
            )
        )
    if config.scorers.retrieval_groundedness.enabled:
        inner = RetrievalGroundedness(
            model=_judge_model(config, config.scorers.retrieval_groundedness),
            timeout=SCORER_TIMEOUT_SECONDS,
        )
        scorers.append(wrap(inner, _requires_retriever))
    if config.scorers.retrieval_relevance.enabled:
        scorer_config = config.scorers.retrieval_relevance
        inner = RetrievalRelevance(
            model=_judge_model(config, scorer_config),
            inference_params=scorer_config.inference_params,
            timeout=SCORER_TIMEOUT_SECONDS,
        )
        scorers.append(wrap(inner, _requires_retriever))
    if config.scorers.retrieval_sufficiency.enabled:
        inner = RetrievalSufficiency(
            model=_judge_model(config, config.scorers.retrieval_sufficiency),
            timeout=SCORER_TIMEOUT_SECONDS,
        )
        scorers.append(wrap(inner, _requires_retriever))
    if config.scorers.safety.enabled:
        scorers.append(
            Safety(
                model=_judge_model(config, config.scorers.safety),
                timeout=SCORER_TIMEOUT_SECONDS,
            )
        )
    if config.scorers.summarization.enabled:
        scorers.append(
            Summarization(
                model=_judge_model(config, config.scorers.summarization),
                timeout=SCORER_TIMEOUT_SECONDS,
            )
        )
    if config.scorers.tool_call_correctness.enabled:
        tool_call_correctness_config = config.scorers.tool_call_correctness
        scorers.append(
            ToolCallCorrectness(
                model=_judge_model(config, tool_call_correctness_config),
                should_exact_match=tool_call_correctness_config.should_exact_match,
                should_consider_ordering=tool_call_correctness_config.should_consider_ordering,
                timeout=SCORER_TIMEOUT_SECONDS,
            )
        )
    if config.scorers.tool_call_efficiency.enabled:
        scorers.append(
            ToolCallEfficiency(
                model=_judge_model(config, config.scorers.tool_call_efficiency),
                timeout=SCORER_TIMEOUT_SECONDS,
            )
        )
    if not scorers:
        raise ValueError("No scorers are enabled in the evaluation config.")
    return scorers
