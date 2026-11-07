"""Builds the list of MLflow scorers from the evaluation config."""

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
)

from evaluations.mlflow.configuration import BaseScorerConfig, MlflowEvaluationConfig


def _judge_model(config: MlflowEvaluationConfig, scorer_config: BaseScorerConfig) -> str:
    """Return the judge model for a scorer, honoring any per-scorer override."""
    return f"gateway:/{scorer_config.model_name_override or config.judge_model_name}"


def build_scorers_list(config: MlflowEvaluationConfig) -> list[Scorer]:  # pylint: disable=too-many-branches
    """Build the scorers enabled in the config; the judge goes through the AI Gateway."""
    scorers: list[Scorer] = []
    if config.scorers.completeness.enabled:
        scorers.append(Completeness(model=_judge_model(config, config.scorers.completeness)))
    if config.scorers.correctness.enabled:
        scorers.append(Correctness(model=_judge_model(config, config.scorers.correctness)))
    if config.scorers.equivalence.enabled:
        scorers.append(Equivalence(model=_judge_model(config, config.scorers.equivalence)))
    if config.scorers.expectations_guidelines.enabled:
        scorers.append(ExpectationsGuidelines(model=_judge_model(config, config.scorers.expectations_guidelines)))
    if config.scorers.fluency.enabled:
        scorers.append(Fluency(model=_judge_model(config, config.scorers.fluency)))
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
                )
            )
    if config.scorers.relevance_to_query.enabled:
        scorers.append(RelevanceToQuery(model=_judge_model(config, config.scorers.relevance_to_query)))
    if config.scorers.retrieval_groundedness.enabled:
        scorers.append(RetrievalGroundedness(model=_judge_model(config, config.scorers.retrieval_groundedness)))
    if config.scorers.retrieval_relevance.enabled:
        scorers.append(RetrievalRelevance(model=_judge_model(config, config.scorers.retrieval_relevance)))
    if config.scorers.retrieval_sufficiency.enabled:
        scorers.append(RetrievalSufficiency(model=_judge_model(config, config.scorers.retrieval_sufficiency)))
    if config.scorers.safety.enabled:
        scorers.append(Safety(model=_judge_model(config, config.scorers.safety)))
    if config.scorers.summarization.enabled:
        scorers.append(Summarization(model=_judge_model(config, config.scorers.summarization)))
    if config.scorers.tool_call_correctness.enabled:
        scorers.append(ToolCallCorrectness(model=_judge_model(config, config.scorers.tool_call_correctness)))
    if config.scorers.tool_call_efficiency.enabled:
        scorers.append(ToolCallEfficiency(model=_judge_model(config, config.scorers.tool_call_efficiency)))
    if not scorers:
        raise ValueError("No scorers are enabled in the evaluation config.")
    return scorers
