"""Config models. All classes here are named XxxConfig."""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

CONFIG_DIR = Path("config") / "evaluations" / "mlflow"


class BaseScorerConfig(BaseModel):
    """Settings for a single scorer."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    # Override the judge model for this scorer; defaults to the global judge model.
    model_name_override: str | None = None


class GuidelinesInstanceConfig(BaseModel):
    """A single instance of the `guidelines` scorer."""

    model_config = ConfigDict(extra="forbid")

    name: str
    guidelines: list[str]


class GuidelinesScorerConfig(BaseScorerConfig):
    """Settings for the `guidelines` scorer, which applies global guidelines."""

    instances: list[GuidelinesInstanceConfig] = []


class ScorersConfig(BaseModel):
    """Which scorers to run, keyed by scorer name."""

    model_config = ConfigDict(extra="forbid")

    completeness: BaseScorerConfig
    correctness: BaseScorerConfig
    equivalence: BaseScorerConfig
    expectations_guidelines: BaseScorerConfig
    fluency: BaseScorerConfig
    guidelines: GuidelinesScorerConfig
    relevance_to_query: BaseScorerConfig
    retrieval_groundedness: BaseScorerConfig
    retrieval_relevance: BaseScorerConfig
    retrieval_sufficiency: BaseScorerConfig
    safety: BaseScorerConfig
    summarization: BaseScorerConfig
    tool_call_correctness: BaseScorerConfig
    tool_call_efficiency: BaseScorerConfig


class MlflowEvaluationConfig(BaseModel):
    """Evaluation settings. The judge runs against an endpoint of the MLflow AI Gateway."""

    model_config = ConfigDict(extra="forbid")

    experiment_name: str
    dataset_name: str
    judge_model_name: str
    scorers: ScorersConfig


def load_config(name: str) -> MlflowEvaluationConfig:
    """Load the evaluation config at config/evaluations/mlflow/<name>.yaml."""
    data = yaml.safe_load((CONFIG_DIR / f"{name}.yaml").read_text())
    return MlflowEvaluationConfig.model_validate(data)
