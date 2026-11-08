"""Config models. All classes here are named XxxConfig."""

from pathlib import Path

from tech_market_researcher.core.common import BasePydanticModel
from tech_market_researcher.core.config_loader import ConfigLoader, ConfigLoaderParams
from tech_market_researcher.core.constants import BASE_DEBUG_DIR_PATH

CONFIG_DIR = Path("config") / "evaluations" / "mlflow"


class BaseScorerConfig(BasePydanticModel):
    """Settings for a single scorer."""

    enabled: bool = False
    # Override the judge model for this scorer; defaults to the global judge model.
    model_name_override: str | None = None


class GuidelinesInstanceConfig(BasePydanticModel):
    """A single instance of the `guidelines` scorer."""

    name: str
    guidelines: list[str]


class GuidelinesScorerConfig(BaseScorerConfig):
    """Settings for the `guidelines` scorer, which applies global guidelines."""

    instances: list[GuidelinesInstanceConfig] = []


class ScorersConfig(BasePydanticModel):
    """Which scorers to run, keyed by scorer name."""

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


class MlflowEvaluationConfig(BasePydanticModel):
    """Evaluation settings. The judge runs against an endpoint of the MLflow AI Gateway."""

    experiment_name: str
    dataset_name: str
    judge_model_name: str
    scorers: ScorersConfig


CONFIG_LOADER = ConfigLoader(
    MlflowEvaluationConfig,
    CONFIG_DIR,
    ConfigLoaderParams(debug_dir_path=BASE_DEBUG_DIR_PATH / "config" / "evaluations" / "mlflow"),
)


def load_config(name: str) -> MlflowEvaluationConfig:
    """Load the evaluation config at config/evaluations/mlflow/<name>.yaml."""
    return CONFIG_LOADER.load(name)
