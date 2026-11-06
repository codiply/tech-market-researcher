"""MLflow evaluation runner."""

from langgraph.graph.state import CompiledStateGraph

from evaluations.mlflow.configuration import MlflowEvaluationConfig


def mlflow_evaluate(config: MlflowEvaluationConfig, agent: CompiledStateGraph) -> None:
    """Run the MLflow evaluation of the agent named by the config."""
    raise NotImplementedError
