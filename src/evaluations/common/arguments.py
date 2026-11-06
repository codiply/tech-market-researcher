"""Command-line argument parsing for evaluations."""

from argparse import ArgumentParser
from typing import Self

from evaluations.common.constants import (
    DEFAULT_AGENT_CONFIG_NAME,
    DEFAULT_EVALS_CONFIG_NAME,
    DEFAULT_EVALS_PLATFORM,
    EvalsPlatform,
)
from tech_market_researcher.core.arguments import (
    AgentCommandLineArguments,
    BaseCommandLineArguments,
    get_args,
)


class EvalsCommandLineArguments(BaseCommandLineArguments):
    """Command-line arguments for running evaluations of the agent."""

    agent_config_name: str
    evals_config_name: str
    platform: EvalsPlatform

    @classmethod
    def add_arguments(cls: type[Self], parser: ArgumentParser) -> None:
        """Register the evaluations' command-line arguments on parser."""
        parser.add_argument(
            "--agent-config-name",
            default=DEFAULT_AGENT_CONFIG_NAME,
            help="Name of the agent config to load; its file <name>.yaml will be read.",
        )
        parser.add_argument(
            "--evals-config-name",
            default=DEFAULT_EVALS_CONFIG_NAME,
            help="Name of the evaluations config to load; its file <name>.yaml will be read.",
        )
        parser.add_argument(
            "--platform",
            choices=[p.value for p in EvalsPlatform],
            default=DEFAULT_EVALS_PLATFORM.value,
            help="Evaluation platform to use.",
        )

    def as_agent_args(self) -> AgentCommandLineArguments:
        """Map these arguments to the researcher's agent command-line arguments."""
        return AgentCommandLineArguments(config_name=self.agent_config_name)


__all__ = ["EvalsCommandLineArguments", "get_args"]
