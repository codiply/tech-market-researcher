"""Command-line argument parsing for evaluations."""

from argparse import ArgumentParser
from typing import Self

from pydantic import Field, ValidationInfo, field_validator

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
from tech_market_researcher.core.collection_utils import find_duplicates


class EvalsCommandLineArguments(BaseCommandLineArguments):
    """Command-line arguments for running evaluations of the agent."""

    agent_config_name: str
    evals_config_name: str
    platform: EvalsPlatform
    extra_mixins_evals: list[str] = Field(default_factory=list)
    extra_mixins_agent: list[str] = Field(default_factory=list)

    @field_validator("extra_mixins_evals", "extra_mixins_agent")
    @classmethod
    def _check_no_duplicate_extra_mixins(cls, value: list[str], info: ValidationInfo) -> list[str]:
        duplicates = find_duplicates(value)
        if duplicates:
            raise ValueError(f"{info.field_name} must not contain duplicates: {', '.join(duplicates)}.")
        return value

    @classmethod
    def add_arguments(cls: type[Self], parser: ArgumentParser) -> None:
        """Register the evaluations' command-line arguments on parser."""
        parser.add_argument(
            "--extra-mixins-evals",
            "--x-mx-ev",
            action="append",
            default=[],
            help=(
                "Mixin from the evals config mixins folder applied on top of the evals config; "
                "can be repeated, in order."
            ),
        )
        parser.add_argument(
            "--extra-mixins-agent",
            "--x-mx-ag",
            action="append",
            default=[],
            help=(
                "Mixin from the agent config mixins folder applied on top of the agent config; "
                "can be repeated, in order."
            ),
        )
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
        return AgentCommandLineArguments(
            config_name=self.agent_config_name,
            extra_mixins=self.extra_mixins_agent,
        )


__all__ = ["EvalsCommandLineArguments", "get_args"]
