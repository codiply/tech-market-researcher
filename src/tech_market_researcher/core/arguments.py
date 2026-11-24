"""Command-line argument parsing."""

from abc import ABC, abstractmethod
from argparse import ArgumentParser
from typing import Self

from pydantic import Field, field_validator

from tech_market_researcher.core.collection_utils import find_duplicates
from tech_market_researcher.core.common import BasePydanticModel
from tech_market_researcher.core.constants import (
    DEFAULT_CONFIG_NAME,
    DEFAULT_TIME_ZONE,
    DEFAULT_USER_ID,
)


class BaseCommandLineArguments(ABC, BasePydanticModel):
    """Base model for command-line arguments; subclasses declare fields and register the arguments."""

    @classmethod
    @abstractmethod
    def add_arguments(cls: type[Self], parser: ArgumentParser) -> None:
        """Register the model's command-line arguments on parser."""


class ExtraMixinsCommandLineArguments(BaseCommandLineArguments):
    """Base for argument classes whose config can be augmented with extra mixins."""

    extra_mixins: list[str] = Field(default_factory=list)

    @field_validator("extra_mixins")
    @classmethod
    def _check_no_duplicate_extra_mixins(cls, value: list[str]) -> list[str]:
        duplicates = find_duplicates(value)
        if duplicates:
            raise ValueError(f"Extra mixins must not contain duplicates: {', '.join(duplicates)}.")
        return value

    @classmethod
    def add_arguments(cls: type[Self], parser: ArgumentParser) -> None:
        """Register the --extra-mixin argument on parser."""
        parser.add_argument(
            "--extra-mixin",
            "--x-mx",
            dest="extra_mixins",
            action="append",
            default=[],
            help="Mixin from the mixins folder applied on top of the loaded config; can be repeated, in order.",
        )


class AgentCommandLineArguments(ExtraMixinsCommandLineArguments):
    """Command-line arguments consumed by the agent."""

    config_name: str

    @classmethod
    def add_arguments(cls: type[AgentCommandLineArguments], parser: ArgumentParser) -> None:
        """Register the agent's command-line arguments on parser."""
        super().add_arguments(parser)
        parser.add_argument(
            "--config-name",
            default=DEFAULT_CONFIG_NAME,
            help="Name of the config to load; its file <name>.yaml will be read.",
        )


class CliCommandLineArguments(AgentCommandLineArguments):
    """Command-line arguments for the CLI, including agent-run identity."""

    user_id: str
    session_id: str | None = None
    time_zone: str

    @classmethod
    def add_arguments(cls: type[CliCommandLineArguments], parser: ArgumentParser) -> None:
        """Register the CLI command-line arguments on parser, including the agent's."""
        super().add_arguments(parser)
        parser.add_argument("--user-id", default=DEFAULT_USER_ID, help="User ID for the agent run.")
        # No default: the runtime context generates a random session id when omitted, so each run starts a new thread.
        parser.add_argument("--session-id", default=None, help="Session ID for the agent run.")
        parser.add_argument(
            "--time-zone",
            default=DEFAULT_TIME_ZONE,
            help="User IANA time zone, used by tools that report times (e.g. 'Europe/Madrid').",
        )


def _build_argument_parser[ArgsT: BaseCommandLineArguments](cls: type[ArgsT]) -> ArgumentParser:
    """Build an argument parser whose arguments are registered by cls."""
    parser = ArgumentParser()
    cls.add_arguments(parser)
    return parser


def get_args[ArgsT: BaseCommandLineArguments](cls: type[ArgsT]) -> ArgsT:
    """Parse the process command line into the given arguments model."""
    ns = _build_argument_parser(cls).parse_args()
    return cls.model_validate(vars(ns))
