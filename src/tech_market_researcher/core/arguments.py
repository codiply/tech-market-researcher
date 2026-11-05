"""Command-line argument parsing."""

from argparse import ArgumentParser

from pydantic import BaseModel

from tech_market_researcher.core.constants import (
    DEFAULT_CONFIG_NAME,
    DEFAULT_SESSION_ID,
    DEFAULT_TIME_ZONE,
    DEFAULT_USER_ID,
)


class AgentCommandLineArguments(BaseModel):
    """Command-line arguments consumed by the agent."""

    config_name: str

    @classmethod
    def add_arguments(cls: type[AgentCommandLineArguments], parser: ArgumentParser) -> None:
        """Register the agent's command-line arguments on parser."""
        parser.add_argument(
            "--config-name",
            default=DEFAULT_CONFIG_NAME,
            help="Name of the config to load; its file <name>.yaml will be read.",
        )


class CliCommandLineArguments(AgentCommandLineArguments):
    """Command-line arguments for the CLI, including agent-run identity."""

    user_id: str
    session_id: str
    time_zone: str

    @classmethod
    def add_arguments(cls: type[CliCommandLineArguments], parser: ArgumentParser) -> None:
        """Register the CLI command-line arguments on parser, including the agent's."""
        super().add_arguments(parser)
        parser.add_argument("--user-id", default=DEFAULT_USER_ID, help="User ID for the agent run.")
        parser.add_argument("--session-id", default=DEFAULT_SESSION_ID, help="Session ID for the agent run.")
        parser.add_argument(
            "--time-zone",
            default=DEFAULT_TIME_ZONE,
            help="User IANA time zone, used by tools that report times (e.g. 'Europe/Madrid').",
        )


def _build_argument_parser[ArgsT: AgentCommandLineArguments](cls: type[ArgsT]) -> ArgumentParser:
    """Build an argument parser whose arguments are registered by cls."""
    parser = ArgumentParser()
    cls.add_arguments(parser)
    return parser


def get_args[ArgsT: AgentCommandLineArguments](cls: type[ArgsT]) -> ArgsT:
    """Parse the process command line into the given arguments model."""
    ns = _build_argument_parser(cls).parse_args()
    return cls.model_validate(vars(ns))
