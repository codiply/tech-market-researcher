"""Command-line argument parsing."""

from argparse import ArgumentParser

from pydantic import BaseModel

from tech_market_researcher.core.constants import DEFAULT_CONFIG_NAME


class CommandLineArguments(BaseModel):
    """Command-line arguments."""

    config_name: str


def _build_argument_parser() -> ArgumentParser:
    """Build the command-line argument parser."""
    parser = ArgumentParser()
    parser.add_argument(
        "--config-name",
        default=DEFAULT_CONFIG_NAME,
        help="Name of the config to load; its file <name>.yaml will be read.",
    )
    return parser


def get_args() -> CommandLineArguments:
    """Parse the process command line into CommandLineArguments."""
    ns = _build_argument_parser().parse_args()
    return CommandLineArguments(config_name=ns.config_name)
