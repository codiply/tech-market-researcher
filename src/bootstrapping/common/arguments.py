"""Command-line argument parsing for the bootstrap."""

from tech_market_researcher.core.arguments import ExtraMixinsCommandLineArguments, get_args


class BootstrapCommandLineArguments(ExtraMixinsCommandLineArguments):
    """Command-line arguments consumed by the bootstrap."""


__all__ = ["BootstrapCommandLineArguments", "get_args"]
