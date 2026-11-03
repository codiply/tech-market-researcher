"""Shared exceptions for bootstrapping."""


class BootstrapError(RuntimeError):
    """Raised when a service is unreachable or in an inconsistent state."""
