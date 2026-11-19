"""Custom errors of the Chroma integration; every chroma error in this package inherits from the base class."""

from tech_market_researcher.core.errors import BaseTechMarketResearcherError


class ChromaIntegrationError(BaseTechMarketResearcherError):
    """Base class for every custom error of the Chroma integration."""


class CollectionMigrationInProgressError(ChromaIntegrationError):
    """Raised when a write is attempted while the collection's embedding migration is in progress."""


class CollectionGoneError(ChromaIntegrationError):
    """Raised when a collection no longer exists on the server (e.g. it was removed during a migration)."""
