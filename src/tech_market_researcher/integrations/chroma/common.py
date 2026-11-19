"""Common building blocks for the Chroma integration: client access, records and embedding models."""

import logging
import os
from enum import StrEnum
from typing import cast
from urllib.parse import urlsplit

import chromadb
import numpy
from chromadb import AsyncClientAPI
from chromadb.api.types import EmbeddingFunction
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from tech_market_researcher.core.configuration import ModelConnectionDetails

EMBEDDINGS_MODEL_NAME_METADATA_KEY = "_embeddings_model_name"

EMBEDDINGS_COMPUTED_AT_METADATA_KEY = "_embeddings_computed_at"

logger = logging.getLogger(__name__)


class EmbeddingModelMismatchBehavior(StrEnum):
    """What to do when an existing collection was created with a different embeddings model."""

    RAISE = "raise"
    MIGRATE = "migrate"


class ChromaDocument(BaseModel):
    """A record to store in a Chroma collection."""

    id: str
    document: str
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)


class QueryResultItem(BaseModel):
    """A single match returned by a collection query."""

    id: str
    document: str | None
    metadata: dict[str, str | int | float | bool] | None
    distance: float | None


class ChromaQueryResult(BaseModel):
    """Zipped query results: one list of items per input query, in input order."""

    items: list[list[QueryResultItem]]


def _split_chroma_url(chroma_url_env_var: str) -> tuple[str, int]:
    """Split the Chroma URL env var into (host, port), defaulting to localhost:8386."""
    url = os.environ.get(chroma_url_env_var, "http://localhost:8386")
    return urlsplit(url).hostname or "localhost", int(urlsplit(url).port or 8386)


async def get_client(chroma_url_env_var: str) -> AsyncClientAPI:
    """Create a client connected to the Chroma server at the given env var's URL."""
    host, port = _split_chroma_url(chroma_url_env_var)
    logger.debug("Connecting to Chroma server at %s:%s", host, port)
    return await chromadb.AsyncHttpClient(host=host, port=port)


def build_embedding_function(
    embeddings_model: ModelConnectionDetails,
) -> EmbeddingFunction[list[str] | list[NDArray[numpy.uint64 | numpy.int64 | numpy.float64]]]:
    """Build an OpenAIEmbeddingFunction backed by the resolved embedding model."""
    # Chroma expects an embedding function over the wider CollectionInput type; we only ever pass documents.
    return cast(
        EmbeddingFunction[list[str] | list[NDArray[numpy.uint64 | numpy.int64 | numpy.float64]]],
        OpenAIEmbeddingFunction(
            api_key=embeddings_model.api_key.get_secret_value(),
            api_base=embeddings_model.base_url.rstrip("/"),
            model_name=embeddings_model.model_name,
        ),
    )
