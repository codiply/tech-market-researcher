"""Helpers for managing collections in our Chroma server."""

import logging
import os
import re
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from enum import StrEnum
from typing import cast
from urllib.parse import urlsplit

import chromadb
import numpy
from chromadb import AsyncClientAPI
from chromadb.api.models.AsyncCollection import AsyncCollection
from chromadb.api.types import EmbeddingFunction, QueryResult
from chromadb.base_types import Metadata
from chromadb.errors import NotFoundError
from chromadb.types import Where, WhereDocument
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from tech_market_researcher.core.configuration import ModelConnectionDetails

EMBEDDINGS_MODEL_NAME_METADATA_KEY = "_embeddings_model_name"

EMBEDDINGS_COMPUTED_AT_METADATA_KEY = "_embeddings_computed_at"

DEFAULT_QUERY_N_RESULTS = 10

# Chroma collection names only allow letters, digits, dots, underscores and dashes.
_COLLECTION_NAME_INVALID_CHARS = re.compile(r"[^a-zA-Z0-9._-]")

logger = logging.getLogger(__name__)


def collection_name_for_thread(base_name: str, thread_id: str) -> str:
    """Scope a collection name to a conversation thread.

    The thread id is suffixed, separated from the base name by the `-.-` separator and
    sanitized of the characters Chroma does not allow in collection names, so each
    conversation reads from and writes to its own collection.
    """
    return f"{base_name}-.-{_COLLECTION_NAME_INVALID_CHARS.sub('-', thread_id)}"


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


async def get_client(chroma_url_env_var: str = "CHROMA_URL") -> AsyncClientAPI:
    """Create a client connected to the Chroma server at the given env var's URL."""
    host, port = _split_chroma_url(chroma_url_env_var)
    logger.debug("Connecting to Chroma server at %s:%s", host, port)
    return await chromadb.AsyncHttpClient(host=host, port=port)


def _build_embedding_function(
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


async def _get_collection(
    client: AsyncClientAPI,
    name: str,
    embedding_function: EmbeddingFunction[list[str] | list[NDArray[numpy.uint64 | numpy.int64 | numpy.float64]]],
) -> AsyncCollection | None:
    """Return the collection with the given name, or None if it does not exist."""
    try:
        return await client.get_collection(name=name, embedding_function=embedding_function)
    except NotFoundError:
        return None


async def get_or_create_collection(
    client: AsyncClientAPI,
    name: str,
    embeddings_model: ModelConnectionDetails,
    mismatch_behavior: EmbeddingModelMismatchBehavior,
    additional_metadata: Mapping[str, str] | None = None,
) -> AsyncCollection:
    """Fetch a collection by name, creating it if it does not exist.

    New collections are created with an OpenAIEmbeddingFunction for the given embedding
    model, so documents get embedded on add. The model name is recorded in the
    collection metadata, together with any additional_metadata.
    """
    embedding_function = _build_embedding_function(embeddings_model)
    collection = await _get_collection(client, name, embedding_function)
    if collection is not None:
        existing_model = collection.metadata.get(EMBEDDINGS_MODEL_NAME_METADATA_KEY)
        if existing_model != embeddings_model.model_name:
            if mismatch_behavior == EmbeddingModelMismatchBehavior.RAISE:
                raise ValueError(
                    f"collection '{name}' was created with embeddings model {existing_model!r}, "
                    f"but {embeddings_model.model_name!r} was requested"
                )
            if mismatch_behavior == EmbeddingModelMismatchBehavior.MIGRATE:
                raise NotImplementedError
            raise ValueError(f"unknown mismatch behavior: {mismatch_behavior!r}")
        return collection
    # Chroma expects an embedding function over the wider CollectionInput type; we only ever pass documents.
    return await client.create_collection(
        name=name,
        embedding_function=embedding_function,
        metadata={EMBEDDINGS_MODEL_NAME_METADATA_KEY: embeddings_model.model_name, **(additional_metadata or {})},
    )


async def delete_collections_with_metadata(client: AsyncClientAPI, metadata: Mapping[str, str]) -> int:
    """Delete every collection whose metadata contains all the given key-value pairs.

    Collections whose metadata does not match every pair are left untouched.
    Returns the number of collections deleted.
    """
    deleted = 0
    for collection in await client.list_collections():
        if all((collection.metadata or {}).get(key) == value for key, value in metadata.items()):
            logger.info("Deleting collection '%s' (metadata matches %s)", collection.name, dict(metadata))
            await client.delete_collection(collection.name)
            deleted += 1
    return deleted


async def delete_where(collection: AsyncCollection, where: Where) -> None:
    """Delete all records matching the given where clause."""
    await collection.delete(where=where)


async def upsert_documents(collection: AsyncCollection, documents: Sequence[ChromaDocument]) -> None:
    """Insert records into a collection, embedding them with the collection's embedding function.

    Records whose ID already exist are replaced; new IDs are inserted.
    """
    # Record which model produced the embeddings and when, so staleness can be detected.
    model_name = collection.metadata.get(EMBEDDINGS_MODEL_NAME_METADATA_KEY)
    computed_at = datetime.now(timezone.utc).isoformat()
    metadatas: list[Metadata] = [
        {
            **document.metadata,
            EMBEDDINGS_MODEL_NAME_METADATA_KEY: model_name,
            EMBEDDINGS_COMPUTED_AT_METADATA_KEY: computed_at,
        }
        for document in documents
    ]
    await collection.upsert(
        ids=[document.id for document in documents],
        documents=[document.document for document in documents],
        metadatas=metadatas,
    )


def _zip_query_result(result: QueryResult) -> ChromaQueryResult:
    """Convert Chroma's column-major query result into zipped per-query items."""
    documents = result["documents"]
    metadatas = result["metadatas"]
    distances = result["distances"]
    items = [
        [
            QueryResultItem(
                id=record_id,
                document=documents[i][j] if documents else None,
                metadata=cast("dict[str, str | int | float | bool] | None", metadatas[i][j] if metadatas else None),
                distance=distances[i][j] if distances else None,
            )
            for j, record_id in enumerate(batch_ids)
        ]
        for i, batch_ids in enumerate(result["ids"])
    ]
    return ChromaQueryResult(items=items)


async def query(
    collection: AsyncCollection,
    query_texts: Sequence[str],
    n_results: int = DEFAULT_QUERY_N_RESULTS,
    where: Where | None = None,
    where_document: WhereDocument | None = None,
) -> ChromaQueryResult:
    """Run a similarity search over the collection, embedding query texts with the collection's embedding function.

    Returns zipped results: one list of items per input query, in input order.
    """
    result = await collection.query(
        query_texts=list(query_texts), n_results=n_results, where=where, where_document=where_document
    )
    return _zip_query_result(result)
