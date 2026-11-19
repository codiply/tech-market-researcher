"""Creating, resolving and writing Chroma collections for the agent."""

import logging
import re
from collections.abc import Mapping, Sequence
from typing import cast

import numpy
from chromadb import AsyncClientAPI
from chromadb.api.models.AsyncCollection import AsyncCollection
from chromadb.api.types import EmbeddingFunction, QueryResult
from chromadb.errors import NotFoundError
from chromadb.types import Where, WhereDocument
from numpy.typing import NDArray

from tech_market_researcher.core.configuration import ModelConnectionDetails
from tech_market_researcher.integrations.chroma._core import upsert_records
from tech_market_researcher.integrations.chroma.common import (
    EMBEDDINGS_MODEL_NAME_METADATA_KEY,
    ChromaDocument,
    ChromaQueryResult,
    EmbeddingModelMismatchBehavior,
    QueryResultItem,
    build_embedding_function,
)
from tech_market_researcher.integrations.chroma.migration import assert_not_migrating, migrate_if_needed

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


async def _get_collection(
    client: AsyncClientAPI,
    name: str,
    embedding_function: EmbeddingFunction[list[str] | list[NDArray[numpy.uint64 | numpy.int64 | numpy.float64]]] | None,
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

    With MIGRATE behavior, an existing collection with a different embeddings model is
    re-embedded with the new model and swapped in under the same name; the same is done
    when the collection's status reports a repair pending, so a crashed migration is
    resumable.
    """
    embedding_function = build_embedding_function(embeddings_model)
    collection = await _get_collection(client, name, embedding_function)
    if collection is not None:
        return await migrate_if_needed(
            client, collection, name, embeddings_model, mismatch_behavior, additional_metadata
        )
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
    """Delete all records matching the given where clause.

    Refuses to delete while an embedding migration of the collection is in progress, like
    upsert_documents, so a delete is not applied to a collection that is about to be swapped out.
    """
    await assert_not_migrating(collection)
    await collection.delete(where=where)


async def upsert_documents(collection: AsyncCollection, documents: Sequence[ChromaDocument]) -> None:
    """Insert records into a collection, embedding them with the collection's embedding function.

    Records whose ID already exist are replaced; new IDs are inserted. Refuses to write while an
    embedding migration of the collection is in progress (checked against the live metadata,
    right before the write). The check is best-effort: without a server-side lock it is not
    atomic with the write, so an in-flight write can still land in the collection moments after
    the check.
    """
    await assert_not_migrating(collection)
    await upsert_records(collection, documents)


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
