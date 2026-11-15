"""Helpers for managing collections in our Chroma server."""

import os
from collections.abc import Sequence
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

EMBEDDINGS_MODEL_NAME_METADATA_KEY = "embeddings_model_name"

DEFAULT_QUERY_N_RESULTS = 10


class EmbeddingModelMismatchBehaviour(StrEnum):
    """What to do when an existing collection was created with a different embeddings model."""

    RAISE = "raise"
    RECOMPUTE = "recompute"


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


# Embeddings come directly from the OMLX server (OpenAI-compatible), configured in env;
# the AI Gateway does not proxy the /embeddings endpoint.
_OMLX_BASE_URL_ENV = "OMLX_BASE_URL"
_OMLX_API_KEY_ENV = "OMLX_API_KEY"


def _split_chroma_url() -> tuple[str, int]:
    """Split the CHROMA_URL env var into (host, port), defaulting to localhost:8385."""
    url = os.environ.get("CHROMA_URL", "http://localhost:8385")
    return urlsplit(url).hostname or "localhost", int(urlsplit(url).port or 8385)


async def get_client() -> AsyncClientAPI:
    """Create a client connected to the Chroma server configured in env."""
    host, port = _split_chroma_url()
    return await chromadb.AsyncHttpClient(host=host, port=port)


async def _get_collection(client: AsyncClientAPI, name: str) -> AsyncCollection | None:
    """Return the collection with the given name, or None if it does not exist."""
    try:
        return await client.get_collection(name=name)
    except NotFoundError:
        return None


async def get_or_create_collection(
    client: AsyncClientAPI, name: str, embeddings_model_name: str, mismatch_behaviour: EmbeddingModelMismatchBehaviour
) -> AsyncCollection:
    """Fetch a collection by name, creating it if it does not exist.

    New collections are created with an OpenAIEmbeddingFunction pointing at
    the OMLX server using the given embedding model, so documents get embedded
    on add. The model name is recorded in the collection metadata.
    """
    collection = await _get_collection(client, name)
    if collection is not None:
        existing_model = collection.metadata.get(EMBEDDINGS_MODEL_NAME_METADATA_KEY)
        if existing_model != embeddings_model_name:
            if mismatch_behaviour == EmbeddingModelMismatchBehaviour.RAISE:
                raise ValueError(
                    f"collection '{name}' was created with embeddings model {existing_model!r}, "
                    f"but {embeddings_model_name!r} was requested"
                )
            if mismatch_behaviour == EmbeddingModelMismatchBehaviour.RECOMPUTE:
                raise NotImplementedError
            raise ValueError(f"unknown mismatch behaviour: {mismatch_behaviour!r}")
        return collection
    # Chroma expects an embedding function over the wider CollectionInput type; we only ever pass documents.
    return await client.create_collection(
        name=name,
        embedding_function=cast(
            EmbeddingFunction[list[str] | list[NDArray[numpy.uint64 | numpy.int64 | numpy.float64]]],
            OpenAIEmbeddingFunction(
                api_key=os.environ[_OMLX_API_KEY_ENV],
                api_base=os.environ[_OMLX_BASE_URL_ENV].rstrip("/"),
                model_name=embeddings_model_name,
            ),
        ),
        metadata={EMBEDDINGS_MODEL_NAME_METADATA_KEY: embeddings_model_name},
    )


async def upsert_documents(collection: AsyncCollection, documents: Sequence[ChromaDocument]) -> None:
    """Insert records into a collection, embedding them with the collection's embedding function.

    Records whose ID already exist are replaced; new IDs are inserted.
    """
    metadatas: list[Metadata] = [document.metadata for document in documents]
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
