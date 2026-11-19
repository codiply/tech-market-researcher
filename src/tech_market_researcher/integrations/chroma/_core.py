"""Package-private internals for the chroma package.

Do not import this module from outside the package. It holds the low-level primitives
(raw record writes, fetching and deleting collections) that the several other modules in
this package use.
"""

import uuid
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import cast

from chromadb import AsyncClientAPI
from chromadb.api import AsyncServerAPI
from chromadb.api.models.AsyncCollection import AsyncCollection
from chromadb.base_types import Metadata
from chromadb.errors import NotFoundError

from tech_market_researcher.integrations.chroma.common import (
    EMBEDDINGS_COMPUTED_AT_METADATA_KEY,
    EMBEDDINGS_MODEL_NAME_METADATA_KEY,
    ChromaDocument,
)


async def upsert_records(collection: AsyncCollection, documents: Sequence[ChromaDocument]) -> None:
    """Raw record write (embed + model stamp) with no migration check."""
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


async def get_collection_by_id(
    client: AsyncClientAPI | AsyncServerAPI,
    collection_id: uuid.UUID,
) -> AsyncCollection | None:
    """Fetch a collection by its UUID, or None if it does not exist.

    Prefers the targeted get_collection_by_id call (one round-trip, no matter how many
    collections exist). The targeted endpoint's error type for a missing id is not
    guaranteed across Chroma versions (e.g. against a server without the endpoint the
    client raises a bare Exception with an empty error body), so any other error falls
    back to scanning list_collections, whose NotFound semantics are known.
    """
    try:
        # Positional: the parameter is named `id` on the client API and `collection_id` on the
        # server API.
        return cast(AsyncCollection, await client.get_collection_by_id(collection_id))
    except NotFoundError:
        return None
    except Exception:  # pylint: disable=broad-exception-caught
        # Fall back to the scan below; it re-runs the same lookup on a known-good endpoint.
        pass
    for collection in await client.list_collections():
        if collection.id == collection_id:
            return cast(AsyncCollection, collection)
    return None


async def safe_delete(client: AsyncClientAPI, name: str) -> None:
    """Delete a collection by name, ignoring the case where it does not exist."""
    try:
        await client.delete_collection(name)
    except NotFoundError:
        pass
