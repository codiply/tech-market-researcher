"""Embeddings model migration for Chroma collections.

Chroma has no in-place re-embedding, so a migration re-embeds the collection's records into a
temporary collection and swaps it in under the same name. Chroma also has no locking primitive,
so the migration coordinates through collection metadata (an in-progress marker on the live
collection, a parked old collection), and every crash leaves the system in a state the next run
can resume from. See the "Embeddings model migration" section of documentation/design.md.

Imported by the collection module but not the other way around, so the package's dependency
graph stays acyclic; shared building blocks live in the common module.
"""

import logging
import uuid
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from enum import StrEnum
from typing import cast

from chromadb import AsyncClientAPI
from chromadb.api import AsyncServerAPI
from chromadb.api.models.AsyncCollection import AsyncCollection
from chromadb.errors import NotFoundError

from tech_market_researcher.core.configuration import ModelConnectionDetails
from tech_market_researcher.integrations.chroma._core import get_collection_by_id, safe_delete, upsert_records
from tech_market_researcher.integrations.chroma.common import (
    EMBEDDINGS_MODEL_NAME_METADATA_KEY,
    ChromaDocument,
    EmbeddingModelMismatchBehavior,
    build_embedding_function,
)
from tech_market_researcher.integrations.chroma.errors import CollectionGoneError, CollectionMigrationInProgressError

logger = logging.getLogger(__name__)

MIGRATION_IN_PROGRESS_METADATA_KEY = "_migrating"
MIGRATION_STARTED_AT_METADATA_KEY = "_migrating_started_at"
MIGRATION_TEMP_FOR_METADATA_KEY = "_migration_temp_for"
MIGRATION_SOURCE_NAME_SUFFIX = "__migrating-source"
MIGRATION_TEMP_NAME_PREFIX = "__migrating-"

# Collection metadata keys written by the migration itself (never carried over to the final collection).
_MIGRATION_METADATA_KEYS = (
    MIGRATION_IN_PROGRESS_METADATA_KEY,
    MIGRATION_STARTED_AT_METADATA_KEY,
    MIGRATION_TEMP_FOR_METADATA_KEY,
)

# A migration marker older than this is considered stale (crashed migration) and is cleared.
MIGRATION_LEASE_PERIOD = timedelta(hours=24)
MIGRATION_COPY_BATCH_SIZE = 512


class CollectionStatus(StrEnum):
    """A collection's status with respect to embeddings model migration."""

    # The collection needs no action.
    AVAILABLE = "available"
    # A migration is live; block writes and duplicate migrations.
    MIGRATING = "migrating"
    # A crashed migration must be resumed (or reported to the caller).
    PENDING_REPAIR = "pending_repair"
    # The live collection no longer exists on the server.
    ABSENT = "absent"


def _migration_source_name(name: str) -> str:
    """Name the old collection is parked under while its embedding migration runs."""
    return f"{name}{MIGRATION_SOURCE_NAME_SUFFIX}"


def is_migration_stale(started_at: str | None) -> bool:
    """Whether a migration started at the given timestamp is older than the lease, or missing/unparseable."""
    if started_at is None:
        return True
    try:
        started = datetime.fromisoformat(started_at)
    except ValueError:
        return True
    if started.tzinfo is None:
        started = started.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) - started > MIGRATION_LEASE_PERIOD


async def get_parked_source(client: AsyncClientAPI | AsyncServerAPI, name: str) -> AsyncCollection | None:
    """Return the old collection parked by a crashed migration of `name`, if any."""
    try:
        return cast(AsyncCollection, await client.get_collection(name=_migration_source_name(name)))
    except NotFoundError:
        return None


async def get_collection_status(
    client: AsyncClientAPI | AsyncServerAPI,
    collection: AsyncCollection | None,
    name: str,
) -> CollectionStatus:
    """Report the collection's migration status, clearing the stale marker of a crashed migration on the way."""
    # Re-fetch the live collection by id, since the object's metadata is cached by Chroma.
    live = await get_collection_by_id(client, collection.id) if collection is not None else None
    if live is None:
        # No live collection: a parked leftover means a crashed migration must be repaired, otherwise it is gone.
        return CollectionStatus.PENDING_REPAIR if await get_parked_source(client, name) else CollectionStatus.ABSENT
    metadata = live.metadata or {}
    if metadata.get(MIGRATION_IN_PROGRESS_METADATA_KEY):
        started_at = metadata.get(MIGRATION_STARTED_AT_METADATA_KEY)
        if not is_migration_stale(started_at):
            return CollectionStatus.MIGRATING
        # A stale marker is from a crashed migration; clear it (logged) so it never reports as in-progress.
        logger.warning("Clearing stale migration marker on collection '%s' (started %s)", live.name, started_at)
        await _remove_migration_metadata(live)
    parked = await get_parked_source(client, name)
    if parked is None:
        return CollectionStatus.AVAILABLE
    # A fresh timestamp on the parked collection means a migration is currently running against it.
    if not is_migration_stale((parked.metadata or {}).get(MIGRATION_STARTED_AT_METADATA_KEY)):
        return CollectionStatus.MIGRATING
    return CollectionStatus.PENDING_REPAIR


async def assert_not_migrating(collection: AsyncCollection) -> None:
    """Refuse writes while the collection's embedding migration is in progress.

    Decides off get_collection_status, which heals a stale marker of a crashed migration
    first, so only a live migration blocks; other statuses proceed.
    """
    # Private attribute; the client is needed to resolve the status.
    client = collection._client  # pylint: disable=protected-access
    if await get_collection_status(client, collection, collection.name) is CollectionStatus.MIGRATING:
        raise CollectionMigrationInProgressError(
            f"collection '{collection.name}' is in the middle of an embedding model migration; retry once it finishes"
        )


async def _clear_migration_marker(client: AsyncClientAPI, collection: AsyncCollection) -> None:
    """Remove the migration marker keys from the collection, if it still exists."""
    await _remove_migration_metadata(await get_collection_by_id(client, collection.id))


async def _remove_migration_metadata(collection: AsyncCollection | None) -> None:
    """Remove the migration bookkeeping keys from the collection's metadata, if present.

    modify() replaces the whole metadata, so the full dict minus the bookkeeping keys is sent;
    the keys are dropped rather than set to false, so no marker residue is left behind.
    """
    if collection is None:
        return
    metadata = collection.metadata or {}
    stripped = {k: v for k, v in metadata.items() if k not in _MIGRATION_METADATA_KEYS}
    if stripped != metadata:
        await collection.modify(metadata=stripped)


async def _clear_parked_marker(client: AsyncClientAPI, name: str) -> None:
    """Remove the migration marker keys from the parked source of `name`, if present.

    Unlike _clear_migration_marker, the started timestamp is removed entirely: other runs use
    the parked collection's timestamp freshness to detect an in-progress migration, so a fresh
    leftover timestamp would wedge the collection until it expires.
    """
    await _remove_migration_metadata(await get_parked_source(client, name))


async def _cleanup_stale_temp_collections(client: AsyncClientAPI, source_id: uuid.UUID) -> None:
    """Delete orphan temp collections left by a crashed migration of the given collection."""
    for candidate in await client.list_collections():
        metadata = candidate.metadata or {}
        if metadata.get(MIGRATION_TEMP_FOR_METADATA_KEY) == str(source_id) and is_migration_stale(
            metadata.get(MIGRATION_STARTED_AT_METADATA_KEY)
        ):
            logger.info("Deleting orphaned migration temp collection '%s'", candidate.name)
            await client.delete_collection(candidate.name)


async def _reembed_into(source: AsyncCollection, temp: AsyncCollection) -> int:
    """Re-embed every record of `source` into `temp` (upsert, idempotent). Returns the record count copied.

    Records without a document are re-embedded as empty documents, and metadata values that are
    not plain scalars (lists, sparse vectors) are dropped; both are logged, as they change what
    the record contains in `temp`.
    """
    ids = list((await source.get(include=[]))["ids"])
    records_without_document = 0
    dropped_metadata_keys = 0
    for start in range(0, len(ids), MIGRATION_COPY_BATCH_SIZE):
        chunk = ids[start : start + MIGRATION_COPY_BATCH_SIZE]
        records = await source.get(ids=chunk)
        documents: list[ChromaDocument] = []
        for record_id, document, metadata in zip(
            records["ids"] or [], records["documents"] or [], records["metadatas"] or []
        ):
            if document is None:
                logger.warning(
                    "Collection '%s': record %s has no document and is re-embedded as an empty document",
                    source.name,
                    record_id,
                )
                records_without_document += 1
            # Keep only plain scalar values; the rest (lists, sparse vectors) is not a supported
            # metadata value and would fail the upsert anyway.
            kept = {k: v for k, v in (metadata or {}).items() if isinstance(v, (str, int, float, bool))}
            dropped_metadata_keys += len(metadata or {}) - len(kept)
            documents.append(
                ChromaDocument(
                    id=record_id,
                    document=document or "",
                    metadata=kept,
                )
            )
        await upsert_records(temp, documents)
    if records_without_document:
        logger.warning(
            "Collection '%s': %d record(s) have no document and are re-embedded as empty documents",
            source.name,
            records_without_document,
        )
    if dropped_metadata_keys:
        logger.warning(
            "Collection '%s': %d metadata value(s) were not plain scalars and are dropped from the migrated records",
            source.name,
            dropped_metadata_keys,
        )
    return len(ids)


async def _copy_records(source: AsyncCollection, temp: AsyncCollection, name: str) -> None:
    """Re-embed the source into the temp collection, retrying once if the source changed mid-copy.

    Raises if the source keeps changing, or if the temp collection's record count does not match
    the copy after the retry (e.g. a concurrent migration writing into the same temp).
    """
    before = await source.count()
    copied = await _reembed_into(source, temp)
    if copied != before or await temp.count() != copied:
        logger.warning("Collection '%s' changed during the migration copy; re-copying", name)
        copied = await _reembed_into(source, temp)
        if copied != before or await source.count() != before or await temp.count() != copied:
            raise RuntimeError(f"collection '{name}' kept changing during migration; retry the migration")


async def _strip_migration_metadata(client: AsyncClientAPI, name: str) -> None:
    """Remove the migration bookkeeping keys from the migrated collection's metadata."""
    try:
        migrated = await client.get_collection(name=name)
    except NotFoundError:
        return
    await _remove_migration_metadata(migrated)


async def _merge_and_remove_name(client: AsyncClientAPI, name: str, temp: AsyncCollection) -> None:
    """Re-embed the collection currently at `name` into `temp`, then delete it.

    Takes over a name whose current holder is a collection whose records are being absorbed by
    the migration: either a collection recreated under the name after a crashed migration, or a
    collection created in the window between the old collection being parked and the temp
    collection being renamed into place. A no-op when no collection holds the name.
    """
    try:
        holder = await client.get_collection(name=name)
    except NotFoundError:
        return
    merged = await _reembed_into(holder, temp)
    if merged:
        logger.info("Re-embedded %d records written to collection '%s' while the migration was running", merged, name)
    await client.delete_collection(name)


async def _swap_collections(
    client: AsyncClientAPI,
    live: AsyncCollection,
    name: str,
    temp: AsyncCollection,
    parked: AsyncCollection | None,
) -> None:
    """Swap the temp collection into the target name, then delete the parked old collection."""
    if parked is None:
        # Re-validate just before the swap: if a concurrent migration renamed or removed the
        # live collection, abort before touching the name.
        current = await get_collection_by_id(client, live.id)
        if current is None or current.name != name:
            raise RuntimeError(f"collection '{name}' was modified by a concurrent migration; aborting")
        # Park the old collection (instead of deleting it) so a crash mid-swap loses nothing.
        await current.modify(name=_migration_source_name(name))
        # The name is free between parking and the rename below; a concurrent
        # get_or_create_collection may have created a fresh collection there. Absorb its records
        # instead of colliding on the rename; if one is created after this check, the rename may
        # fail, and the next run resumes from the parked collection and merges that collection's
        # records, so nothing is lost.
        await _merge_and_remove_name(client, name, temp)
        await temp.modify(name=name)
        await client.delete_collection(_migration_source_name(name))
    else:
        # Resume: records may have been written to a fresh collection under the target name since
        # the crashed migration; re-embed them into the temp before taking the name over.
        await _merge_and_remove_name(client, name, temp)
        await temp.modify(name=name)
        await client.delete_collection(_migration_source_name(name))


async def migrate_if_needed(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    client: AsyncClientAPI,
    collection: AsyncCollection,
    name: str,
    embeddings_model: ModelConnectionDetails,
    mismatch_behavior: EmbeddingModelMismatchBehavior,
    additional_metadata: Mapping[str, str] | None,
) -> AsyncCollection:
    """Return the collection, running a migration first if the embeddings model mismatches or a repair is pending."""
    status = await get_collection_status(client, collection, name)
    if status is CollectionStatus.ABSENT:
        raise CollectionGoneError(f"collection '{name}' no longer exists on the server")
    if status is CollectionStatus.MIGRATING:
        raise CollectionMigrationInProgressError(
            f"a migration of collection '{name}' is in progress; retry once it finishes"
        )
    existing_model = collection.metadata.get(EMBEDDINGS_MODEL_NAME_METADATA_KEY)
    if existing_model != embeddings_model.model_name or status is CollectionStatus.PENDING_REPAIR:
        if mismatch_behavior is not EmbeddingModelMismatchBehavior.MIGRATE:
            if existing_model != embeddings_model.model_name:
                raise ValueError(
                    f"collection '{name}' was created with embeddings model {existing_model!r}, "
                    f"but {embeddings_model.model_name!r} was requested"
                )
            raise ValueError(
                f"collection '{name}' has leftovers from a crashed embedding migration; "
                "re-run with the migrate behavior to repair it"
            )
        return await migrate_collection(client, collection, name, embeddings_model, additional_metadata)
    return collection


async def migrate_collection(
    client: AsyncClientAPI,
    collection: AsyncCollection,
    name: str,
    embeddings_model: ModelConnectionDetails,
    additional_metadata: Mapping[str, str] | None,
) -> AsyncCollection:
    """Re-embed every record of `collection` with a new embedding model and swap the result in.

    Crash-safe: the old collection is parked (renamed) rather than deleted before the replacement
    has taken its name, writers are blocked via the live migration marker, and leftovers from a
    crashed migration make the next run resume from the parked collection. A resume re-stamps a
    fresh marker on both the parked and the live collection, so it too blocks writers and a
    concurrent second migration. Returns a fresh handle to the migrated collection bound to the
    new embedding function.
    """
    embedding_function = build_embedding_function(embeddings_model)
    now_iso = datetime.now(timezone.utc).isoformat()

    parked = await get_parked_source(client, name)
    source = parked if parked is not None else collection
    if parked is None:
        # modify() replaces the whole metadata, so the full dict is sent.
        await collection.modify(
            metadata={
                **(collection.metadata or {}),
                MIGRATION_IN_PROGRESS_METADATA_KEY: True,
                MIGRATION_STARTED_AT_METADATA_KEY: now_iso,
            }
        )
        logger.info(
            "Migrating collection '%s' from embeddings model %r to %r",
            name,
            (source.metadata or {}).get(EMBEDDINGS_MODEL_NAME_METADATA_KEY),
            embeddings_model.model_name,
        )
    else:
        # Resume: refresh the in-progress protection. A fresh marker on the parked collection
        # stops a concurrent run from starting a second migration of the same collection, and a
        # marker on the live collection blocks writers while its records are re-embedded and the
        # collection is taken over.
        for target in (parked, collection):
            await target.modify(
                metadata={
                    **(target.metadata or {}),
                    MIGRATION_IN_PROGRESS_METADATA_KEY: True,
                    MIGRATION_STARTED_AT_METADATA_KEY: now_iso,
                }
            )
        logger.info("Resuming migration of collection '%s' from parked collection '%s'", name, parked.name)

    temp: AsyncCollection | None = None
    try:
        await _cleanup_stale_temp_collections(client, source.id)
        temp_name = f"{name}{MIGRATION_TEMP_NAME_PREFIX}{uuid.uuid4().hex[:12]}"
        # Carry over the source collection's metadata so the migrated collection is indistinguishable
        # from a newly created one, except for the model name and the migration bookkeeping keys.
        inherited_metadata = {k: v for k, v in (source.metadata or {}).items() if k not in _MIGRATION_METADATA_KEYS}
        temp = await client.create_collection(
            name=temp_name,
            embedding_function=embedding_function,
            metadata={
                **inherited_metadata,
                **(additional_metadata or {}),
                EMBEDDINGS_MODEL_NAME_METADATA_KEY: embeddings_model.model_name,
                MIGRATION_TEMP_FOR_METADATA_KEY: str(source.id),
                MIGRATION_STARTED_AT_METADATA_KEY: now_iso,
            },
        )
        logger.info("Re-embedding records of collection '%s' into temp collection '%s'", name, temp_name)
        await _copy_records(source, temp, name)
        await _swap_collections(client, collection, name, temp, parked)
        await _strip_migration_metadata(client, name)
    except BaseException:
        if temp is not None:
            await safe_delete(client, temp.name)
        # Clear the migration markers, so the next run can retry or resume. For the parked
        # collection (possibly just created by this run's swap) the marker keys are removed
        # entirely, since its timestamp freshness is what other runs read to detect an
        # in-progress migration; leaving a fresh timestamp would wedge the collection until
        # the lease expires.
        await _clear_migration_marker(client, collection)
        await _clear_parked_marker(client, name)
        raise
    logger.info("Migration of collection '%s' to embeddings model %r completed", name, embeddings_model.model_name)
    return await client.get_collection(name=name, embedding_function=embedding_function)
