"""Agent tools for a document index: upserting documents and querying them."""

import logging
import uuid
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, cast

import mlflow
from chromadb.api.models.AsyncCollection import AsyncCollection
from chromadb.types import Where
from langchain.tools import ToolRuntime
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field, create_model, model_validator

from tech_market_researcher.capabilities.chunking import get_token_encoder, split_text_into_chunks
from tech_market_researcher.core.common import AgentRuntimeContext
from tech_market_researcher.core.configuration import ModelConnectionDetails
from tech_market_researcher.integrations.chroma.collection import (
    ChromaDocument,
    EmbeddingModelMismatchBehavior,
    collection_name_for_thread,
    delete_where,
    get_or_create_collection,
)
from tech_market_researcher.integrations.chroma.collection import query as chroma_query
from tech_market_researcher.integrations.chroma.collection import upsert_documents

INDEXING_TOOL_NAME_UPSERT_SUFFIX = "index_upsert"
INDEXING_TOOL_NAME_QUERY_SUFFIX = "index_query"
QUERY_INDEX_MAX_RESULTS_DEFAULT = 5
QUERY_INDEX_MAX_RESULTS_MAX = 25
METADATA_PARAMETER_NAME_PREFIX = "metadata_"
FILTER_PARAMETER_NAME_PREFIX = "filter_"

# Reserved metadata keys written automatically by the upsert tool.
SOURCE_ID_METADATA_KEY = "_source_id"
CHUNK_INDEX_METADATA_KEY = "_chunk_index"
CHUNK_COUNT_METADATA_KEY = "_chunk_count"

logger = logging.getLogger(__name__)


class DocumentRecord(BaseModel):
    """A document's key and content, the input to metadata mappers."""

    key: str
    content: str


class MetadataParameter(BaseModel):
    """A metadata field of the collection, exposed as a dedicated upsert tool parameter."""

    name: str
    description: str


class FetchedDocument(BaseModel):
    """A document fetched server-side from its ID, with optional record metadata."""

    content: str
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)


class DocumentMetadataMapper(BaseModel):
    """An extra record metadata field, computed server-side from the record's key and content."""

    name: str
    description: str
    # Maps the record's key and content to the metadata value, or None to omit the field.
    mapper: Callable[[DocumentRecord], str | None]


class MetadataField(BaseModel):
    """The name and description of a record metadata field the query tool can filter on."""

    name: str
    description: str


class IndexToolsParams(BaseModel):
    """Parameters for building the index tools for a collection."""

    tool_name_prefix: str
    collection_name: str
    collection_description: str
    document_description: str = "The document text to index"
    is_document_id_required: bool = False
    document_id_description: str = "Optional explicit record ID; a UUID is generated when omitted"
    metadata_parameters: list[MetadataParameter] | None = None
    # Extra metadata fields whose value is computed from the document's key and content
    # instead of provided by the model. Each is also exposed as a query filter argument.
    metadata_mappers: list[DocumentMetadataMapper] | None = None
    embeddings_model: ModelConnectionDetails
    document_fetcher: Callable[[str], Awaitable[FetchedDocument]] | None = None
    chunk_size_tokens: int | None = None
    chunk_overlap_tokens: int = 50
    # Suffix the collection name with the conversation's thread id, so each conversation
    # reads from and writes to its own collection. The tools then resolve the collection
    # at call time, from the tool runtime's context, instead of up front.
    do_suffix_collection_name_with_thread_id: bool = False
    # Extra metadata recorded on every collection created for the tools.
    additional_collection_metadata: dict[str, str] = Field(default_factory=dict)
    # What to do when an existing collection was created with a different embeddings model.
    mismatch_behavior: EmbeddingModelMismatchBehavior = Field(default=EmbeddingModelMismatchBehavior.RAISE)

    @model_validator(mode="after")
    def _validate_document_fetcher_fields(self) -> "IndexToolsParams":
        if self.document_fetcher is not None and not self.is_document_id_required:
            raise ValueError("document_fetcher requires is_document_id_required to be true")
        if self.chunk_size_tokens is not None:
            if self.document_fetcher is None:
                raise ValueError("chunk_size_tokens requires document_fetcher")
            if not 0 <= self.chunk_overlap_tokens < self.chunk_size_tokens:
                raise ValueError("chunk_overlap_tokens must be less than chunk_size_tokens")
        return self

    @model_validator(mode="after")
    def _validate_metadata_field_names(self) -> "IndexToolsParams":
        names = [p.name for p in self.metadata_parameters or []] + [m.name for m in self.metadata_mappers or []]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"duplicate metadata field names: {duplicates}")
        return self


def _build_upsert_args_model(params: IndexToolsParams) -> type[BaseModel]:
    """Build the upsert tool args schema, adding one optional str field per metadata parameter.

    When a document fetcher is configured, the schema carries no document text: the document is
    fetched server-side from the given document ID, so the model only provides the ID and metadata.
    """
    parameters = params.metadata_parameters or []
    # Unique per tool prefix, so each index tool's upsert args schema carries a distinct name.
    prefix = params.tool_name_prefix if params.tool_name_prefix.endswith("_") else f"{params.tool_name_prefix}_"
    schema_name = f"{prefix}IndexUpsertArgs"
    metadata_fields: dict[str, Any] = {
        f"{METADATA_PARAMETER_NAME_PREFIX}{p.name}": (str | None, Field(default=None, description=p.description))
        for p in parameters
    }
    if params.document_fetcher is not None:
        return create_model(
            schema_name,
            document_id=(str, Field(description=params.document_id_description)),
            **metadata_fields,
        )
    document_id_field: Any = (
        (str, Field(description=params.document_id_description))
        if params.is_document_id_required
        else (str | None, Field(default=None, description=params.document_id_description))
    )
    return create_model(
        schema_name,
        document=(str, Field(description=params.document_description)),
        document_id=document_id_field,
        **metadata_fields,
    )


def _metadata_arg_lines(parameters: list[MetadataParameter]) -> list[str]:
    return [f"    {METADATA_PARAMETER_NAME_PREFIX}{p.name}: {p.description}" for p in parameters]


def _metadata_field_specs(params: IndexToolsParams) -> list[MetadataField]:
    """Every record metadata field the query tool can filter on."""
    fields = [MetadataField(name=p.name, description=p.description) for p in params.metadata_parameters or []]
    fields.extend(MetadataField(name=m.name, description=m.description) for m in params.metadata_mappers or [])
    return fields


def _metadata_from_args(
    metadata_fields: dict[str, str], parameters: list[MetadataParameter]
) -> dict[str, str | int | float | bool]:
    """Extract the record metadata from the tool's metadata_<name> arguments."""
    metadata: dict[str, str | int | float | bool] = {}
    for parameter in parameters:
        value = metadata_fields.get(f"{METADATA_PARAMETER_NAME_PREFIX}{parameter.name}")
        if value is not None:
            metadata[parameter.name] = value
    return metadata


def _mapped_metadata(mappers: list[DocumentMetadataMapper], record: DocumentRecord) -> dict[str, str]:
    """Compute the mapper-derived metadata values, omitting fields whose mapper returns None."""
    return {mapper.name: value for mapper in mappers if (value := mapper.mapper(record)) is not None}


def _build_record_metadata(
    tool_args_metadata: Mapping[str, str | int | float | bool],
    fetched_metadata: Mapping[str, str | int | float | bool],
    mapper_metadata: Mapping[str, str | int | float | bool],
) -> dict[str, str | int | float | bool]:
    """Merge the record metadata layers into one dict; later arguments take precedence over earlier ones."""
    metadata = dict(tool_args_metadata)
    metadata.update(fetched_metadata)
    metadata.update(mapper_metadata)
    return metadata


def _build_upsert_docstring(
    params: IndexToolsParams, parameters: list[MetadataParameter], per_conversation: bool = False
) -> str:
    """Build the upsert tool docstring, including the collection description and argument docs."""
    collection_sentence = (
        f"Add a document to the {params.collection_name} collection for this conversation, "
        "replacing any existing records of the same ID. Only the current conversation's documents are visible."
        if per_conversation
        else f"Add a document to the {params.collection_name} collection, "
        "replacing any existing records of the same ID."
    )
    if params.document_fetcher is not None:
        document_section_lines = [
            "The document content is fetched server-side from the given document ID and stored "
            "directly; you only provide the ID and metadata, never the content.",
            "",
        ]
        document_args_lines: list[str] = []
    else:
        document_section_lines = []
        document_args_lines = [f"    document: {params.document_description}"]
    return "\n".join(
        [
            collection_sentence,
            "",
            "About this collection:",
            params.collection_description,
            *document_section_lines,
            "Args:",
            *document_args_lines,
            f"    document_id: {params.document_id_description}",
            *_metadata_arg_lines(parameters),
            "",
            "Returns:",
            "    The ID of the stored record",
        ]
    )


def _build_query_args_model(params: IndexToolsParams) -> type[BaseModel]:
    """Build the query tool args schema, adding one optional str filter per record metadata field."""
    prefix = params.tool_name_prefix if params.tool_name_prefix.endswith("_") else f"{params.tool_name_prefix}_"
    schema_name = f"{prefix}IndexQueryArgs"
    filter_fields: dict[str, Any] = {
        f"{FILTER_PARAMETER_NAME_PREFIX}{field.name}": (str | None, Field(default=None, description=field.description))
        for field in _metadata_field_specs(params)
    }
    return create_model(
        schema_name,
        query=(str, Field(description="Search query to execute")),
        max_results=(
            int,
            Field(default=QUERY_INDEX_MAX_RESULTS_DEFAULT, description="Maximum number of results to return"),
        ),
        **filter_fields,
    )


def _build_query_docstring(params: IndexToolsParams, per_conversation: bool = False) -> str:
    """Build the query tool docstring, including the collection description and argument docs."""
    collection_sentence = (
        f"Retrieve the documents most relevant to a query from the {params.collection_name} collection "
        "for this conversation. Only the current conversation's documents are visible."
        if per_conversation
        else f"Retrieve the documents most relevant to a query from the {params.collection_name} collection."
    )
    filter_lines = [
        f"    {FILTER_PARAMETER_NAME_PREFIX}{field.name}: Only return documents whose metadata field "
        f"{field.name!r} equals this value (exact match)"
        for field in _metadata_field_specs(params)
    ]
    return "\n".join(
        [
            collection_sentence,
            "",
            "About this collection:",
            params.collection_description,
            "",
            "Args:",
            "    query: Search query to execute",
            "    max_results: Maximum number of results to return (max 25, default: 5)",
            *filter_lines,
            "",
            "Returns:",
            "    Numbered documents with ID and distance",
        ]
    )


async def _build_collection_resolver(
    chroma_client,
    params: IndexToolsParams,
) -> Callable[[ToolRuntime[AgentRuntimeContext]], Awaitable[AsyncCollection]]:
    """Build the per-call collection resolver for the index tools.

    When the collection name is suffixed with the thread id, each call resolves the
    conversation's own collection from the tool runtime's context; otherwise all
    conversations share a single collection fetched (or created) once up front,
    and the runtime is ignored."""
    if params.do_suffix_collection_name_with_thread_id:

        async def get_conversation_collection(
            runtime: ToolRuntime[AgentRuntimeContext],
        ) -> AsyncCollection:
            collection_name = collection_name_for_thread(params.collection_name, runtime.context.thread_id)
            return await get_or_create_collection(
                chroma_client,
                collection_name,
                params.embeddings_model,
                params.mismatch_behavior,
                additional_metadata=params.additional_collection_metadata,
            )

    else:
        # Without per-thread scoping all conversations share a single collection, so it is
        # fetched (or created) once up front and the runtime is ignored.
        shared_collection = await get_or_create_collection(
            chroma_client,
            params.collection_name,
            params.embeddings_model,
            params.mismatch_behavior,
            additional_metadata=params.additional_collection_metadata,
        )

        async def get_conversation_collection(
            runtime: ToolRuntime[AgentRuntimeContext],  # pylint: disable=unused-argument
        ) -> AsyncCollection:
            return shared_collection

    return get_conversation_collection


async def build_index_tools(
    chroma_client,
    params: IndexToolsParams,
) -> list[BaseTool]:
    """Return the index upsert and query tools for the named Chroma collection.

    The tool names are the prefix (a trailing underscore is added if missing)
    followed by the upsert/query suffixes. The collection is resolved on every call
    through a conversation-collection helper: when the collection name is suffixed with
    the thread id, each call resolves the conversation's own collection from the tool
    runtime's context; otherwise the same helper returns the single collection fetched
    (or created) up front. The collection description is injected into the
    docstrings of the generated tools. Each metadata parameter is exposed as
    a dedicated optional `metadata_<name>` argument of the upsert tool whose
    value is stored under the parameter's name in the record's metadata; each
    metadata mapper is computed server-side from the record's key and content,
    and every such field is exposed as an optional `filter_<name>` argument of
    the query tool (exact match)."""
    parameters = params.metadata_parameters or []
    tool_name_prefix = params.tool_name_prefix
    if not tool_name_prefix.endswith("_"):
        tool_name_prefix += "_"
    get_conversation_collection = await _build_collection_resolver(chroma_client, params)
    token_encoder = get_token_encoder() if params.chunk_size_tokens is not None else None
    query_span_name = f"{tool_name_prefix}{INDEXING_TOOL_NAME_QUERY_SUFFIX}"

    async def upsert_index(
        collection: AsyncCollection, document: str, document_id: str | None = None, **metadata_fields: str
    ) -> str:
        record_id = document_id or str(uuid.uuid4())
        tool_args_metadata = _metadata_from_args(metadata_fields, parameters)
        mapper_metadata = _mapped_metadata(
            params.metadata_mappers or [], DocumentRecord(key=record_id, content=document)
        )
        metadata = _build_record_metadata(tool_args_metadata, {}, mapper_metadata)
        logger.info("Upserting 1 document into collection '%s' (id=%s)", collection.name, record_id)
        await upsert_documents(collection, [ChromaDocument(id=record_id, document=document, metadata=metadata)])
        return f"Stored document with id {record_id}"

    async def upsert_index_fetched(collection: AsyncCollection, document_id: str, **metadata_fields: str) -> str:
        assert params.document_fetcher is not None
        logger.info("Fetching and indexing document %s into collection '%s'", document_id, collection.name)
        fetched = await params.document_fetcher(document_id)
        content = fetched.content
        tool_args_metadata = _metadata_from_args(metadata_fields, parameters)
        mapper_metadata = _mapped_metadata(
            params.metadata_mappers or [], DocumentRecord(key=document_id, content=content)
        )
        # Mapper-derived metadata takes precedence over fetched and tool-argument values.
        metadata = _build_record_metadata(tool_args_metadata, fetched.metadata, mapper_metadata)
        if params.chunk_size_tokens is not None:
            assert token_encoder is not None
            chunks = split_text_into_chunks(
                content, token_encoder, params.chunk_size_tokens, params.chunk_overlap_tokens
            )
            # Drop the previous chunks of this document, so a shorter re-fetch does not leave stale chunks behind.
            await delete_where(collection, {SOURCE_ID_METADATA_KEY: document_id})
            documents = [
                ChromaDocument(
                    id=f"{document_id}#{chunk_index}",
                    document=chunk,
                    metadata={
                        **metadata,
                        SOURCE_ID_METADATA_KEY: document_id,
                        CHUNK_INDEX_METADATA_KEY: chunk_index,
                        CHUNK_COUNT_METADATA_KEY: len(chunks),
                    },
                )
                for chunk_index, chunk in enumerate(chunks)
            ]
            chunk_suffix = f" in {len(chunks)} chunks"
        else:
            documents = [ChromaDocument(id=document_id, document=content, metadata=metadata)]
            chunk_suffix = ""
        logger.debug(
            "Upserting %d document(s) into collection '%s' (document_id=%s)",
            len(documents),
            collection.name,
            document_id,
        )
        await upsert_documents(collection, documents)
        return f"Stored document with id {document_id}{chunk_suffix}"

    async def query_index(
        collection: AsyncCollection,
        query: str,
        max_results: int = QUERY_INDEX_MAX_RESULTS_DEFAULT,
        where: Where | None = None,
    ) -> str:
        effective_max_results = min(max_results, QUERY_INDEX_MAX_RESULTS_MAX)
        logger.info(
            "Querying collection '%s' (max_results=%d, where=%s)",
            collection.name,
            effective_max_results,
            where or None,
        )
        with mlflow.start_span(name=query_span_name, span_type="RETRIEVER") as span:
            span.set_inputs({"query": query, "max_results": effective_max_results, "where": where or None})
            result = await chroma_query(collection, query, n_results=effective_max_results, where=where)
            items = [item for item in result.items[0] if item.document is not None]
            span.set_outputs(
                [
                    {
                        "content": item.document,
                        "metadata": {**(item.metadata or {}), "distance": item.distance},
                    }
                    for item in items
                ]
            )
        result_texts = [
            f"{index}. **{item.id}** (distance: {item.distance:.4f})\n   {item.document}".rstrip()
            for index, item in enumerate(items, start=1)
        ]
        return f"Found {len(items)} result(s) for '{query}':\n\n" + "\n\n".join(result_texts)

    async def upsert_tool_fn(
        runtime: ToolRuntime[AgentRuntimeContext],
        document: str | None = None,
        document_id: str | None = None,
        **metadata_fields: str,
    ) -> str:
        collection = await get_conversation_collection(runtime)
        if params.document_fetcher is not None:
            assert document_id is not None, "document_id is required when the document is fetched from a URL"
            return await upsert_index_fetched(collection, document_id, **metadata_fields)
        assert document is not None
        return await upsert_index(collection, document, document_id, **metadata_fields)

    async def query_tool_fn(
        runtime: ToolRuntime[AgentRuntimeContext],
        query: str,
        max_results: int = QUERY_INDEX_MAX_RESULTS_DEFAULT,
        **filter_fields: str,
    ) -> str:
        filters = {
            name[len(FILTER_PARAMETER_NAME_PREFIX) :]: value
            for name, value in filter_fields.items()
            if value is not None
        }
        where: Where | None = cast("Where | None", filters) if filters else None
        return await query_index(await get_conversation_collection(runtime), query, max_results, where)

    upsert_tool = StructuredTool.from_function(
        coroutine=upsert_tool_fn,
        name=f"{tool_name_prefix}{INDEXING_TOOL_NAME_UPSERT_SUFFIX}",
        description=_build_upsert_docstring(
            params, parameters, per_conversation=params.do_suffix_collection_name_with_thread_id
        ),
        args_schema=_build_upsert_args_model(params),
    )
    query_tool = StructuredTool.from_function(
        coroutine=query_tool_fn,
        name=f"{tool_name_prefix}{INDEXING_TOOL_NAME_QUERY_SUFFIX}",
        description=_build_query_docstring(params, per_conversation=params.do_suffix_collection_name_with_thread_id),
        args_schema=_build_query_args_model(params),
    )
    return [upsert_tool, query_tool]
