"""Agent tools for a document index: upserting documents and querying them."""

import inspect
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

import mlflow
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field, create_model, model_validator

from tech_market_researcher.capabilities.chunking import get_token_encoder, split_text_into_chunks
from tech_market_researcher.integrations.chroma import (
    ChromaDocument,
    EmbeddingModelMismatchBehavior,
    delete_where,
    get_or_create_collection,
)
from tech_market_researcher.integrations.chroma import query as chroma_query
from tech_market_researcher.integrations.chroma import upsert_documents

INDEXING_TOOL_NAME_UPSERT_SUFFIX = "index_upsert"
INDEXING_TOOL_NAME_QUERY_SUFFIX = "index_query"
QUERY_INDEX_MAX_RESULTS_DEFAULT = 5
QUERY_INDEX_MAX_RESULTS_MAX = 25
METADATA_PARAMETER_NAME_PREFIX = "metadata_"

# Reserved metadata keys written automatically by the upsert tool.
SOURCE_ID_METADATA_KEY = "_source_id"
CHUNK_INDEX_METADATA_KEY = "_chunk_index"
CHUNK_COUNT_METADATA_KEY = "_chunk_count"


class MetadataParameter(BaseModel):
    """A metadata field of the collection, exposed as a dedicated upsert tool parameter."""

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
    embedding_model_name: str
    document_fetcher: Callable[[str], str | Awaitable[str]] | None = None
    chunk_size_tokens: int | None = None
    chunk_overlap_tokens: int = 50

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


def _build_upsert_docstring(params: IndexToolsParams, parameters: list[MetadataParameter]) -> str:
    """Build the upsert tool docstring, including the collection description and argument docs."""
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
            f"Add a document to the {params.collection_name} collection, "
            "replacing any existing records of the same ID.",
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


def _build_query_docstring(params: IndexToolsParams) -> str:
    """Build the query tool docstring, including the collection description."""
    return "\n".join(
        [
            f"Retrieve the documents most relevant to a query from the {params.collection_name} collection.",
            "",
            "About this collection:",
            params.collection_description,
            "",
            "Args:",
            "    query: Search query to execute",
            "    max_results: Maximum number of results to return (max 25, default: 5)",
            "",
            "Returns:",
            "    Numbered documents with ID and distance",
        ]
    )


async def build_index_tools(
    chroma_client,
    params: IndexToolsParams,
) -> list[BaseTool]:
    """Return the index upsert and query tools for the named Chroma collection.

    The tool names are the prefix (a trailing underscore is added if missing)
    followed by the upsert/query suffixes. The collection is fetched (or
    created) up front. The collection description is injected into the
    docstrings of the generated tools. Each metadata parameter is exposed as
    a dedicated optional `metadata_<name>` argument of the upsert tool whose
    value is stored under the parameter's name in the record's metadata."""
    parameters = params.metadata_parameters or []
    tool_name_prefix = params.tool_name_prefix
    if not tool_name_prefix.endswith("_"):
        tool_name_prefix += "_"
    collection = await get_or_create_collection(
        chroma_client,
        params.collection_name,
        params.embedding_model_name,
        EmbeddingModelMismatchBehavior.RAISE,
    )
    token_encoder = get_token_encoder() if params.chunk_size_tokens is not None else None
    query_span_name = f"{tool_name_prefix}{INDEXING_TOOL_NAME_QUERY_SUFFIX}"

    async def upsert_index(document: str, document_id: str | None = None, **metadata_fields: str) -> str:
        metadata = _metadata_from_args(metadata_fields, parameters)
        record_id = document_id or str(uuid.uuid4())
        await upsert_documents(collection, [ChromaDocument(id=record_id, document=document, metadata=metadata)])
        return f"Stored document with id {record_id}"

    async def upsert_index_fetched(document_id: str, **metadata_fields: str) -> str:
        assert params.document_fetcher is not None
        content = params.document_fetcher(document_id)
        if inspect.isawaitable(content):
            content = await content
        metadata = _metadata_from_args(metadata_fields, parameters)
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
        await upsert_documents(collection, documents)
        return f"Stored document with id {document_id}{chunk_suffix}"

    async def query_index(
        query: str,
        max_results: int = QUERY_INDEX_MAX_RESULTS_DEFAULT,
    ) -> str:
        effective_max_results = min(max_results, QUERY_INDEX_MAX_RESULTS_MAX)
        with mlflow.start_span(name=query_span_name, span_type="RETRIEVER") as span:
            span.set_inputs({"query": query, "max_results": effective_max_results})
            result = await chroma_query(collection, query, n_results=effective_max_results)
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

    upsert_tool = StructuredTool.from_function(
        coroutine=upsert_index_fetched if params.document_fetcher is not None else upsert_index,
        name=f"{tool_name_prefix}{INDEXING_TOOL_NAME_UPSERT_SUFFIX}",
        description=_build_upsert_docstring(params, parameters),
        args_schema=_build_upsert_args_model(params),
    )
    query_tool = StructuredTool.from_function(
        coroutine=query_index,
        name=f"{tool_name_prefix}{INDEXING_TOOL_NAME_QUERY_SUFFIX}",
        description=_build_query_docstring(params),
    )
    return [upsert_tool, query_tool]
