"""Agent tools for a document index: upserting documents and querying them."""

import uuid
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field, create_model

from tech_market_researcher.integrations.chroma import (
    ChromaDocument,
    EmbeddingModelMismatchBehavior,
    get_or_create_collection,
)
from tech_market_researcher.integrations.chroma import query as chroma_query
from tech_market_researcher.integrations.chroma import upsert_documents

INDEXING_TOOL_NAME_UPSERT_SUFFIX = "index_upsert"
INDEXING_TOOL_NAME_QUERY_SUFFIX = "index_query"
QUERY_INDEX_MAX_RESULTS_DEFAULT = 5
QUERY_INDEX_MAX_RESULTS_MAX = 25
METADATA_PARAMETER_NAME_PREFIX = "metadata_"


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


def _build_upsert_args_model(params: IndexToolsParams) -> type[BaseModel]:
    """Build the upsert tool args schema, adding one optional str field per metadata parameter."""
    parameters = params.metadata_parameters or []
    metadata_fields: dict[str, Any] = {
        f"{METADATA_PARAMETER_NAME_PREFIX}{p.name}": (str | None, Field(default=None, description=p.description))
        for p in parameters
    }
    document_id_field: Any = (
        (str, Field(description=params.document_id_description))
        if params.is_document_id_required
        else (str | None, Field(default=None, description=params.document_id_description))
    )
    return create_model(
        "IndexUpsertArgs",
        document=(str, Field(description=params.document_description)),
        document_id=document_id_field,
        **metadata_fields,
    )


def _metadata_arg_lines(parameters: list[MetadataParameter]) -> list[str]:
    return [f"    {METADATA_PARAMETER_NAME_PREFIX}{p.name}: {p.description}" for p in parameters]


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

    upsert_docstring = "\n".join(
        [
            f"Add a document to the {params.collection_name} collection, "
            "replacing any existing record with the same ID.",
            "",
            "About this collection:",
            params.collection_description,
            "",
            "Args:",
            f"    document: {params.document_description}",
            f"    document_id: {params.document_id_description}",
            *_metadata_arg_lines(parameters),
            "",
            "Returns:",
            "    The ID of the stored record",
        ]
    )

    query_docstring = "\n".join(
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

    async def upsert_index(document: str, document_id: str | None = None, **metadata_fields: str) -> str:
        metadata: dict[str, str | int | float | bool] = {}
        for parameter in parameters:
            value = metadata_fields.get(f"{METADATA_PARAMETER_NAME_PREFIX}{parameter.name}")
            if value is not None:
                metadata[parameter.name] = value
        record_id = document_id or str(uuid.uuid4())
        await upsert_documents(collection, [ChromaDocument(id=record_id, document=document, metadata=metadata)])
        return f"Stored document with id {record_id}"

    async def query_index(
        query: str,
        max_results: int = QUERY_INDEX_MAX_RESULTS_DEFAULT,
    ) -> str:
        effective_max_results = min(max_results, QUERY_INDEX_MAX_RESULTS_MAX)
        result = await chroma_query(collection, query, n_results=effective_max_results)
        items = result.items[0]
        result_texts = [
            f"{index}. **{item.id}** (distance: {item.distance:.4f})\n   {item.document}".rstrip()
            for index, item in enumerate(items, start=1)
            if item.document is not None
        ]
        return f"Found {len(items)} result(s) for '{query}':\n\n" + "\n\n".join(result_texts)

    upsert_tool = StructuredTool.from_function(
        coroutine=upsert_index,
        name=f"{tool_name_prefix}{INDEXING_TOOL_NAME_UPSERT_SUFFIX}",
        description=upsert_docstring,
        args_schema=_build_upsert_args_model(params),
    )
    query_tool = StructuredTool.from_function(
        coroutine=query_index,
        name=f"{tool_name_prefix}{INDEXING_TOOL_NAME_QUERY_SUFFIX}",
        description=query_docstring,
    )
    return [upsert_tool, query_tool]
