"""Agent tools for the web page index: upserting pages and querying them."""

import uuid
from typing import Annotated

from langchain_core.tools import BaseTool, StructuredTool

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


async def build_indexing_tools(
    chroma_client,
    collection_name: str,
    embedding_model_name: str,
    tool_name_prefix: str,
) -> list[BaseTool]:
    """Return the index upsert and query tools for the named Chroma collection.

    The tool names are the prefix (a trailing underscore is added if missing)
    followed by the upsert/query suffixes. The collection is fetched (or
    created) up front."""
    if not tool_name_prefix.endswith("_"):
        tool_name_prefix += "_"
    collection = await get_or_create_collection(
        chroma_client,
        collection_name,
        embedding_model_name,
        EmbeddingModelMismatchBehavior.RAISE,
    )

    async def upsert_web_page_index(
        document: str,
        document_id: Annotated[str | None, "Optional explicit ID; a UUID is generated when omitted"] = None,
        metadata: Annotated[
            dict[str, str | int | float | bool] | None, "Optional metadata to store with the document"
        ] = None,
    ) -> str:
        """Add a document to the web page index, replacing any existing record with the same ID.

        Args:
            document: The document text to index
            document_id: Optional explicit record ID; a UUID is generated when omitted
            metadata: Optional metadata to store with the document

        Returns:
            The ID of the stored record
        """
        record_id = document_id or str(uuid.uuid4())
        await upsert_documents(collection, [ChromaDocument(id=record_id, document=document, metadata=metadata or {})])
        return f"Stored document with id {record_id}"

    async def query_web_page_index(
        query: str,
        max_results: int = QUERY_INDEX_MAX_RESULTS_DEFAULT,
    ) -> str:
        """Retrieve the documents most relevant to a query from the web page index.

        Args:
            query: Search query to execute
            max_results: Maximum number of results to return (max 25, default: 5)

        Returns:
            Numbered documents with ID and distance
        """
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
        coroutine=upsert_web_page_index,
        name=f"{tool_name_prefix}{INDEXING_TOOL_NAME_UPSERT_SUFFIX}",
        description=upsert_web_page_index.__doc__,
    )
    query_tool = StructuredTool.from_function(
        coroutine=query_web_page_index,
        name=f"{tool_name_prefix}{INDEXING_TOOL_NAME_QUERY_SUFFIX}",
        description=query_web_page_index.__doc__,
    )
    return [upsert_tool, query_tool]
