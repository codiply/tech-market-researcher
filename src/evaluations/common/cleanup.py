"""Post-evaluation cleanup of the evals vector store."""

import asyncio
import logging

from tech_market_researcher.core.arguments import AgentCommandLineArguments
from tech_market_researcher.core.configuration import load_config
from tech_market_researcher.integrations.chroma import delete_collections_with_metadata, get_client

logger = logging.getLogger(__name__)

# The tag the agent config (evals mixin) stamps on every collection the agent creates, so the
# cleanup can identify exactly the collections made during the run.
EVALS_COLLECTION_TAG: dict[str, str] = {"created_during_evals": "true"}


def cleanup_evals_collections(agent_args: AgentCommandLineArguments) -> None:
    """Delete every collection in the evals vector store tagged as created during evals.

    Only collections carrying the EVALS_COLLECTION_TAG are deleted, so any untagged
    collection is left untouched. Failures are logged, never raised: the cleanup runs
    after the evaluation, so it must not mask the evaluation's own outcome.
    """
    try:
        config = load_config(agent_args.config_name, agent_args.extra_mixins)

        async def _delete_tagged_collections() -> None:
            client = await get_client(config.indexing.chroma_url_env_var)
            deleted = await delete_collections_with_metadata(client, EVALS_COLLECTION_TAG)
            logger.info("Deleted %d evals collection(s) tagged with %s", deleted, EVALS_COLLECTION_TAG)

        asyncio.run(_delete_tagged_collections())
    except Exception:  # pylint: disable=broad-exception-caught
        logger.exception("Failed to clean up the evals vector store; the tagged collections may need manual deletion")
