"""Checkpointer factories."""

from collections.abc import Awaitable, Callable
from typing import Any, cast
from urllib.parse import quote

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from tech_market_researcher.core.configuration import (
    CheckpointerBackend,
    CheckpointerConfig,
    PostgresCheckpointerConfig,
)
from tech_market_researcher.core.constants import POSTGRES_SCHEMA_NAME_PATTERN


class CloseableCheckpointerWrapper:  # pylint: disable=too-few-public-methods
    """A checkpointer wrapper that adds `close()` for releasing the backend's resources."""

    def __init__(self, checkpointer: BaseCheckpointSaver[Any], close: Callable[[], Awaitable[None]]) -> None:
        self.checkpointer = checkpointer
        self._close = close

    async def close(self) -> None:
        """Release the backend's resources. A no-op for backends that hold nothing."""
        await self._close()


async def _noop_close() -> None:
    pass


async def get_checkpointer(config: CheckpointerConfig) -> CloseableCheckpointerWrapper:
    """Create the checkpointer selected by the config.

    Must be awaited on the event loop the agent is run on: the postgres backend
    creates its async connection pool here, which binds to the loop it is opened on.
    """
    match config.backend:
        case CheckpointerBackend.MEMORY:
            return CloseableCheckpointerWrapper(MemorySaver(), _noop_close)
        case CheckpointerBackend.POSTGRES:
            assert config.postgres is not None
            return await _create_postgres_saver(config.postgres)


def _build_postgres_connection_info(config: PostgresCheckpointerConfig) -> str:
    """Build a Postgres connection string from the config; the password comes from the env."""
    password = quote(config.db_password.get_secret_value(), safe="")
    return f"postgresql://{config.db_user}:{password}@{config.host}:{config.port}/{config.db_name}"


async def _create_postgres_saver(config: PostgresCheckpointerConfig) -> CloseableCheckpointerWrapper:
    """Create a checkpointer over a connection pool, creating the schema and tables if missing.

    The checkpoint tables live in the configured schema: the pool's connections get a
    search_path pointing at it, so the checkpointer's unqualified table names resolve there.
    """
    if not POSTGRES_SCHEMA_NAME_PATTERN.match(config.schema_name):
        raise ValueError(f"Invalid schema name '{config.schema_name}'.")
    connection_info = _build_postgres_connection_info(config)
    async with await AsyncConnection.connect(connection_info) as conn:
        await conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{config.schema_name}"')
    pool = AsyncConnectionPool(
        # Connection string used for every connection the pool opens.
        conninfo=connection_info,
        kwargs={
            # Commit each statement immediately instead of inside a transaction block.
            "autocommit": True,
            # Disable server-side prepared statements.
            "prepare_threshold": 0,
            # Return query rows as dicts, which the checkpointer expects.
            "row_factory": dict_row,
            # Set the schema search_path so unqualified table names resolve to the checkpoint tables.
            "options": f"-c search_path={config.schema_name}",
        },
        # Defer opening the pool until pool.open() is awaited.
        open=False,
    )
    await pool.open()
    checkpointer = AsyncPostgresSaver(cast(AsyncConnectionPool[AsyncConnection[dict[str, Any]]], pool))
    await checkpointer.setup()

    async def close_pool() -> None:
        await pool.close()

    return CloseableCheckpointerWrapper(checkpointer, close_pool)
