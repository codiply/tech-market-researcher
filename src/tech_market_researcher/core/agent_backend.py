"""Helpers for the agent's deep agents backend."""

import logging

from deepagents.backends import BackendProtocol

logger = logging.getLogger(__name__)


def delete_backend_path(backend: BackendProtocol, path: str) -> None:
    """Recursively delete an existing path in the backend, tolerating an absent path.

    Raises NotImplementedError if the backend does not support deletion; the agent requires
    a backend that does.
    """
    try:
        result = backend.delete(path)
    except NotImplementedError as exc:
        msg = f"backend {type(backend).__name__} does not support delete; an agent backend must support deletion"
        raise NotImplementedError(msg) from exc
    if result.error and "not found" not in result.error.lower():
        msg = f"failed to delete {path} in the backend: {result.error}"
        raise ValueError(msg)
    if not result.error:
        logger.info("Removed backend path %s", path)
