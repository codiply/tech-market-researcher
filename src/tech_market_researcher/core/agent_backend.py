"""Helpers for the agent's deep agents backend."""

import base64
import importlib.metadata
import logging
import shlex
from collections.abc import Sequence

from deepagents.backends import BackendProtocol, FilesystemBackend
from deepagents.backends.protocol import ExecuteResponse, FileUploadResponse
from langchain_kubernetes import KubernetesProvider, KubernetesProviderConfig, KubernetesSandbox
from langchain_kubernetes.backends.raw import RawK8sBackend

from tech_market_researcher.core.common import AsyncReleasableResource
from tech_market_researcher.core.configuration import (
    AgentBackendConfig,
    AgentBackendKind,
    FilesystemBackendConfig,
    SandboxBackendConfig,
)
from tech_market_researcher.core.constants import AGENT_BACKEND_MAX_FILE_SIZE_MB

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


# Versions of langchain-kubernetes whose raw-mode upload (backends/raw_transport.upload_files_tar)
# never closes the pod's stdin after writing the tar archive, so the pod-side `base64 -d | tar`
# waits for EOF forever and the upload blocks until the exec timeout. Fix once a release
# addresses it: remove this workaround (K8sSandbox, _disable_buggy_kubernetes_upload, this set).
_K8S_BUGGY_UPLOAD_VERSIONS = {"0.4.0"}


class K8sSandbox(KubernetesSandbox):
    """KubernetesSandbox with a working upload path on versions where the native one is broken.

    The inherited fallback is unusable (BaseSandbox.upload_files is abstract), so when the
    native upload raises (it is disabled on known-buggy versions, see
    _disable_buggy_kubernetes_upload) each file is transferred as base64 via execute: the
    encoded payload is written to a temp file, decoded in place, and the temp file removed.
    """

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        try:
            return self._backend.upload_files(files)
        except Exception:  # pylint: disable=broad-exception-caught
            logger.warning("Kubernetes upload unavailable; transferring files as base64 over execute")
            return [self._upload_file_via_execute(path, content) for path, content in files]

    def _upload_file_via_execute(self, path: str, content: bytes) -> FileUploadResponse:
        payload = base64.b64encode(content).decode("ascii")
        temp_path = f"{path}.b64"
        result = self.execute(
            f'mkdir -p "$(dirname {shlex.quote(path)})" '
            f"&& printf '%s' {shlex.quote(payload)} > {shlex.quote(temp_path)} "
            f"&& base64 -d {shlex.quote(temp_path)} > {shlex.quote(path)} "
            f"&& rm {shlex.quote(temp_path)}"
        )
        if isinstance(result, ExecuteResponse) and result.exit_code not in (0, None):
            return FileUploadResponse(
                path=path, error=f"base64 upload failed (exit {result.exit_code}): {result.output}"
            )
        return FileUploadResponse(path=path, error=None)


def _disabled_raw_upload_files(self: RawK8sBackend, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
    raise RuntimeError(
        "kubernetes raw upload disabled: known stdin EOF bug in the installed langchain-kubernetes version"
    )


def _disable_buggy_kubernetes_upload() -> None:
    """Make the raw-mode upload raise instead of hanging, on known-buggy versions.

    K8sSandbox catches the raised exception and transfers files as base64 over execute,
    which is unaffected by the bug. No-op once a fixed version is installed, so this
    self-disables.
    """
    version = importlib.metadata.version("langchain_kubernetes")
    if version not in _K8S_BUGGY_UPLOAD_VERSIONS:
        return
    if RawK8sBackend.upload_files is _disabled_raw_upload_files:
        return  # Already patched by an earlier call in this process.

    RawK8sBackend.upload_files = _disabled_raw_upload_files  # type: ignore[method-assign]
    logger.warning(
        "langchain-kubernetes %s raw upload is disabled (stdin EOF bug); "
        "file uploads will use the base64-via-execute fallback",
        version,
    )


def build_agent_backend(
    backend_config: AgentBackendConfig,
) -> tuple[BackendProtocol, Sequence[AsyncReleasableResource]]:
    """Build the agent backend selected by backend_config.

    Returns the backend together with any async resources that must be released when the
    agent shuts down (the filesystem backend releases nothing; the sandbox backend releases
    its sandbox pod).
    """
    if backend_config.kind is AgentBackendKind.FILESYSTEM:
        return _build_filesystem_backend(backend_config.filesystem), ()
    return _build_sandbox_backend(backend_config.sandbox)


def _build_filesystem_backend(config: FilesystemBackendConfig) -> BackendProtocol:
    """Build the on-disk filesystem backend."""
    logger.info("Building filesystem backend with root %s", config.root_dir)
    return FilesystemBackend(
        root_dir=config.root_dir,
        virtual_mode=True,
        max_file_size_mb=AGENT_BACKEND_MAX_FILE_SIZE_MB,
    )


def _build_sandbox_backend(
    sandbox_config: SandboxBackendConfig,
) -> tuple[BackendProtocol, Sequence[AsyncReleasableResource]]:
    """Build the Kubernetes sandbox backend."""
    _disable_buggy_kubernetes_upload()
    logger.info(
        "Building Kubernetes sandbox backend: namespace=%s, image=%s",
        sandbox_config.namespace,
        sandbox_config.image,
    )
    provider = KubernetesProvider(
        KubernetesProviderConfig(
            mode="raw",
            namespace=sandbox_config.namespace,
            image=sandbox_config.image,
            workdir=sandbox_config.workdir,
            cpu_request=sandbox_config.cpu_request,
            cpu_limit=sandbox_config.cpu_limit,
            memory_request=sandbox_config.memory_request,
            memory_limit=sandbox_config.memory_limit,
            startup_timeout_seconds=sandbox_config.startup_timeout_seconds,
            default_exec_timeout=sandbox_config.default_exec_timeout_seconds,
        )
    )
    kubernetes_sandbox = provider.get_or_create()
    logger.info("Kubernetes sandbox ready: id=%s", kubernetes_sandbox.id)
    # The provider's sandbox is just a wrapper around its execution backend; reuse it to avoid
    # a second connection.
    sandbox = K8sSandbox(backend=kubernetes_sandbox._backend)  # pylint: disable=protected-access
    return sandbox, (KubernetesSandboxReleasableResource(provider, kubernetes_sandbox.id),)


class KubernetesSandboxReleasableResource:  # pylint: disable=too-few-public-methods
    """Releases the Kubernetes sandbox pod when the agent shuts down."""

    def __init__(self, provider: KubernetesProvider, sandbox_id: str) -> None:
        self._provider = provider
        self._sandbox_id = sandbox_id

    async def close(self) -> None:
        """Delete the sandbox pod."""
        self._provider.delete(sandbox_id=self._sandbox_id)
        logger.info("Deleted Kubernetes sandbox %s", self._sandbox_id)
