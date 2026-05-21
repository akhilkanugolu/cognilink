"""CogniLink custom exceptions.

All exceptions inherit from a base CogniLinkError class to enable
catch-all handling while preserving specific error semantics.
"""

from typing import List, Optional


class CogniLinkError(Exception):
    """Base exception for all CogniLink errors."""


class NodeAlreadyExistsError(CogniLinkError):
    """Raised when attempting to insert a node with an ID that already exists in the graph store."""

    def __init__(self, node_id: str) -> None:
        self.node_id = node_id
        super().__init__(f"Node '{node_id}' already exists in the graph store")


class NodeNotFoundError(CogniLinkError):
    """Raised when a referenced node ID does not exist in the graph store."""

    def __init__(self, node_id: str) -> None:
        self.node_id = node_id
        super().__init__(f"Node '{node_id}' not found in the graph store")


class CycleDetectedError(CogniLinkError):
    """Raised when a cycle is detected in the dependency graph during topological sort or traversal.

    Accepts a list of node IDs involved in the cycle for diagnostic purposes.
    """

    def __init__(self, involved_nodes: List[str], message: Optional[str] = None) -> None:
        self.involved_nodes = involved_nodes
        msg = message or f"Cycle detected among nodes: {involved_nodes}"
        super().__init__(msg)


class LLMUnavailableError(CogniLinkError):
    """Raised when the configured LLM provider is unreachable or returns an error during inference.

    Nodes that could not be summarized remain in PENDING state and can be retried
    once the provider is available again.
    """

    def __init__(self, provider: str = "unknown", reason: str = "") -> None:
        self.provider = provider
        self.reason = reason
        msg = f"LLM provider '{provider}' is unavailable"
        if reason:
            msg += f": {reason}"
        super().__init__(msg)


class DatabaseCorruptionError(CogniLinkError):
    """Raised when a persistent SQLite database file is corrupted or has a schema mismatch.

    The existing file is never silently overwritten — the caller must resolve
    the corruption before proceeding.
    """

    def __init__(self, db_path: str, reason: str = "") -> None:
        self.db_path = db_path
        self.reason = reason
        msg = f"Database at '{db_path}' is corrupted or has a schema mismatch"
        if reason:
            msg += f": {reason}"
        super().__init__(msg)


class VisualizationWriteError(CogniLinkError):
    """Raised when the visualization output directory is not writable or a write operation fails.

    The in-memory graph state remains valid even when visualization output fails.
    """

    def __init__(self, output_path: str, reason: str = "") -> None:
        self.output_path = output_path
        self.reason = reason
        msg = f"Failed to write visualization output to '{output_path}'"
        if reason:
            msg += f": {reason}"
        super().__init__(msg)


class InvalidStateTransitionError(CogniLinkError):
    """Raised when an invalid node state transition is attempted.

    Valid transitions are: PENDING→RUNNING, RUNNING→COMPLETED, any→STALE, STALE→RUNNING.
    """

    def __init__(self, current_state: str, attempted_state: str) -> None:
        self.current_state = current_state
        self.attempted_state = attempted_state
        super().__init__(
            f"Invalid state transition from '{current_state}' to '{attempted_state}'"
        )
