"""CogniLink core data models.

Defines the fundamental data structures used throughout the CogniLink system:
nodes, edges, session configuration, and result types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel


class NodeState(Enum):
    """Lifecycle state of a node in the graph.

    State transitions:
        PENDING → RUNNING → COMPLETED
        Any state → STALE
        STALE → RUNNING
    """

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    STALE = "STALE"


@dataclass
class Node:
    """An atomic data block in the graph representing an ingested document or section.

    Attributes:
        id: Deterministic ID in the format NODE_{FILENAME_UPPER}.
        source_type: One of PDF, TEXT, EXCEL, SECTION, or CUSTOM.
        source_path: Original file path or URI.
        raw_content: Full extracted text (programmatic, zero LLM tokens).
        summary: 2-3 sentence LLM-generated summary.
        system_role_prompt: Default system prompt for this node type.
        output_artifact: Generated code/docs stored here (optional).
        state: Current lifecycle state (PENDING, RUNNING, COMPLETED, STALE).
        parent_node_id: For section hierarchy — None for top-level nodes.
        page_start: Starting page number (section-level parsing).
        page_end: Ending page number (section-level parsing).
    """

    id: str
    source_type: str
    source_path: str
    raw_content: str
    summary: str
    system_role_prompt: str
    output_artifact: Optional[str]
    state: NodeState
    parent_node_id: Optional[str] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None


@dataclass
class Edge:
    """A directed relationship between two nodes.

    Attributes:
        id: Auto-incremented integer identifier.
        upstream_id: Foreign key to the dependency node.
        downstream_id: Foreign key to the dependent node.
        relationship_type: Semantic label (e.g. 'defines_logic_for', 'contains_section').
    """

    id: int
    upstream_id: str
    downstream_id: str
    relationship_type: str


@dataclass
class SessionConfig:
    """Configuration for a CogniLink session.

    Attributes:
        model: Any LangChain BaseChatModel instance (ChatOpenAI, ChatBedrock, etc.)
               The user passes their own configured model.
        workspace_path: Root path of the workspace being managed.
        persist: If True, use file-based SQLite; if False, use in-memory.
        db_path: Path to the SQLite database file (required when persist=True).
        auto_cascade: If True, automatically propagate STALE on mutations.
        section_mode: If True, parse PDFs into hierarchical section nodes.
        default_prompt: System prompt used for summarization.
    """

    model: "BaseChatModel"  # Any LangChain BaseChatModel instance
    workspace_path: Path
    persist: bool = False
    db_path: Optional[Path] = None
    auto_cascade: bool = True
    section_mode: bool = False
    default_prompt: str = "Summarize the following document concisely in 2-3 sentences, capturing the key points."


@dataclass
class IngestionResult:
    """Result of a document ingestion operation.

    Attributes:
        node_ids: List of node IDs created during ingestion.
        edges_created: Number of edges created.
        status: Status string (e.g. 'OK').
    """

    node_ids: List[str]
    edges_created: int
    status: str


@dataclass
class ContextPayload:
    """Bounded context window returned by the context retrieval API.

    Attributes:
        target_node: The target node with full content.
        parent_summaries: Summaries of direct upstream parent nodes.
        relationship_chain: Edges connecting parents to the target.
    """

    target_node: Node
    parent_summaries: List[Dict[str, str]]
    relationship_chain: List[Edge]


@dataclass
class CascadeReport:
    """Result of a mutation cascade propagation.

    Attributes:
        source_node_id: The node that was mutated.
        stale_node_ids: All downstream nodes marked STALE.
        depth_reached: Maximum BFS traversal depth.
    """

    source_node_id: str
    stale_node_ids: List[str]
    depth_reached: int
