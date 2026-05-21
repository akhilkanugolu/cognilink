"""CogniLink — Document-to-graph knowledge mapper powered by LangChain.

Simple API:
    import cognilink
    from langchain_openai import ChatOpenAI

    result = cognilink.map("./doc.pdf", model=ChatOpenAI(api_key="sk-..."))
    # Returns: {"nodes": [...], "edges": [...], "html_path": "...", "json_path": "..."}
"""

__version__ = "0.1.0"

from cognilink.core.models import (
    CascadeReport,
    ContextPayload,
    Edge,
    IngestionResult,
    Node,
    NodeState,
    SessionConfig,
)
from cognilink.orchestrator import SessionOrchestrator

from pathlib import Path
from typing import Any, Dict, List, Optional, Union


def map(
    path: Union[str, List[str]],
    model: Any = None,
    prompt: Optional[str] = None,
    persist: bool = False,
    relationships: Optional[List[Dict[str, str]]] = None,
    output_dir: Optional[str] = None,
) -> dict:
    """Simple entry point: give a document path, get a graph map back.

    Args:
        path: A file path or list of file paths to ingest.
        model: A LangChain BaseChatModel instance (ChatOpenAI, ChatAnthropic, etc.)
        prompt: Optional custom summarization prompt.
        persist: If True, persist the graph to disk.
        relationships: Optional list of edge dicts with keys "upstream", "downstream", "type".
        output_dir: Directory for output files (default: current directory).

    Returns:
        Dict with keys:
            - "nodes": list of node dicts (id, source_type, state, summary, source_path)
            - "edges": list of edge dicts (id, upstream_id, downstream_id, relationship_type)
            - "html_path": path to the generated HTML visualization
            - "json_path": path to the generated JSON export

    Example:
        >>> import cognilink
        >>> from langchain_openai import ChatOpenAI
        >>> result = cognilink.map("./doc.pdf", model=ChatOpenAI(api_key="sk-..."))
    """
    if model is None:
        raise ValueError(
            "A LangChain model is required. Example:\n"
            "  from langchain_openai import ChatOpenAI\n"
            "  cognilink.map('doc.pdf', model=ChatOpenAI(api_key='sk-...'))"
        )

    # Normalize path to list
    paths: List[str] = [path] if isinstance(path, str) else list(path)

    workspace = Path(output_dir) if output_dir else Path(".")
    db_path = workspace / "cognilink.db" if persist else None

    config = SessionConfig(
        model=model,
        workspace_path=workspace,
        persist=persist,
        db_path=db_path,
        default_prompt=prompt or SessionConfig.default_prompt,
    )

    orchestrator = SessionOrchestrator(config)

    try:
        orchestrator.ingest_documents(paths, relationships=relationships)
        outputs = orchestrator.render_visualization()

        # Build response
        all_nodes = orchestrator.graph_store.get_all_nodes()
        all_edges = orchestrator.graph_store.get_all_edges()

        return {
            "nodes": [
                {
                    "id": n.id,
                    "source_type": n.source_type,
                    "state": n.state.value,
                    "summary": n.summary,
                    "source_path": n.source_path,
                }
                for n in all_nodes
            ],
            "edges": [
                {
                    "id": e.id,
                    "upstream_id": e.upstream_id,
                    "downstream_id": e.downstream_id,
                    "relationship_type": e.relationship_type,
                }
                for e in all_edges
            ],
            "html_path": str(outputs["html"]),
            "json_path": str(outputs["json"]),
        }
    finally:
        orchestrator.close()


__all__ = [
    "map",
    "SessionOrchestrator",
    "SessionConfig",
    "Node",
    "Edge",
    "NodeState",
    "IngestionResult",
    "ContextPayload",
    "CascadeReport",
]
