"""JSON export for CogniLink graph state.

Provides export_json() as a standalone function for programmatic consumption.
The Visualizer class in html_renderer.py also provides this functionality.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cognilink.core.graph_store import GraphStore


def export_json(graph_store: "GraphStore", output_dir: Path) -> Path:
    """Export graph state as a JSON file.

    Only includes summaries, never raw_content, to limit data leakage.
    Uses atomic writes (temp file + rename) to prevent partial corruption.

    Args:
        graph_store: The GraphStore instance to export from.
        output_dir: Directory to write the JSON file to.

    Returns:
        Path to the written JSON file.
    """
    nodes = graph_store.get_all_nodes()
    edges = graph_store.get_all_edges()

    graph_data = {
        "nodes": [
            {
                "id": node.id,
                "source_type": node.source_type,
                "state": node.state.value,
                "summary": node.summary,
                "source_path": node.source_path,
            }
            for node in nodes
        ],
        "edges": [
            {
                "id": edge.id,
                "upstream_id": edge.upstream_id,
                "downstream_id": edge.downstream_id,
                "relationship_type": edge.relationship_type,
            }
            for edge in edges
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "cognilink_graph.json"
    temp_path = output_path.with_suffix(".tmp")
    temp_path.write_text(json.dumps(graph_data, indent=2), encoding="utf-8")
    temp_path.rename(output_path)
    return output_path
