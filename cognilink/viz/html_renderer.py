"""HTML visualization renderer for CogniLink."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, TYPE_CHECKING

from jinja2 import Environment, PackageLoader, select_autoescape

from cognilink.core.models import NodeState

if TYPE_CHECKING:
    from cognilink.core.graph_store import GraphStore


_STATE_COLORS = {
    NodeState.COMPLETED: "#4CAF50",
    NodeState.RUNNING: "#FFC107",
    NodeState.STALE: "#F44336",
    NodeState.PENDING: "#9E9E9E",
}


class Visualizer:
    """Renders graph state as interactive HTML and JSON exports."""

    def __init__(self, output_dir: Path, graph_store: "GraphStore") -> None:
        self._output_dir = output_dir
        self._graph_store = graph_store
        self._env = Environment(
            loader=PackageLoader("cognilink.viz", "templates"),
            autoescape=select_autoescape(["html"]),
        )

    def render_html(self) -> Path:
        """Render interactive HTML visualization."""
        nodes = self._graph_store.get_all_nodes()
        edges = self._graph_store.get_all_edges()

        vis_nodes: List[Dict[str, Any]] = []
        node_details: Dict[str, Dict[str, str]] = {}
        for node in nodes:
            vis_nodes.append({
                "id": node.id,
                "label": node.id,
                "color": _STATE_COLORS.get(node.state, "#9E9E9E"),
            })
            node_details[node.id] = {
                "state": node.state.value,
                "source": node.source_path,
                "summary": node.summary,
            }

        vis_edges: List[Dict[str, Any]] = []
        for edge in edges:
            vis_edges.append({
                "from": edge.upstream_id,
                "to": edge.downstream_id,
                "label": edge.relationship_type,
            })

        template = self._env.get_template("graph.html")
        html_content = template.render(
            nodes=vis_nodes,
            edges=vis_edges,
            node_details=node_details,
        )

        self._output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self._output_dir / "cognilink_graph.html"
        temp_path = output_path.with_suffix(".tmp")
        temp_path.write_text(html_content, encoding="utf-8")
        temp_path.rename(output_path)
        return output_path

    def export_json(self) -> Path:
        """Export graph state as JSON."""
        nodes = self._graph_store.get_all_nodes()
        edges = self._graph_store.get_all_edges()

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

        self._output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self._output_dir / "cognilink_graph.json"
        temp_path = output_path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(graph_data, indent=2), encoding="utf-8")
        temp_path.rename(output_path)
        return output_path

    def render(self) -> Dict[str, Path]:
        """Render both HTML and JSON outputs."""
        return {
            "html": self.render_html(),
            "json": self.export_json(),
        }
