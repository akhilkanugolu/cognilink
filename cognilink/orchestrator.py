"""CogniLink Session Orchestrator — top-level coordinator.

Wires together ExtractorRegistry, GraphStore, LangChain model,
MutationCascader, and Visualizer into a unified API.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from cognilink.core.cascader import MutationCascader
from cognilink.core.exceptions import (
    LLMUnavailableError,
    NodeNotFoundError,
    VisualizationWriteError,
)
from cognilink.core.graph_store import GraphStore
from cognilink.core.models import (
    CascadeReport,
    ContextPayload,
    Edge,
    IngestionResult,
    Node,
    NodeState,
    SessionConfig,
)
from cognilink.extract.pdf import pdf_parser
from cognilink.extract.registry import ExtractorRegistry
from cognilink.inference.provider import invoke_llm
from cognilink.viz.html_renderer import Visualizer

logger = logging.getLogger(__name__)


class SessionOrchestrator:
    """Top-level coordinator wiring all CogniLink components.

    Args:
        config: SessionConfig with model, workspace_path, persist settings, etc.
    """

    def __init__(self, config: SessionConfig) -> None:
        self.config = config
        self.model = config.model
        self.graph_store = GraphStore(persist=config.persist, db_path=config.db_path)
        self.cascader = MutationCascader(self.graph_store)
        self.extractor = ExtractorRegistry()
        self.extractor.register_parser(".pdf", pdf_parser)
        self.visualizer = Visualizer(
            output_dir=config.workspace_path,
            graph_store=self.graph_store,
        )

    def ingest_documents(
        self,
        paths: List[str],
        relationships: Optional[List[Dict[str, str]]] = None,
    ) -> IngestionResult:
        """Full ingestion pipeline: extract → store → summarize → link → render.

        Args:
            paths: List of file paths to ingest.
            relationships: Optional list of edge dicts with keys
                          "upstream", "downstream", "type".

        Returns:
            IngestionResult with node_ids, edges_created, and status.
        """
        raw_blocks = self.extractor.parse_workspace_concurrently(paths)

        node_ids: List[str] = []
        for block in raw_blocks:
            node_id = block["id"]

            # Handle re-ingestion as update
            existing = self.graph_store.get_node(node_id)
            if existing is not None:
                self.update_node(node_id, block["raw_text"])
                node_ids.append(node_id)
                continue

            node = Node(
                id=node_id,
                source_type=self._infer_source_type(block["ext"]),
                source_path=block["path"],
                raw_content=block["raw_text"],
                summary="",
                system_role_prompt=self.config.default_prompt,
                output_artifact=None,
                state=NodeState.PENDING,
            )
            self.graph_store.insert_node(node)

            # Summarize via LangChain model
            try:
                self.graph_store.set_node_state(node_id, NodeState.RUNNING)
                summary = invoke_llm(
                    model=self.model,
                    system_prompt=self.config.default_prompt,
                    user_prompt=node.raw_content[:4000],
                )
                self.graph_store.update_node(node_id, summary=summary)
                self.graph_store.set_node_state(node_id, NodeState.COMPLETED)
            except Exception as e:
                # LLM unavailable — node stays in RUNNING state, mark summary as empty
                self.graph_store.update_node(node_id, summary="(summarization pending - LLM unavailable)")
                logger.warning("LLM unavailable for node '%s': %s", node_id, e)

            node_ids.append(node_id)

        # Create edges
        edges_created = 0
        if relationships:
            for rel in relationships:
                self.graph_store.insert_edge(
                    upstream_id=rel["upstream"],
                    downstream_id=rel["downstream"],
                    relationship_type=rel["type"],
                )
                edges_created += 1

        # Render visualization
        try:
            self.visualizer.render()
        except Exception as e:
            logger.warning("Visualization render failed: %s", e)

        return IngestionResult(node_ids=node_ids, edges_created=edges_created, status="OK")

    def update_node(self, node_id: str, new_content: str) -> CascadeReport:
        """Update node content, regenerate summary, trigger cascade.

        Args:
            node_id: The ID of the node to update.
            new_content: New raw content for the node.

        Returns:
            CascadeReport with affected downstream nodes.

        Raises:
            NodeNotFoundError: If the node doesn't exist.
        """
        node = self.graph_store.get_node(node_id)
        if node is None:
            raise NodeNotFoundError(node_id)

        # Save previous state for rollback
        prev_content = node.raw_content
        prev_summary = node.summary
        prev_state = node.state

        try:
            self.graph_store.update_node(node_id, raw_content=new_content)
            self.graph_store.set_node_state(node_id, NodeState.RUNNING)
            summary = invoke_llm(
                model=self.model,
                system_prompt=self.config.default_prompt,
                user_prompt=new_content[:4000],
            )
            self.graph_store.update_node(node_id, summary=summary)
            self.graph_store.set_node_state(node_id, NodeState.COMPLETED)
        except Exception:
            # Rollback on failure
            self.graph_store.update_node(
                node_id, raw_content=prev_content, summary=prev_summary, state=prev_state.value
            )
            raise

        # Cascade
        if self.config.auto_cascade:
            cascade = self.cascader.propagate_stale(node_id)
        else:
            cascade = CascadeReport(source_node_id=node_id, stale_node_ids=[], depth_reached=0)

        try:
            self.visualizer.render()
        except Exception as e:
            logger.warning("Visualization render failed: %s", e)

        return cascade

    def add_edge(self, upstream_id: str, downstream_id: str, relationship_type: str) -> Edge:
        """Add a directed edge between two nodes.

        Args:
            upstream_id: The upstream (dependency) node ID.
            downstream_id: The downstream (dependent) node ID.
            relationship_type: Semantic label for the relationship.

        Returns:
            The created Edge object.
        """
        return self.graph_store.insert_edge(upstream_id, downstream_id, relationship_type)

    def get_context(self, target_node_id: str) -> ContextPayload:
        """Assemble bounded context window for agent consumption.

        Args:
            target_node_id: The node to get context for.

        Returns:
            ContextPayload with target node, parent summaries, and edges.

        Raises:
            NodeNotFoundError: If the target node doesn't exist.
        """
        target_node = self.graph_store.get_node(target_node_id)
        if target_node is None:
            raise NodeNotFoundError(target_node_id)

        upstream_edges = self.graph_store.get_upstream(target_node_id)
        parent_summaries: List[Dict[str, str]] = []

        for edge in upstream_edges:
            parent_node = self.graph_store.get_node(edge.upstream_id)
            if parent_node is not None:
                parent_summaries.append({
                    "node_id": parent_node.id,
                    "summary": parent_node.summary,
                    "relationship": edge.relationship_type,
                })

        return ContextPayload(
            target_node=target_node,
            parent_summaries=parent_summaries,
            relationship_chain=upstream_edges,
        )

    def regenerate_stale(self) -> List[str]:
        """Regenerate all STALE nodes in topological order.

        Returns:
            List of node IDs that were regenerated, in order.
        """
        all_nodes = self.graph_store.get_all_nodes()
        stale_ids = [n.id for n in all_nodes if n.state == NodeState.STALE]

        if not stale_ids:
            return []

        order = self.cascader.get_regeneration_order(stale_ids)
        for node_id in order:
            self.cascader.regenerate_node(node_id, self.model)

        try:
            self.visualizer.render()
        except Exception as e:
            logger.warning("Visualization render failed: %s", e)

        return order

    def render_visualization(self) -> Dict[str, Path]:
        """Render HTML and JSON visualization.

        Returns:
            Dict with "html" and "json" keys pointing to output file paths.

        Raises:
            VisualizationWriteError: If the output directory is not writable.
        """
        try:
            return self.visualizer.render()
        except Exception as e:
            raise VisualizationWriteError(str(self.config.workspace_path), str(e)) from e

    def get_graph_state(self) -> Dict[str, Any]:
        """Return current graph state as a dict."""
        return {
            "nodes": self.graph_store.get_all_nodes(),
            "edges": self.graph_store.get_all_edges(),
        }

    def close(self) -> None:
        """Close database connection and release resources."""
        self.graph_store.close()

    @staticmethod
    def _infer_source_type(ext: str) -> str:
        mapping = {".pdf": "PDF", ".txt": "TEXT", ".md": "TEXT", ".xlsx": "EXCEL", ".xls": "EXCEL"}
        return mapping.get(ext, "CUSTOM")
