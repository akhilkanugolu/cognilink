"""Mutation cascade propagation for CogniLink."""
from __future__ import annotations

from typing import Any, Dict, List, Set

from cognilink.core.exceptions import CycleDetectedError
from cognilink.core.models import CascadeReport, NodeState

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cognilink.core.graph_store import GraphStore


class MutationCascader:
    """Propagates STALE state downstream when a node is mutated."""

    def __init__(self, graph_store: "GraphStore") -> None:
        self._graph_store = graph_store

    def propagate_stale(self, mutated_node_id: str) -> CascadeReport:
        """BFS traversal marking all downstream nodes STALE.

        Args:
            mutated_node_id: The node that was just mutated.

        Returns:
            CascadeReport with source_node_id, stale_node_ids, and depth_reached.
        """
        visited: Set[str] = set()
        queue: List[str] = [mutated_node_id]
        stale_ids: List[str] = []
        depth = 0

        while queue:
            next_level: List[str] = []
            for current_id in queue:
                if current_id in visited:
                    continue
                visited.add(current_id)

                downstream_edges = self._graph_store.get_downstream(current_id)
                for edge in downstream_edges:
                    child_id = edge.downstream_id
                    if child_id not in visited:
                        self._graph_store.set_node_state(child_id, NodeState.STALE)
                        stale_ids.append(child_id)
                        next_level.append(child_id)

            queue = next_level
            if next_level:
                depth += 1

        return CascadeReport(
            source_node_id=mutated_node_id,
            stale_node_ids=stale_ids,
            depth_reached=depth,
        )

    def get_regeneration_order(self, stale_ids: List[str]) -> List[str]:
        """Kahn's algorithm for topological sort of stale nodes.

        Args:
            stale_ids: List of node IDs that are currently STALE.

        Returns:
            Topologically sorted list where for any edge (A→B), A appears before B.

        Raises:
            CycleDetectedError: If a cycle is detected among the stale nodes.
        """
        stale_set = set(stale_ids)
        in_degree: Dict[str, int] = {nid: 0 for nid in stale_ids}

        for nid in stale_ids:
            upstream_edges = self._graph_store.get_upstream(nid)
            for edge in upstream_edges:
                if edge.upstream_id in stale_set:
                    in_degree[nid] += 1

        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        result: List[str] = []

        while queue:
            current = queue.pop(0)
            result.append(current)

            downstream_edges = self._graph_store.get_downstream(current)
            for edge in downstream_edges:
                if edge.downstream_id in stale_set:
                    in_degree[edge.downstream_id] -= 1
                    if in_degree[edge.downstream_id] == 0:
                        queue.append(edge.downstream_id)

        if len(result) != len(stale_ids):
            unresolved = [nid for nid in stale_ids if nid not in result]
            raise CycleDetectedError(unresolved)

        return result

    def regenerate_node(self, node_id: str, model: Any) -> None:
        """Regenerate a single stale node's summary using a LangChain model.

        Args:
            node_id: The ID of the node to regenerate.
            model: A LangChain BaseChatModel instance.
        """
        from cognilink.inference.provider import invoke_llm

        node = self._graph_store.get_node(node_id)
        if node is None:
            return

        self._graph_store.set_node_state(node_id, NodeState.RUNNING)
        summary = invoke_llm(
            model=model,
            system_prompt="Summarize the following document concisely in 2-3 sentences.",
            user_prompt=node.raw_content[:4000],
        )
        self._graph_store.update_node(node_id, summary=summary)
        self._graph_store.set_node_state(node_id, NodeState.COMPLETED)
