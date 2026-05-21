"""Unit tests for MutationCascader.

Tests cover:
- Linear chain propagation
- Branching graph propagation
- Diamond dependency propagation
- Cycle detection (terminates without infinite loop)
- Topological sort with valid DAG
- CycleDetectedError on cyclic stale subgraph
- Cascade isolation (unrelated nodes unaffected)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from cognilink.core.cascader import MutationCascader
from cognilink.core.exceptions import CycleDetectedError
from cognilink.core.graph_store import GraphStore
from cognilink.core.models import Node, NodeState


# --- Helpers ---


def _make_node(node_id: str, state: NodeState = NodeState.COMPLETED) -> Node:
    """Create a minimal Node for testing."""
    return Node(
        id=node_id,
        source_type="TEXT",
        source_path=f"/tmp/{node_id}.txt",
        raw_content=f"Content of {node_id}",
        summary=f"Summary of {node_id}",
        system_role_prompt="You are a summarizer.",
        output_artifact=None,
        state=state,
    )


def _build_graph(nodes: list[str], edges: list[tuple[str, str]]) -> GraphStore:
    """Build an in-memory GraphStore with given nodes and edges."""
    store = GraphStore(persist=False)
    for nid in nodes:
        store.insert_node(_make_node(nid))
    for upstream, downstream in edges:
        store.insert_edge(upstream, downstream, "depends_on")
    return store


class FakeLangChainModel:
    """Fake LangChain model for testing regenerate_node.

    Mimics the BaseChatModel.invoke() interface.
    """

    def __init__(self, response_text: str = "Regenerated summary") -> None:
        self._response_text = response_text
        self.call_count = 0

    def invoke(self, messages: List[Any], **kwargs: Any) -> Any:
        self.call_count += 1
        # Return a mock AIMessage
        mock_response = MagicMock()
        mock_response.content = self._response_text
        return mock_response


# --- Tests: propagate_stale ---


class TestPropagateStale:
    """Tests for MutationCascader.propagate_stale."""

    def test_linear_chain(self) -> None:
        """A→B→C: mutate A, B and C become STALE."""
        store = _build_graph(["A", "B", "C"], [("A", "B"), ("B", "C")])
        cascader = MutationCascader(store)

        report = cascader.propagate_stale("A")

        assert set(report.stale_node_ids) == {"B", "C"}
        assert report.source_node_id == "A"
        assert report.depth_reached == 2

        # Verify actual node states
        assert store.get_node("A").state == NodeState.COMPLETED  # Not marked stale
        assert store.get_node("B").state == NodeState.STALE
        assert store.get_node("C").state == NodeState.STALE

    def test_branching_graph(self) -> None:
        """A→B, A→C: mutate A, both B and C become STALE."""
        store = _build_graph(["A", "B", "C"], [("A", "B"), ("A", "C")])
        cascader = MutationCascader(store)

        report = cascader.propagate_stale("A")

        assert set(report.stale_node_ids) == {"B", "C"}
        assert report.depth_reached == 1

    def test_diamond_dependency(self) -> None:
        """A→B, A→C, B→D, C→D: mutate A, B, C, D all become STALE."""
        store = _build_graph(
            ["A", "B", "C", "D"],
            [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")],
        )
        cascader = MutationCascader(store)

        report = cascader.propagate_stale("A")

        assert set(report.stale_node_ids) == {"B", "C", "D"}
        assert report.depth_reached == 2

        assert store.get_node("A").state == NodeState.COMPLETED
        assert store.get_node("D").state == NodeState.STALE

    def test_cycle_detection_terminates(self) -> None:
        """A→B→C→A (cycle): mutate A, terminates without infinite loop."""
        store = GraphStore(persist=False)
        store.insert_node(_make_node("A"))
        store.insert_node(_make_node("B"))
        store.insert_node(_make_node("C"))
        store.insert_edge("A", "B", "depends_on")
        store.insert_edge("B", "C", "depends_on")
        store.insert_edge("C", "A", "depends_on")

        cascader = MutationCascader(store)

        # Should terminate without hanging
        report = cascader.propagate_stale("A")

        # B and C should be marked STALE; the cycle back to A is caught by visited set
        assert "B" in report.stale_node_ids
        assert "C" in report.stale_node_ids
        # A itself should NOT be in stale_ids (it's the source)
        assert "A" not in report.stale_node_ids

    def test_cascade_isolation(self) -> None:
        """Nodes not downstream of the mutated node are unaffected."""
        store = _build_graph(["A", "B", "C", "D"], [("A", "B"), ("C", "D")])
        cascader = MutationCascader(store)

        report = cascader.propagate_stale("A")

        assert set(report.stale_node_ids) == {"B"}
        assert store.get_node("C").state == NodeState.COMPLETED
        assert store.get_node("D").state == NodeState.COMPLETED

    def test_mutated_node_not_marked_stale(self) -> None:
        """The mutated node itself should NOT be marked STALE."""
        store = _build_graph(["A", "B"], [("A", "B")])
        cascader = MutationCascader(store)

        report = cascader.propagate_stale("A")

        assert store.get_node("A").state == NodeState.COMPLETED
        assert "A" not in report.stale_node_ids


# --- Tests: get_regeneration_order ---


class TestGetRegenerationOrder:
    """Tests for MutationCascader.get_regeneration_order."""

    def test_topological_sort_valid_dag(self) -> None:
        """Linear DAG: A→B→C. Order should be [A, B, C] or respect dependencies."""
        store = _build_graph(["A", "B", "C"], [("A", "B"), ("B", "C")])
        store.set_node_state("A", NodeState.STALE)
        store.set_node_state("B", NodeState.STALE)
        store.set_node_state("C", NodeState.STALE)

        cascader = MutationCascader(store)
        order = cascader.get_regeneration_order(["A", "B", "C"])

        assert order.index("A") < order.index("B")
        assert order.index("B") < order.index("C")

    def test_topological_sort_diamond(self) -> None:
        """Diamond: A→B, A→C, B→D, C→D. A before B and C; B and C before D."""
        store = _build_graph(
            ["A", "B", "C", "D"],
            [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")],
        )
        for nid in ["A", "B", "C", "D"]:
            store.set_node_state(nid, NodeState.STALE)

        cascader = MutationCascader(store)
        order = cascader.get_regeneration_order(["A", "B", "C", "D"])

        assert order.index("A") < order.index("B")
        assert order.index("A") < order.index("C")
        assert order.index("B") < order.index("D")
        assert order.index("C") < order.index("D")

    def test_cycle_detected_error(self) -> None:
        """Cycle among stale nodes raises CycleDetectedError."""
        store = GraphStore(persist=False)
        store.insert_node(_make_node("X"))
        store.insert_node(_make_node("Y"))
        store.insert_node(_make_node("Z"))
        store.insert_edge("X", "Y", "depends_on")
        store.insert_edge("Y", "Z", "depends_on")
        store.insert_edge("Z", "X", "depends_on")

        store.set_node_state("X", NodeState.STALE)
        store.set_node_state("Y", NodeState.STALE)
        store.set_node_state("Z", NodeState.STALE)

        cascader = MutationCascader(store)

        with pytest.raises(CycleDetectedError) as exc_info:
            cascader.get_regeneration_order(["X", "Y", "Z"])

        assert set(exc_info.value.involved_nodes) == {"X", "Y", "Z"}

    def test_partial_stale_subgraph(self) -> None:
        """Only stale nodes are considered; non-stale upstream is ignored."""
        store = _build_graph(["A", "B", "C"], [("A", "B"), ("B", "C")])
        store.set_node_state("B", NodeState.STALE)
        store.set_node_state("C", NodeState.STALE)

        cascader = MutationCascader(store)
        order = cascader.get_regeneration_order(["B", "C"])

        assert order == ["B", "C"]


# --- Tests: regenerate_node ---


class TestRegenerateNode:
    """Tests for MutationCascader.regenerate_node."""

    def test_regenerate_updates_summary(self) -> None:
        """regenerate_node should call LLM and update the node's summary."""
        store = _build_graph(["A"], [])
        store.set_node_state("A", NodeState.STALE)

        cascader = MutationCascader(store)
        fake_model = FakeLangChainModel(response_text="New summary for A")

        cascader.regenerate_node("A", fake_model)

        node = store.get_node("A")
        assert node.summary == "New summary for A"
        assert node.state == NodeState.COMPLETED
        assert fake_model.call_count == 1

    def test_regenerate_transitions_through_running(self) -> None:
        """regenerate_node should transition STALE → RUNNING → COMPLETED."""
        store = _build_graph(["B"], [])
        store.set_node_state("B", NodeState.STALE)

        states_observed: list[NodeState] = []

        original_set_state = store.set_node_state

        def tracking_set_state(node_id: str, state: NodeState) -> None:
            states_observed.append(state)
            original_set_state(node_id, state)

        store.set_node_state = tracking_set_state  # type: ignore[assignment]

        cascader = MutationCascader(store)
        fake_model = FakeLangChainModel()

        cascader.regenerate_node("B", fake_model)

        assert NodeState.RUNNING in states_observed
        assert NodeState.COMPLETED in states_observed
        assert states_observed.index(NodeState.RUNNING) < states_observed.index(
            NodeState.COMPLETED
        )
