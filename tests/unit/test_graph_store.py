"""Unit tests for GraphStore.

Tests CRUD operations, state transitions, dual-mode storage,
transaction safety, and edge integrity constraints.
"""

import os
import tempfile
from pathlib import Path

import pytest

from cognilink.core.exceptions import (
    DatabaseCorruptionError,
    InvalidStateTransitionError,
    NodeAlreadyExistsError,
    NodeNotFoundError,
)
from cognilink.core.graph_store import GraphStore
from cognilink.core.models import Edge, Node, NodeState


def _make_node(
    node_id: str = "NODE_TEST",
    state: NodeState = NodeState.PENDING,
    parent_node_id: str | None = None,
    page_start: int | None = None,
    page_end: int | None = None,
) -> Node:
    """Helper to create a test node."""
    return Node(
        id=node_id,
        source_type="TEXT",
        source_path="/tmp/test.txt",
        raw_content="Hello world",
        summary="A test node",
        system_role_prompt="You are a helpful assistant.",
        output_artifact=None,
        state=state,
        parent_node_id=parent_node_id,
        page_start=page_start,
        page_end=page_end,
    )


@pytest.fixture
def store() -> GraphStore:
    """Create an in-memory GraphStore for testing."""
    gs = GraphStore(persist=False)
    yield gs
    gs.close()


@pytest.fixture
def persistent_store(tmp_path: Path) -> GraphStore:
    """Create a persistent GraphStore for testing."""
    db_path = tmp_path / "test.db"
    gs = GraphStore(persist=True, db_path=db_path)
    yield gs
    gs.close()


class TestNodeCRUD:
    """Tests for node create, read, update, delete operations."""

    def test_insert_and_get_node(self, store: GraphStore) -> None:
        node = _make_node()
        store.insert_node(node)
        retrieved = store.get_node("NODE_TEST")
        assert retrieved is not None
        assert retrieved.id == "NODE_TEST"
        assert retrieved.source_type == "TEXT"
        assert retrieved.raw_content == "Hello world"
        assert retrieved.state == NodeState.PENDING

    def test_insert_node_with_optional_fields(self, store: GraphStore) -> None:
        node = _make_node(
            parent_node_id="NODE_PARENT",
            page_start=1,
            page_end=5,
        )
        store.insert_node(node)
        retrieved = store.get_node("NODE_TEST")
        assert retrieved is not None
        assert retrieved.parent_node_id == "NODE_PARENT"
        assert retrieved.page_start == 1
        assert retrieved.page_end == 5

    def test_insert_duplicate_node_raises(self, store: GraphStore) -> None:
        node = _make_node()
        store.insert_node(node)
        with pytest.raises(NodeAlreadyExistsError) as exc_info:
            store.insert_node(node)
        assert exc_info.value.node_id == "NODE_TEST"

    def test_get_nonexistent_node_returns_none(self, store: GraphStore) -> None:
        assert store.get_node("NODE_MISSING") is None

    def test_update_node(self, store: GraphStore) -> None:
        node = _make_node()
        store.insert_node(node)
        store.update_node("NODE_TEST", summary="Updated summary")
        retrieved = store.get_node("NODE_TEST")
        assert retrieved is not None
        assert retrieved.summary == "Updated summary"

    def test_update_nonexistent_node_raises(self, store: GraphStore) -> None:
        with pytest.raises(NodeNotFoundError):
            store.update_node("NODE_MISSING", summary="x")

    def test_delete_node(self, store: GraphStore) -> None:
        node = _make_node()
        store.insert_node(node)
        store.delete_node("NODE_TEST")
        assert store.get_node("NODE_TEST") is None

    def test_delete_nonexistent_node_raises(self, store: GraphStore) -> None:
        with pytest.raises(NodeNotFoundError):
            store.delete_node("NODE_MISSING")

    def test_delete_node_removes_associated_edges(self, store: GraphStore) -> None:
        node_a = _make_node("NODE_A")
        node_b = _make_node("NODE_B")
        store.insert_node(node_a)
        store.insert_node(node_b)
        store.insert_edge("NODE_A", "NODE_B", "depends_on")
        store.delete_node("NODE_A")
        assert store.get_all_edges() == []

    def test_get_all_nodes(self, store: GraphStore) -> None:
        store.insert_node(_make_node("NODE_A"))
        store.insert_node(_make_node("NODE_B"))
        nodes = store.get_all_nodes()
        assert len(nodes) == 2
        ids = {n.id for n in nodes}
        assert ids == {"NODE_A", "NODE_B"}


class TestEdgeCRUD:
    """Tests for edge create, read, delete operations."""

    def test_insert_and_get_edge(self, store: GraphStore) -> None:
        store.insert_node(_make_node("NODE_A"))
        store.insert_node(_make_node("NODE_B"))
        edge = store.insert_edge("NODE_A", "NODE_B", "depends_on")
        assert edge.upstream_id == "NODE_A"
        assert edge.downstream_id == "NODE_B"
        assert edge.relationship_type == "depends_on"
        assert isinstance(edge.id, int)

    def test_insert_edge_nonexistent_upstream_raises(self, store: GraphStore) -> None:
        store.insert_node(_make_node("NODE_B"))
        with pytest.raises(NodeNotFoundError) as exc_info:
            store.insert_edge("NODE_MISSING", "NODE_B", "depends_on")
        assert exc_info.value.node_id == "NODE_MISSING"

    def test_insert_edge_nonexistent_downstream_raises(self, store: GraphStore) -> None:
        store.insert_node(_make_node("NODE_A"))
        with pytest.raises(NodeNotFoundError) as exc_info:
            store.insert_edge("NODE_A", "NODE_MISSING", "depends_on")
        assert exc_info.value.node_id == "NODE_MISSING"

    def test_insert_self_edge_raises(self, store: GraphStore) -> None:
        store.insert_node(_make_node("NODE_A"))
        with pytest.raises(ValueError, match="Self-edges are not allowed"):
            store.insert_edge("NODE_A", "NODE_A", "depends_on")

    def test_insert_duplicate_edge_raises(self, store: GraphStore) -> None:
        store.insert_node(_make_node("NODE_A"))
        store.insert_node(_make_node("NODE_B"))
        store.insert_edge("NODE_A", "NODE_B", "depends_on")
        with pytest.raises(ValueError, match="Duplicate edge"):
            store.insert_edge("NODE_A", "NODE_B", "depends_on")

    def test_same_nodes_different_type_allowed(self, store: GraphStore) -> None:
        store.insert_node(_make_node("NODE_A"))
        store.insert_node(_make_node("NODE_B"))
        store.insert_edge("NODE_A", "NODE_B", "depends_on")
        edge2 = store.insert_edge("NODE_A", "NODE_B", "contains_section")
        assert edge2.relationship_type == "contains_section"

    def test_get_downstream(self, store: GraphStore) -> None:
        store.insert_node(_make_node("NODE_A"))
        store.insert_node(_make_node("NODE_B"))
        store.insert_node(_make_node("NODE_C"))
        store.insert_edge("NODE_A", "NODE_B", "depends_on")
        store.insert_edge("NODE_A", "NODE_C", "defines_logic_for")
        downstream = store.get_downstream("NODE_A")
        assert len(downstream) == 2
        downstream_ids = {e.downstream_id for e in downstream}
        assert downstream_ids == {"NODE_B", "NODE_C"}

    def test_get_upstream(self, store: GraphStore) -> None:
        store.insert_node(_make_node("NODE_A"))
        store.insert_node(_make_node("NODE_B"))
        store.insert_node(_make_node("NODE_C"))
        store.insert_edge("NODE_A", "NODE_C", "depends_on")
        store.insert_edge("NODE_B", "NODE_C", "defines_logic_for")
        upstream = store.get_upstream("NODE_C")
        assert len(upstream) == 2
        upstream_ids = {e.upstream_id for e in upstream}
        assert upstream_ids == {"NODE_A", "NODE_B"}

    def test_delete_edge(self, store: GraphStore) -> None:
        store.insert_node(_make_node("NODE_A"))
        store.insert_node(_make_node("NODE_B"))
        edge = store.insert_edge("NODE_A", "NODE_B", "depends_on")
        store.delete_edge(edge.id)
        assert store.get_all_edges() == []

    def test_get_all_edges(self, store: GraphStore) -> None:
        store.insert_node(_make_node("NODE_A"))
        store.insert_node(_make_node("NODE_B"))
        store.insert_node(_make_node("NODE_C"))
        store.insert_edge("NODE_A", "NODE_B", "depends_on")
        store.insert_edge("NODE_B", "NODE_C", "defines_logic_for")
        edges = store.get_all_edges()
        assert len(edges) == 2


class TestStateTransitions:
    """Tests for node state machine enforcement."""

    def test_pending_to_running(self, store: GraphStore) -> None:
        store.insert_node(_make_node("NODE_A", state=NodeState.PENDING))
        store.set_node_state("NODE_A", NodeState.RUNNING)
        node = store.get_node("NODE_A")
        assert node is not None
        assert node.state == NodeState.RUNNING

    def test_running_to_completed(self, store: GraphStore) -> None:
        store.insert_node(_make_node("NODE_A", state=NodeState.RUNNING))
        store.set_node_state("NODE_A", NodeState.COMPLETED)
        node = store.get_node("NODE_A")
        assert node is not None
        assert node.state == NodeState.COMPLETED

    def test_any_to_stale(self, store: GraphStore) -> None:
        for initial_state in NodeState:
            node_id = f"NODE_{initial_state.value}"
            store.insert_node(_make_node(node_id, state=initial_state))
            store.set_node_state(node_id, NodeState.STALE)
            node = store.get_node(node_id)
            assert node is not None
            assert node.state == NodeState.STALE

    def test_stale_to_running(self, store: GraphStore) -> None:
        store.insert_node(_make_node("NODE_A", state=NodeState.STALE))
        store.set_node_state("NODE_A", NodeState.RUNNING)
        node = store.get_node("NODE_A")
        assert node is not None
        assert node.state == NodeState.RUNNING

    def test_invalid_pending_to_completed(self, store: GraphStore) -> None:
        store.insert_node(_make_node("NODE_A", state=NodeState.PENDING))
        with pytest.raises(InvalidStateTransitionError) as exc_info:
            store.set_node_state("NODE_A", NodeState.COMPLETED)
        assert exc_info.value.current_state == "PENDING"
        assert exc_info.value.attempted_state == "COMPLETED"

    def test_invalid_completed_to_running(self, store: GraphStore) -> None:
        store.insert_node(_make_node("NODE_A", state=NodeState.COMPLETED))
        with pytest.raises(InvalidStateTransitionError):
            store.set_node_state("NODE_A", NodeState.RUNNING)

    def test_invalid_completed_to_pending(self, store: GraphStore) -> None:
        store.insert_node(_make_node("NODE_A", state=NodeState.COMPLETED))
        with pytest.raises(InvalidStateTransitionError):
            store.set_node_state("NODE_A", NodeState.PENDING)

    def test_set_state_nonexistent_node_raises(self, store: GraphStore) -> None:
        with pytest.raises(NodeNotFoundError):
            store.set_node_state("NODE_MISSING", NodeState.RUNNING)


class TestDualMode:
    """Tests for ephemeral vs persistent storage modes."""

    def test_ephemeral_mode_no_disk_writes(self, tmp_path: Path) -> None:
        """In-memory mode should not create any files."""
        initial_files = set(os.listdir(tmp_path))
        store = GraphStore(persist=False)
        store.insert_node(_make_node())
        store.close()
        final_files = set(os.listdir(tmp_path))
        assert initial_files == final_files

    def test_persistent_mode_survives_close_reopen(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        store = GraphStore(persist=True, db_path=db_path)
        store.insert_node(_make_node("NODE_A"))
        store.insert_node(_make_node("NODE_B"))
        store.insert_edge("NODE_A", "NODE_B", "depends_on")
        store.close()

        # Reopen and verify data persists
        store2 = GraphStore(persist=True, db_path=db_path)
        node = store2.get_node("NODE_A")
        assert node is not None
        assert node.id == "NODE_A"
        edges = store2.get_all_edges()
        assert len(edges) == 1
        store2.close()

    def test_persistent_mode_file_permissions(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        GraphStore(persist=True, db_path=db_path).close()
        # Check file permissions are 0600
        stat = os.stat(db_path)
        permissions = oct(stat.st_mode)[-3:]
        assert permissions == "600"

    def test_persistent_mode_requires_db_path(self) -> None:
        with pytest.raises(ValueError, match="db_path is required"):
            GraphStore(persist=True, db_path=None)

    def test_corrupted_database_raises(self, tmp_path: Path) -> None:
        db_path = tmp_path / "corrupt.db"
        db_path.write_text("this is not a valid sqlite database")
        with pytest.raises(DatabaseCorruptionError):
            GraphStore(persist=True, db_path=db_path)

    def test_schema_mismatch_raises(self, tmp_path: Path) -> None:
        """A database with wrong schema should raise DatabaseCorruptionError."""
        import sqlite3

        db_path = tmp_path / "wrong_schema.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE nodes (id TEXT PRIMARY KEY)")
        conn.execute("CREATE TABLE edges (id INTEGER PRIMARY KEY)")
        conn.commit()
        conn.close()

        with pytest.raises(DatabaseCorruptionError):
            GraphStore(persist=True, db_path=db_path)


class TestTransactionSafety:
    """Tests for transaction rollback on failure."""

    def test_duplicate_insert_leaves_db_unchanged(self, store: GraphStore) -> None:
        node = _make_node()
        store.insert_node(node)
        with pytest.raises(NodeAlreadyExistsError):
            store.insert_node(node)
        # Original node should still be intact
        retrieved = store.get_node("NODE_TEST")
        assert retrieved is not None
        assert retrieved.raw_content == "Hello world"
