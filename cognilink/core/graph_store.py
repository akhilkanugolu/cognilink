"""CogniLink Graph Storage Engine.

SQLite-backed storage managing nodes and edges with support for both
ephemeral in-memory and persistent file-based modes. Provides CRUD
operations with transactional safety, relationship traversal, and
state machine enforcement.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from cognilink.core.exceptions import (
    DatabaseCorruptionError,
    InvalidStateTransitionError,
    NodeAlreadyExistsError,
    NodeNotFoundError,
)
from cognilink.core.models import Edge, Node, NodeState

# Valid state transitions: (current_state, new_state)
_VALID_TRANSITIONS = {
    (NodeState.PENDING, NodeState.RUNNING),
    (NodeState.RUNNING, NodeState.COMPLETED),
    (NodeState.STALE, NodeState.RUNNING),
    # Any state → STALE
    (NodeState.PENDING, NodeState.STALE),
    (NodeState.RUNNING, NodeState.STALE),
    (NodeState.COMPLETED, NodeState.STALE),
    (NodeState.STALE, NodeState.STALE),
}


class GraphStore:
    """SQLite database wrapper managing nodes and edges tables.

    Supports both ephemeral in-memory (:memory:) and persistent on-disk
    storage modes. Provides CRUD operations with transactional safety
    and relationship traversal queries.

    Args:
        persist: If True, use file-based SQLite; if False, use in-memory.
        db_path: Path to the SQLite database file (required when persist=True).
    """

    def __init__(self, persist: bool = False, db_path: Optional[Path] = None) -> None:
        self._persist = persist
        self._db_path = db_path

        if persist:
            if db_path is None:
                raise ValueError("db_path is required when persist=True")
            db_str = str(db_path)
            existing = db_path.exists()
            self._conn = sqlite3.connect(db_str)
            self._conn.execute("PRAGMA foreign_keys = ON")
            if existing:
                self._validate_schema()
            else:
                self._initialize_schema()
                os.chmod(db_str, 0o600)
        else:
            self._conn = sqlite3.connect(":memory:")
            self._conn.execute("PRAGMA foreign_keys = ON")
            self._initialize_schema()

    def _initialize_schema(self) -> None:
        """Create the nodes and edges tables with indices."""
        with self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS nodes (
                    id TEXT PRIMARY KEY,
                    source_type TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    raw_content TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    system_role_prompt TEXT NOT NULL,
                    output_artifact TEXT,
                    state TEXT NOT NULL,
                    parent_node_id TEXT,
                    page_start INTEGER,
                    page_end INTEGER
                );

                CREATE TABLE IF NOT EXISTS edges (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    upstream_id TEXT NOT NULL,
                    downstream_id TEXT NOT NULL,
                    relationship_type TEXT NOT NULL,
                    FOREIGN KEY (upstream_id) REFERENCES nodes(id),
                    FOREIGN KEY (downstream_id) REFERENCES nodes(id)
                );

                CREATE INDEX IF NOT EXISTS idx_edges_upstream
                    ON edges(upstream_id);
                CREATE INDEX IF NOT EXISTS idx_edges_downstream
                    ON edges(downstream_id);
                """
            )

    def _validate_schema(self) -> None:
        """Validate that an existing database has the expected schema."""
        try:
            cursor = self._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
            tables = {row[0] for row in cursor.fetchall()}
            if "nodes" not in tables or "edges" not in tables:
                raise DatabaseCorruptionError(
                    str(self._db_path),
                    reason="Missing required tables (nodes, edges)",
                )

            # Validate nodes table columns
            cursor = self._conn.execute("PRAGMA table_info(nodes)")
            node_columns = {row[1] for row in cursor.fetchall()}
            required_node_cols = {
                "id", "source_type", "source_path", "raw_content",
                "summary", "system_role_prompt", "output_artifact",
                "state", "parent_node_id", "page_start", "page_end",
            }
            if not required_node_cols.issubset(node_columns):
                missing = required_node_cols - node_columns
                raise DatabaseCorruptionError(
                    str(self._db_path),
                    reason=f"Missing columns in nodes table: {missing}",
                )

            # Validate edges table columns
            cursor = self._conn.execute("PRAGMA table_info(edges)")
            edge_columns = {row[1] for row in cursor.fetchall()}
            required_edge_cols = {
                "id", "upstream_id", "downstream_id", "relationship_type",
            }
            if not required_edge_cols.issubset(edge_columns):
                missing = required_edge_cols - edge_columns
                raise DatabaseCorruptionError(
                    str(self._db_path),
                    reason=f"Missing columns in edges table: {missing}",
                )
        except sqlite3.DatabaseError as e:
            raise DatabaseCorruptionError(
                str(self._db_path), reason=str(e)
            ) from e

    def insert_node(self, node: Node) -> None:
        """Insert a node into the graph store.

        Args:
            node: The Node to insert.

        Raises:
            NodeAlreadyExistsError: If a node with the same ID already exists.
        """
        try:
            with self._conn:
                self._conn.execute(
                    """
                    INSERT INTO nodes (
                        id, source_type, source_path, raw_content, summary,
                        system_role_prompt, output_artifact, state,
                        parent_node_id, page_start, page_end
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        node.id,
                        node.source_type,
                        node.source_path,
                        node.raw_content,
                        node.summary,
                        node.system_role_prompt,
                        node.output_artifact,
                        node.state.value,
                        node.parent_node_id,
                        node.page_start,
                        node.page_end,
                    ),
                )
        except sqlite3.IntegrityError:
            raise NodeAlreadyExistsError(node.id)

    def get_node(self, node_id: str) -> Optional[Node]:
        """Retrieve a node by its ID.

        Args:
            node_id: The ID of the node to retrieve.

        Returns:
            The Node if found, None otherwise.
        """
        cursor = self._conn.execute(
            """
            SELECT id, source_type, source_path, raw_content, summary,
                   system_role_prompt, output_artifact, state,
                   parent_node_id, page_start, page_end
            FROM nodes WHERE id = ?
            """,
            (node_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_node(row)

    def update_node(self, node_id: str, **fields: Any) -> None:
        """Update specific fields of an existing node.

        Args:
            node_id: The ID of the node to update.
            **fields: Keyword arguments of field names and their new values.

        Raises:
            NodeNotFoundError: If the node does not exist.
        """
        if not fields:
            return

        # Verify node exists
        if self.get_node(node_id) is None:
            raise NodeNotFoundError(node_id)

        # Build SET clause from provided fields
        allowed_fields = {
            "source_type", "source_path", "raw_content", "summary",
            "system_role_prompt", "output_artifact", "state",
            "parent_node_id", "page_start", "page_end",
        }
        update_fields: Dict[str, Any] = {}
        for key, value in fields.items():
            if key not in allowed_fields:
                raise ValueError(f"Cannot update field '{key}' on Node")
            if key == "state" and isinstance(value, NodeState):
                update_fields[key] = value.value
            else:
                update_fields[key] = value

        set_clause = ", ".join(f"{k} = ?" for k in update_fields)
        values = list(update_fields.values()) + [node_id]

        with self._conn:
            self._conn.execute(
                f"UPDATE nodes SET {set_clause} WHERE id = ?",  # noqa: S608
                values,
            )

    def delete_node(self, node_id: str) -> None:
        """Delete a node and all its associated edges.

        Args:
            node_id: The ID of the node to delete.

        Raises:
            NodeNotFoundError: If the node does not exist.
        """
        if self.get_node(node_id) is None:
            raise NodeNotFoundError(node_id)

        with self._conn:
            # Delete associated edges first
            self._conn.execute(
                "DELETE FROM edges WHERE upstream_id = ? OR downstream_id = ?",
                (node_id, node_id),
            )
            self._conn.execute("DELETE FROM nodes WHERE id = ?", (node_id,))

    def get_all_nodes(self) -> List[Node]:
        """Retrieve all nodes in the graph store.

        Returns:
            A list of all Node objects.
        """
        cursor = self._conn.execute(
            """
            SELECT id, source_type, source_path, raw_content, summary,
                   system_role_prompt, output_artifact, state,
                   parent_node_id, page_start, page_end
            FROM nodes
            """
        )
        return [self._row_to_node(row) for row in cursor.fetchall()]

    def insert_edge(
        self, upstream_id: str, downstream_id: str, relationship_type: str
    ) -> Edge:
        """Insert a directed edge between two nodes.

        Args:
            upstream_id: The ID of the upstream (dependency) node.
            downstream_id: The ID of the downstream (dependent) node.
            relationship_type: Semantic label for the relationship.

        Returns:
            The created Edge object.

        Raises:
            NodeNotFoundError: If either node does not exist.
            ValueError: If upstream_id equals downstream_id (self-edge)
                or if a duplicate edge already exists.
        """
        # Check for self-edges
        if upstream_id == downstream_id:
            raise ValueError(
                f"Self-edges are not allowed: '{upstream_id}' cannot link to itself"
            )

        # Verify both nodes exist
        if self.get_node(upstream_id) is None:
            raise NodeNotFoundError(upstream_id)
        if self.get_node(downstream_id) is None:
            raise NodeNotFoundError(downstream_id)

        # Check for duplicate edges
        cursor = self._conn.execute(
            """
            SELECT id FROM edges
            WHERE upstream_id = ? AND downstream_id = ? AND relationship_type = ?
            """,
            (upstream_id, downstream_id, relationship_type),
        )
        if cursor.fetchone() is not None:
            raise ValueError(
                f"Duplicate edge: '{upstream_id}' → '{downstream_id}' "
                f"with type '{relationship_type}' already exists"
            )

        with self._conn:
            cursor = self._conn.execute(
                """
                INSERT INTO edges (upstream_id, downstream_id, relationship_type)
                VALUES (?, ?, ?)
                """,
                (upstream_id, downstream_id, relationship_type),
            )
            edge_id = cursor.lastrowid

        return Edge(
            id=edge_id,  # type: ignore[arg-type]
            upstream_id=upstream_id,
            downstream_id=downstream_id,
            relationship_type=relationship_type,
        )

    def get_downstream(self, node_id: str) -> List[Edge]:
        """Get all edges where the given node is the upstream (dependency).

        Args:
            node_id: The ID of the upstream node.

        Returns:
            A list of Edge objects representing downstream relationships.
        """
        cursor = self._conn.execute(
            """
            SELECT id, upstream_id, downstream_id, relationship_type
            FROM edges WHERE upstream_id = ?
            """,
            (node_id,),
        )
        return [self._row_to_edge(row) for row in cursor.fetchall()]

    def get_upstream(self, node_id: str) -> List[Edge]:
        """Get all edges where the given node is the downstream (dependent).

        Args:
            node_id: The ID of the downstream node.

        Returns:
            A list of Edge objects representing upstream relationships.
        """
        cursor = self._conn.execute(
            """
            SELECT id, upstream_id, downstream_id, relationship_type
            FROM edges WHERE downstream_id = ?
            """,
            (node_id,),
        )
        return [self._row_to_edge(row) for row in cursor.fetchall()]

    def delete_edge(self, edge_id: int) -> None:
        """Delete an edge by its ID.

        Args:
            edge_id: The ID of the edge to delete.
        """
        with self._conn:
            self._conn.execute("DELETE FROM edges WHERE id = ?", (edge_id,))

    def get_all_edges(self) -> List[Edge]:
        """Retrieve all edges in the graph store.

        Returns:
            A list of all Edge objects.
        """
        cursor = self._conn.execute(
            """
            SELECT id, upstream_id, downstream_id, relationship_type
            FROM edges
            """
        )
        return [self._row_to_edge(row) for row in cursor.fetchall()]

    def set_node_state(self, node_id: str, state: NodeState) -> None:
        """Set the state of a node with transition validation.

        Valid transitions:
            PENDING → RUNNING
            RUNNING → COMPLETED
            Any state → STALE
            STALE → RUNNING

        Args:
            node_id: The ID of the node to update.
            state: The new NodeState to transition to.

        Raises:
            NodeNotFoundError: If the node does not exist.
            InvalidStateTransitionError: If the transition is not valid.
        """
        node = self.get_node(node_id)
        if node is None:
            raise NodeNotFoundError(node_id)

        current_state = node.state
        if (current_state, state) not in _VALID_TRANSITIONS:
            raise InvalidStateTransitionError(current_state.value, state.value)

        with self._conn:
            self._conn.execute(
                "UPDATE nodes SET state = ? WHERE id = ?",
                (state.value, node_id),
            )

    def close(self) -> None:
        """Close the database connection cleanly."""
        self._conn.close()

    # --- Private helpers ---

    @staticmethod
    def _row_to_node(row: tuple) -> Node:
        """Convert a database row tuple to a Node object."""
        return Node(
            id=row[0],
            source_type=row[1],
            source_path=row[2],
            raw_content=row[3],
            summary=row[4],
            system_role_prompt=row[5],
            output_artifact=row[6],
            state=NodeState(row[7]),
            parent_node_id=row[8],
            page_start=row[9],
            page_end=row[10],
        )

    @staticmethod
    def _row_to_edge(row: tuple) -> Edge:
        """Convert a database row tuple to an Edge object."""
        return Edge(
            id=row[0],
            upstream_id=row[1],
            downstream_id=row[2],
            relationship_type=row[3],
        )
