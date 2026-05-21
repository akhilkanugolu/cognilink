"""Unit tests for cognilink.core.exceptions."""

import pytest

from cognilink.core.exceptions import (
    CogniLinkError,
    CycleDetectedError,
    DatabaseCorruptionError,
    InvalidStateTransitionError,
    LLMUnavailableError,
    NodeAlreadyExistsError,
    NodeNotFoundError,
    VisualizationWriteError,
)


class TestExceptionHierarchy:
    """All custom exceptions inherit from CogniLinkError."""

    @pytest.mark.parametrize(
        "exc_class",
        [
            NodeAlreadyExistsError,
            NodeNotFoundError,
            CycleDetectedError,
            LLMUnavailableError,
            DatabaseCorruptionError,
            VisualizationWriteError,
            InvalidStateTransitionError,
        ],
    )
    def test_inherits_from_cognilink_error(self, exc_class):
        assert issubclass(exc_class, CogniLinkError)

    def test_cognilink_error_inherits_from_exception(self):
        assert issubclass(CogniLinkError, Exception)


class TestNodeAlreadyExistsError:
    def test_stores_node_id(self):
        err = NodeAlreadyExistsError("NODE_FOO")
        assert err.node_id == "NODE_FOO"

    def test_message_contains_node_id(self):
        err = NodeAlreadyExistsError("NODE_BAR")
        assert "NODE_BAR" in str(err)


class TestNodeNotFoundError:
    def test_stores_node_id(self):
        err = NodeNotFoundError("NODE_MISSING")
        assert err.node_id == "NODE_MISSING"

    def test_message_contains_node_id(self):
        err = NodeNotFoundError("NODE_X")
        assert "NODE_X" in str(err)


class TestCycleDetectedError:
    def test_accepts_list_of_node_ids(self):
        nodes = ["NODE_A", "NODE_B", "NODE_C"]
        err = CycleDetectedError(nodes)
        assert err.involved_nodes == nodes

    def test_message_contains_node_ids(self):
        nodes = ["NODE_A", "NODE_B"]
        err = CycleDetectedError(nodes)
        assert "NODE_A" in str(err)
        assert "NODE_B" in str(err)

    def test_custom_message(self):
        err = CycleDetectedError(["NODE_A"], message="Custom cycle message")
        assert str(err) == "Custom cycle message"
        assert err.involved_nodes == ["NODE_A"]


class TestLLMUnavailableError:
    def test_stores_provider_and_reason(self):
        err = LLMUnavailableError("openai", "timeout")
        assert err.provider == "openai"
        assert err.reason == "timeout"

    def test_default_values(self):
        err = LLMUnavailableError()
        assert err.provider == "unknown"
        assert err.reason == ""

    def test_message_includes_provider(self):
        err = LLMUnavailableError("bedrock", "credentials expired")
        assert "bedrock" in str(err)
        assert "credentials expired" in str(err)


class TestDatabaseCorruptionError:
    def test_stores_db_path_and_reason(self):
        err = DatabaseCorruptionError("/data/cognilink.db", "schema mismatch")
        assert err.db_path == "/data/cognilink.db"
        assert err.reason == "schema mismatch"

    def test_message_includes_path(self):
        err = DatabaseCorruptionError("/tmp/test.db")
        assert "/tmp/test.db" in str(err)


class TestVisualizationWriteError:
    def test_stores_output_path_and_reason(self):
        err = VisualizationWriteError("/output", "permission denied")
        assert err.output_path == "/output"
        assert err.reason == "permission denied"

    def test_message_includes_path(self):
        err = VisualizationWriteError("/viz/output")
        assert "/viz/output" in str(err)


class TestInvalidStateTransitionError:
    def test_stores_current_and_attempted_state(self):
        err = InvalidStateTransitionError("PENDING", "COMPLETED")
        assert err.current_state == "PENDING"
        assert err.attempted_state == "COMPLETED"

    def test_message_includes_both_states(self):
        err = InvalidStateTransitionError("RUNNING", "PENDING")
        assert "RUNNING" in str(err)
        assert "PENDING" in str(err)


class TestCatchAll:
    """Verify that all exceptions can be caught via CogniLinkError."""

    def test_catch_node_already_exists(self):
        with pytest.raises(CogniLinkError):
            raise NodeAlreadyExistsError("NODE_X")

    def test_catch_node_not_found(self):
        with pytest.raises(CogniLinkError):
            raise NodeNotFoundError("NODE_X")

    def test_catch_cycle_detected(self):
        with pytest.raises(CogniLinkError):
            raise CycleDetectedError(["NODE_A", "NODE_B"])

    def test_catch_llm_unavailable(self):
        with pytest.raises(CogniLinkError):
            raise LLMUnavailableError("openai")

    def test_catch_database_corruption(self):
        with pytest.raises(CogniLinkError):
            raise DatabaseCorruptionError("/tmp/db")

    def test_catch_visualization_write(self):
        with pytest.raises(CogniLinkError):
            raise VisualizationWriteError("/output")

    def test_catch_invalid_state_transition(self):
        with pytest.raises(CogniLinkError):
            raise InvalidStateTransitionError("PENDING", "STALE")
