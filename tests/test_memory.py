"""
Tests for memory.py

Coverage:
- load_history: happy path, chronological ordering, session_id coercion,
                error returns [], supabase-py chain used (no raw SQL)
- save_interaction: happy path, session_id coercion, error is swallowed,
                    supabase-py chain used (no raw SQL)
"""

import pytest
from unittest.mock import MagicMock, call, patch


# ---------------------------------------------------------------------------
# load_history
# ---------------------------------------------------------------------------

class TestLoadHistory:

    def test_returns_list(self, patch_supabase):
        from memory import load_history
        result = load_history("session-abc")
        assert isinstance(result, list)

    def test_empty_when_no_rows(self, patch_supabase):
        from memory import load_history
        result = load_history("empty-session")
        assert result == []

    def test_returns_rows_in_chronological_order(self, patch_supabase):
        """
        Supabase returns rows newest-first (desc); load_history must reverse
        them so callers get oldest-first (chronological).
        """
        client_mock, chain_mock = patch_supabase
        # Simulate DB returning newest first
        chain_mock.execute.return_value = MagicMock(data=[
            {"role": "assistant", "content": "Reply", "created_at": "2024-01-02T00:00:00"},
            {"role": "user",      "content": "Hello", "created_at": "2024-01-01T00:00:00"},
        ])
        from memory import load_history
        result = load_history("s1")
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"

    def test_session_id_coerced_to_str(self, patch_supabase):
        """Integer session IDs must be converted to str before the DB call."""
        client_mock, chain_mock = patch_supabase
        from memory import load_history
        load_history(42)  # type: ignore[arg-type]
        # The .eq() call must receive a string, not an int
        eq_call_args = chain_mock.eq.call_args
        _, session_value = eq_call_args.args
        assert isinstance(session_value, str)
        assert session_value == "42"

    def test_uses_supabase_client_not_raw_sql(self, patch_supabase):
        """Verify the fluent supabase-py chain is used — no raw SQL."""
        client_mock, chain_mock = patch_supabase
        from memory import load_history
        load_history("s1")
        client_mock.table.assert_called_once_with("n8n_chat_histories")
        chain_mock.select.assert_called_once()
        chain_mock.eq.assert_called_once()
        chain_mock.order.assert_called_once()
        chain_mock.limit.assert_called_once()
        chain_mock.execute.assert_called_once()

    def test_select_specifies_correct_columns(self, patch_supabase):
        client_mock, chain_mock = patch_supabase
        from memory import load_history
        load_history("s1")
        select_args = chain_mock.select.call_args.args[0]
        for col in ("role", "content", "created_at"):
            assert col in select_args

    def test_default_limit_is_5(self, patch_supabase):
        client_mock, chain_mock = patch_supabase
        from memory import load_history
        load_history("s1")  # no explicit n
        limit_call = chain_mock.limit.call_args.args[0]
        assert limit_call == 5

    def test_custom_n_is_forwarded_to_limit(self, patch_supabase):
        client_mock, chain_mock = patch_supabase
        from memory import load_history
        load_history("s1", n=12)
        limit_call = chain_mock.limit.call_args.args[0]
        assert limit_call == 12

    def test_order_desc_true(self, patch_supabase):
        """The query must order by created_at descending (newest first from DB)."""
        client_mock, chain_mock = patch_supabase
        from memory import load_history
        load_history("s1")
        order_kwargs = chain_mock.order.call_args.kwargs
        assert order_kwargs.get("desc") is True

    def test_returns_empty_list_on_exception(self, patch_supabase):
        """Errors must be swallowed and an empty list returned."""
        client_mock, chain_mock = patch_supabase
        chain_mock.execute.side_effect = Exception("DB down")
        from memory import load_history
        result = load_history("s1")
        assert result == []

    def test_never_raises_on_error(self, patch_supabase):
        client_mock, chain_mock = patch_supabase
        chain_mock.execute.side_effect = RuntimeError("catastrophic failure")
        from memory import load_history
        try:
            result = load_history("s1")
        except Exception:
            pytest.fail("load_history raised an exception instead of returning []")

    def test_result_count_matches_data(self, patch_supabase):
        client_mock, chain_mock = patch_supabase
        chain_mock.execute.return_value = MagicMock(data=[
            {"role": "user", "content": "A", "created_at": "2024-01-01"},
            {"role": "assistant", "content": "B", "created_at": "2024-01-02"},
            {"role": "user", "content": "C", "created_at": "2024-01-03"},
        ])
        from memory import load_history
        result = load_history("s1")
        assert len(result) == 3


# ---------------------------------------------------------------------------
# save_interaction
# ---------------------------------------------------------------------------

class TestSaveInteraction:

    def test_returns_none(self, patch_supabase):
        """save_interaction has no return value — it returns None implicitly."""
        from memory import save_interaction
        result = save_interaction("s1", "user", "hello")
        assert result is None

    def test_session_id_coerced_to_str(self, patch_supabase):
        """Integer session IDs must be str when passed to Supabase insert."""
        client_mock, chain_mock = patch_supabase
        from memory import save_interaction
        save_interaction(7, "user", "text")  # type: ignore[arg-type]
        insert_payload = chain_mock.insert.call_args.args[0]
        assert isinstance(insert_payload["session_id"], str)
        assert insert_payload["session_id"] == "7"

    def test_uses_supabase_client_not_raw_sql(self, patch_supabase):
        """Verify the supabase-py client insert chain is used — no raw SQL."""
        client_mock, chain_mock = patch_supabase
        from memory import save_interaction
        save_interaction("s1", "assistant", "content")
        client_mock.table.assert_called_once_with("n8n_chat_histories")
        chain_mock.insert.assert_called_once()
        chain_mock.execute.assert_called()

    def test_insert_payload_contains_all_fields(self, patch_supabase):
        client_mock, chain_mock = patch_supabase
        from memory import save_interaction
        save_interaction("sess-xyz", "assistant", "The answer is 42.")
        payload = chain_mock.insert.call_args.args[0]
        assert payload["session_id"] == "sess-xyz"
        assert payload["role"] == "assistant"
        assert payload["content"] == "The answer is 42."

    def test_does_not_raise_on_supabase_error(self, patch_supabase):
        """Errors must be silently swallowed."""
        client_mock, chain_mock = patch_supabase
        chain_mock.execute.side_effect = Exception("write failed")
        from memory import save_interaction
        try:
            save_interaction("s1", "user", "text")
        except Exception:
            pytest.fail("save_interaction raised instead of swallowing the error")

    def test_returns_none_on_error(self, patch_supabase):
        client_mock, chain_mock = patch_supabase
        chain_mock.execute.side_effect = Exception("write failed")
        from memory import save_interaction
        result = save_interaction("s1", "user", "text")
        assert result is None
