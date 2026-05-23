"""
Tests for tools.py

Coverage:
- execute_pinecone_search: happy path, no hits, error path, return type
- execute_save_to_memory: happy path, session_id coercion, error path, return type
- execute_tool dispatcher: known tools, unknown tool
- TOOL_DEFINITIONS structure sanity check
"""

import pytest
from unittest.mock import MagicMock, patch

# conftest puts PROJECT_ROOT on sys.path
from conftest import make_text_block, make_tool_use_block


# ---------------------------------------------------------------------------
# execute_pinecone_search
# ---------------------------------------------------------------------------

class TestExecutePineconeSearch:

    def test_returns_str_on_success(self, patch_pinecone):
        """Return type is always str."""
        from tools import execute_pinecone_search
        result = execute_pinecone_search("test query")
        assert isinstance(result, str)

    def test_returns_str_never_none(self, patch_pinecone):
        """None is never returned — convention from CLAUDE.md."""
        from tools import execute_pinecone_search
        result = execute_pinecone_search("anything")
        assert result is not None

    def test_no_hits_returns_no_documents_message(self, patch_pinecone):
        """Empty hits list produces the 'no documents' sentinel string."""
        from tools import execute_pinecone_search
        result = execute_pinecone_search("no match query")
        assert result == "No relevant documents found."

    def test_hits_are_formatted_correctly(self, patch_pinecone):
        """Each hit is rendered with its 1-based index, score, and text."""
        patch_pinecone.search.return_value = {
            "result": {
                "hits": [
                    {"_score": 0.9876, "fields": {"text": "First document text"}},
                    {"_score": 0.5432, "fields": {"text": "Second document text"}},
                ]
            }
        }
        from tools import execute_pinecone_search
        result = execute_pinecone_search("some query", top_k=2)
        assert "[1]" in result
        assert "[2]" in result
        assert "0.988" in result  # rounded to 3 dp
        assert "First document text" in result
        assert "Second document text" in result

    def test_top_k_is_forwarded_to_pinecone(self, patch_pinecone):
        """The top_k argument is forwarded inside the search payload."""
        from tools import execute_pinecone_search
        execute_pinecone_search("query", top_k=7)
        call_kwargs = patch_pinecone.search.call_args
        assert call_kwargs[1]["query"]["top_k"] == 7 or \
               call_kwargs[0][1]["top_k"] == 7 if call_kwargs[0] else True
        # More direct check via keyword arg
        search_kwargs = patch_pinecone.search.call_args.kwargs
        query_arg = search_kwargs.get("query") or patch_pinecone.search.call_args.args[1]
        # Accept either positional or keyword
        all_args = str(patch_pinecone.search.call_args)
        assert "7" in all_args

    def test_error_returns_error_string(self, patch_pinecone):
        """When Pinecone raises, the return value starts with 'ERROR:'."""
        patch_pinecone.search.side_effect = RuntimeError("connection refused")
        from tools import execute_pinecone_search
        result = execute_pinecone_search("query")
        assert result.startswith("ERROR:")
        assert isinstance(result, str)

    def test_error_string_contains_exception_message(self, patch_pinecone):
        patch_pinecone.search.side_effect = RuntimeError("timeout after 30s")
        from tools import execute_pinecone_search
        result = execute_pinecone_search("query")
        assert "timeout after 30s" in result

    def test_missing_fields_key_does_not_crash(self, patch_pinecone):
        """Hits without a 'fields' key should not raise — text defaults to ''."""
        patch_pinecone.search.return_value = {
            "result": {"hits": [{"_score": 0.8}]}
        }
        from tools import execute_pinecone_search
        result = execute_pinecone_search("query")
        assert isinstance(result, str)
        assert result is not None


# ---------------------------------------------------------------------------
# execute_save_to_memory
# ---------------------------------------------------------------------------

class TestExecuteSaveToMemory:

    def test_returns_str_on_success(self, patch_supabase):
        """Return type is always str."""
        from tools import execute_save_to_memory
        result = execute_save_to_memory("sess-1", "user", "Hello")
        assert isinstance(result, str)

    def test_returns_str_never_none(self, patch_supabase):
        from tools import execute_save_to_memory
        result = execute_save_to_memory("sess-1", "assistant", "Answer")
        assert result is not None

    def test_success_message(self, patch_supabase):
        """Successful save returns 'Saved to memory.'."""
        from tools import execute_save_to_memory
        result = execute_save_to_memory("sess-42", "user", "content")
        assert result == "Saved to memory."

    def test_session_id_coerced_to_str(self, patch_supabase):
        """Session IDs must always be strings — an integer must be coerced."""
        client_mock, chain_mock = patch_supabase
        from tools import execute_save_to_memory
        execute_save_to_memory(99, "user", "content")  # type: ignore[arg-type]
        # Retrieve what was inserted
        insert_call = chain_mock.insert.call_args
        payload = insert_call.args[0] if insert_call.args else insert_call.kwargs.get("json", {})
        assert isinstance(payload["session_id"], str)
        assert payload["session_id"] == "99"

    def test_uses_supabase_client_not_raw_sql(self, patch_supabase):
        """No raw SQL — the supabase client .table().insert().execute() chain is used."""
        client_mock, chain_mock = patch_supabase
        from tools import execute_save_to_memory, TABLE
        execute_save_to_memory("s1", "user", "text")
        # tools.py uses the TABLE constant for the table name
        table_calls = [c.args[0] for c in client_mock.table.call_args_list]
        assert TABLE in table_calls, f"table() call args: {table_calls}"
        chain_mock.insert.assert_called()
        chain_mock.execute.assert_called()

    def test_error_returns_error_string(self, patch_supabase):
        """When Supabase raises, result starts with 'ERROR:'."""
        client_mock, chain_mock = patch_supabase
        chain_mock.execute.side_effect = Exception("auth error")
        from tools import execute_save_to_memory
        result = execute_save_to_memory("s1", "user", "text")
        assert result.startswith("ERROR:")
        assert isinstance(result, str)

    def test_error_string_contains_exception_message(self, patch_supabase):
        client_mock, chain_mock = patch_supabase
        chain_mock.execute.side_effect = Exception("network unreachable")
        from tools import execute_save_to_memory
        result = execute_save_to_memory("s1", "user", "text")
        assert "network unreachable" in result


# ---------------------------------------------------------------------------
# execute_tool dispatcher
# ---------------------------------------------------------------------------

class TestExecuteToolDispatcher:

    def test_pinecone_search_routed_correctly(self, patch_pinecone):
        from tools import execute_tool
        result = execute_tool("pinecone_search", {"query": "test"})
        assert isinstance(result, str)
        assert result is not None

    def test_save_to_memory_routed_correctly(self, patch_supabase):
        from tools import execute_tool
        result = execute_tool("save_to_memory", {
            "session_id": "s1", "role": "user", "content": "hi"
        })
        assert result == "Saved to memory."

    def test_unknown_tool_returns_error_string(self):
        from tools import execute_tool
        result = execute_tool("nonexistent_tool", {})
        assert result.startswith("ERROR:")
        assert "nonexistent_tool" in result

    def test_dispatcher_always_returns_str(self, patch_pinecone, patch_supabase):
        from tools import execute_tool
        for name, inp in [
            ("pinecone_search", {"query": "q"}),
            ("save_to_memory", {"session_id": "s", "role": "user", "content": "c"}),
            ("unknown", {}),
        ]:
            result = execute_tool(name, inp)
            assert isinstance(result, str), f"execute_tool({name!r}) returned {type(result)}"
            assert result is not None

    def test_dispatcher_never_returns_none(self, patch_pinecone, patch_supabase):
        from tools import execute_tool
        assert execute_tool("pinecone_search", {"query": "q"}) is not None
        assert execute_tool("save_to_memory", {"session_id": "s", "role": "user", "content": "c"}) is not None
        assert execute_tool("bad_name", {}) is not None


# ---------------------------------------------------------------------------
# TOOL_DEFINITIONS sanity checks
# ---------------------------------------------------------------------------

class TestToolDefinitions:

    def test_tool_definitions_is_list(self):
        from tools import TOOL_DEFINITIONS
        assert isinstance(TOOL_DEFINITIONS, list)

    def test_all_custom_tools_have_name(self):
        from tools import TOOL_DEFINITIONS
        for tool in TOOL_DEFINITIONS:
            assert "name" in tool, f"Tool missing 'name': {tool}"

    def test_pinecone_search_definition_present(self):
        from tools import TOOL_DEFINITIONS
        names = [t["name"] for t in TOOL_DEFINITIONS]
        assert "pinecone_search" in names

    def test_save_to_memory_definition_present(self):
        from tools import TOOL_DEFINITIONS
        names = [t["name"] for t in TOOL_DEFINITIONS]
        assert "save_to_memory" in names

    def test_pinecone_search_requires_query(self):
        from tools import TOOL_DEFINITIONS
        tool = next(t for t in TOOL_DEFINITIONS if t["name"] == "pinecone_search")
        required = tool["input_schema"]["required"]
        assert "query" in required

    def test_save_to_memory_session_id_is_string_type(self):
        from tools import TOOL_DEFINITIONS
        tool = next(t for t in TOOL_DEFINITIONS if t["name"] == "save_to_memory")
        props = tool["input_schema"]["properties"]
        assert props["session_id"]["type"] == "string"

    def test_tool_definitions_has_at_least_two_entries(self):
        """TOOL_DEFINITIONS must contain at least the two custom tools."""
        from tools import TOOL_DEFINITIONS
        custom_names = {t["name"] for t in TOOL_DEFINITIONS if "input_schema" in t}
        assert "pinecone_search" in custom_names
        assert "save_to_memory" in custom_names
