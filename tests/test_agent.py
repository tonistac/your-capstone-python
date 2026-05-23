"""
Tests for agent.py

The live agent.py (as read from disk) has these key behaviours:
- session_id = str(session_id) at the top
- save_interaction(session_id, "user", query) is called before load_history
- load_history returns messages already containing the saved user message
- messages list is built from load_history only (no manual append of query)
- Tool loop: MAX_ITERATIONS = 10, else-branch logs a warning
- _text() joins b.text for all blocks that hasattr(b, "text")
- On empty _text() result → "ERROR: Agent reached max iterations..."
- save_interaction(session_id, "assistant", final) at the end
- run_agent does NOT have a try/except — model errors propagate
"""

import logging
from typing import Optional
import pytest
from unittest.mock import MagicMock, patch, call


# ---------------------------------------------------------------------------
# Helpers — note: .text must be a real str, NOT a MagicMock sub-attribute
# ---------------------------------------------------------------------------

def _text_block(text: str = "Final answer."):
    """Fake TextBlock with a genuine str .text attribute."""
    b = MagicMock()
    b.type = "text"
    b.text = text          # real str so " ".join() works
    return b


def _tool_use_block(name: str = "pinecone_search", tool_id: str = "tu_1",
                    input_data: Optional[dict] = None):
    """Fake ToolUseBlock — NO .text attribute so _text() skips it."""
    b = MagicMock(spec=["type", "name", "id", "input"])
    b.type = "tool_use"
    b.name = name
    b.id = tool_id
    b.input = input_data or {"query": "test"}
    return b


def _make_response(stop_reason: str = "end_turn", content=None,
                   in_tokens: int = 100, out_tokens: int = 50):
    resp = MagicMock()
    resp.stop_reason = stop_reason
    resp.content = content if content is not None else [_text_block()]
    resp.usage = MagicMock(input_tokens=in_tokens, output_tokens=out_tokens)
    return resp


# ---------------------------------------------------------------------------
# Shared patch helper: patches save_interaction AND load_history in agent.py
# so tests can control what load_history returns without real DB calls.
# ---------------------------------------------------------------------------

def _patch_memory(history=None):
    """Context manager that patches both memory functions used in agent.py."""
    if history is None:
        history = []
    return (
        patch("agent.save_interaction"),
        patch("agent.load_history", return_value=history),
    )


# ---------------------------------------------------------------------------
# Basic return-type and session_id conventions
# ---------------------------------------------------------------------------

class TestRunAgentBasics:

    def test_returns_str(self, patch_anthropic_agent):
        p_save = patch("agent.save_interaction")
        p_load = patch("agent.load_history", return_value=[])
        with p_save, p_load:
            from agent import run_agent
            result = run_agent("What is AI?", "session-1")
        assert isinstance(result, str)

    def test_never_returns_none(self, patch_anthropic_agent):
        p_save = patch("agent.save_interaction")
        p_load = patch("agent.load_history", return_value=[])
        with p_save, p_load:
            from agent import run_agent
            result = run_agent("question", "sess")
        assert result is not None

    def test_session_id_integer_coerced_to_str(self, patch_anthropic_agent):
        """run_agent converts integer session_id to str (CLAUDE.md convention)."""
        received_ids = []

        def capture_save(sid, *a, **kw):
            received_ids.append(sid)

        with patch("agent.save_interaction", side_effect=capture_save):
            with patch("agent.load_history", return_value=[]):
                from agent import run_agent
                run_agent("hello", 42)  # type: ignore[arg-type]

        assert received_ids[0] == "42"
        assert isinstance(received_ids[0], str)

    def test_returns_text_from_response(self, patch_anthropic_agent):
        patch_anthropic_agent.messages.create.return_value = \
            _make_response(content=[_text_block("The sky is blue.")])
        with patch("agent.save_interaction"):
            with patch("agent.load_history", return_value=[]):
                from agent import run_agent
                result = run_agent("sky colour?", "s1")
        assert "The sky is blue." in result

    def test_multiple_text_blocks_joined(self, patch_anthropic_agent):
        patch_anthropic_agent.messages.create.return_value = _make_response(
            content=[_text_block("Part one."), _text_block("Part two.")]
        )
        with patch("agent.save_interaction"):
            with patch("agent.load_history", return_value=[]):
                from agent import run_agent
                result = run_agent("q", "s1")
        assert "Part one." in result
        assert "Part two." in result

    def test_no_text_blocks_returns_error_string(self, patch_anthropic_agent):
        """
        If all content blocks are non-text (no .text attr), _text() returns '',
        and agent.py returns the max-iterations error sentinel string.
        """
        patch_anthropic_agent.messages.create.return_value = _make_response(
            content=[_tool_use_block()],   # spec=['type','name','id','input'] — no .text
            stop_reason="end_turn",
        )
        with patch("agent.save_interaction"):
            with patch("agent.load_history", return_value=[]):
                from agent import run_agent
                result = run_agent("q", "s1")
        # _text() returns '' → fallback error message is returned
        assert isinstance(result, str)
        assert "ERROR" in result


# ---------------------------------------------------------------------------
# History loading — the user message is saved via save_interaction; load_history
# is expected to return all messages including the just-saved user message.
# The messages list passed to the API = exactly what load_history returns.
# ---------------------------------------------------------------------------

class TestRunAgentHistory:

    def test_load_history_called_with_session_id(self, patch_anthropic_agent):
        with patch("agent.save_interaction"):
            with patch("agent.load_history", return_value=[]) as mock_lh:
                from agent import run_agent
                run_agent("question", "my-session")
        mock_lh.assert_called_once_with("my-session")

    def test_save_interaction_called_for_user_message(self, patch_anthropic_agent):
        """The user query must be persisted before the API call."""
        save_calls = []
        with patch("agent.save_interaction", side_effect=lambda *a: save_calls.append(a)):
            with patch("agent.load_history", return_value=[]):
                from agent import run_agent
                run_agent("My question", "s1")
        user_save = next((c for c in save_calls if c[1] == "user"), None)
        assert user_save is not None
        assert user_save[2] == "My question"

    def test_history_rows_form_messages_list(self, patch_anthropic_agent):
        """
        Messages sent to the API come directly from load_history output.
        (The user query was already saved by save_interaction before the call.)
        """
        history = [
            {"role": "user",      "content": "Earlier question"},
            {"role": "assistant", "content": "Earlier answer"},
            {"role": "user",      "content": "My question"},
        ]
        messages_sent = []

        def capture_create(**kwargs):
            messages_sent.extend(kwargs["messages"])
            return _make_response()

        patch_anthropic_agent.messages.create.side_effect = capture_create

        with patch("agent.save_interaction"):
            with patch("agent.load_history", return_value=history):
                from agent import run_agent
                run_agent("My question", "s1")

        assert messages_sent[0]["content"] == "Earlier question"
        assert messages_sent[1]["content"] == "Earlier answer"
        assert messages_sent[2]["content"] == "My question"

    def test_empty_history_means_empty_initial_messages(self, patch_anthropic_agent):
        """If load_history returns [], no messages are passed before the loop."""
        messages_sent = []

        def capture_create(**kwargs):
            messages_sent.extend(kwargs["messages"])
            return _make_response()

        patch_anthropic_agent.messages.create.side_effect = capture_create

        with patch("agent.save_interaction"):
            with patch("agent.load_history", return_value=[]):
                from agent import run_agent
                run_agent("Solo question", "s1")

        # No pre-existing messages — messages list starts empty
        assert len(messages_sent) == 0

    def test_save_interaction_called_for_assistant_response(self, patch_anthropic_agent):
        """The assistant's final answer must also be persisted."""
        patch_anthropic_agent.messages.create.return_value = \
            _make_response(content=[_text_block("The answer.")])
        save_calls = []
        with patch("agent.save_interaction", side_effect=lambda *a: save_calls.append(a)):
            with patch("agent.load_history", return_value=[]):
                from agent import run_agent
                run_agent("q", "s1")
        assistant_save = next((c for c in save_calls if c[1] == "assistant"), None)
        assert assistant_save is not None
        assert "The answer." in assistant_save[2]


# ---------------------------------------------------------------------------
# Tool-use loop
# ---------------------------------------------------------------------------

class TestRunAgentToolLoop:

    def test_execute_tool_called_on_tool_use_response(self, patch_anthropic_agent):
        tool_resp = _make_response(
            stop_reason="tool_use",
            content=[_tool_use_block("pinecone_search", "tu_1", {"query": "AI"})],
        )
        end_resp = _make_response(stop_reason="end_turn", content=[_text_block("Done.")])
        patch_anthropic_agent.messages.create.side_effect = [tool_resp, end_resp]

        with patch("agent.save_interaction"):
            with patch("agent.load_history", return_value=[]):
                with patch("agent.execute_tool", return_value="Search result") as mock_et:
                    from agent import run_agent
                    run_agent("query", "s1")

        mock_et.assert_called_once_with("pinecone_search", {"query": "AI"})

    def test_tool_result_appended_as_user_message(self, patch_anthropic_agent):
        tool_resp = _make_response(
            stop_reason="tool_use",
            content=[_tool_use_block("pinecone_search", "tu_abc", {"query": "X"})],
        )
        end_resp = _make_response(stop_reason="end_turn", content=[_text_block("Final.")])

        all_messages = []

        def capture(**kwargs):
            all_messages.append(list(kwargs["messages"]))
            if len(all_messages) == 1:
                return tool_resp
            return end_resp

        patch_anthropic_agent.messages.create.side_effect = capture

        with patch("agent.save_interaction"):
            with patch("agent.load_history", return_value=[]):
                with patch("agent.execute_tool", return_value="tool output"):
                    from agent import run_agent
                    run_agent("q", "s1")

        second_call_messages = all_messages[1]
        user_tool_msg = next(
            (m for m in second_call_messages if m["role"] == "user"
             and isinstance(m["content"], list)),
            None,
        )
        assert user_tool_msg is not None
        assert any(
            item.get("type") == "tool_result" for item in user_tool_msg["content"]
        )

    def test_multiple_tool_calls_all_dispatched(self, patch_anthropic_agent):
        tool_resp = _make_response(
            stop_reason="tool_use",
            content=[
                _tool_use_block("pinecone_search", "tu_1", {"query": "A"}),
                _tool_use_block("save_to_memory", "tu_2",
                                {"session_id": "s1", "role": "user", "content": "note"}),
            ],
        )
        end_resp = _make_response(stop_reason="end_turn", content=[_text_block("ok")])
        patch_anthropic_agent.messages.create.side_effect = [tool_resp, end_resp]

        with patch("agent.save_interaction"):
            with patch("agent.load_history", return_value=[]):
                with patch("agent.execute_tool", return_value="ok") as mock_et:
                    from agent import run_agent
                    run_agent("q", "s1")

        assert mock_et.call_count == 2

    def test_tool_use_with_no_tool_blocks_exits_loop(self, patch_anthropic_agent):
        weird_resp = _make_response(
            stop_reason="tool_use",
            content=[_text_block("Hmm.")],
        )
        patch_anthropic_agent.messages.create.return_value = weird_resp

        with patch("agent.save_interaction"):
            with patch("agent.load_history", return_value=[]):
                with patch("agent.execute_tool", return_value="x"):
                    from agent import run_agent
                    result = run_agent("q", "s1")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Max iterations boundary
# ---------------------------------------------------------------------------

class TestRunAgentMaxIterations:

    def test_loop_exits_after_max_iterations(self, patch_anthropic_agent):
        from agent import MAX_ITERATIONS

        always_tool = _make_response(
            stop_reason="tool_use",
            content=[_tool_use_block("pinecone_search", "tu_x", {"query": "q"})],
        )
        patch_anthropic_agent.messages.create.return_value = always_tool

        with patch("agent.save_interaction"):
            with patch("agent.load_history", return_value=[]):
                with patch("agent.execute_tool", return_value="result"):
                    from agent import run_agent
                    run_agent("query", "s1")

        assert patch_anthropic_agent.messages.create.call_count == MAX_ITERATIONS

    def test_max_iterations_warning_logged(self, patch_anthropic_agent, caplog):
        from agent import MAX_ITERATIONS

        always_tool = _make_response(
            stop_reason="tool_use",
            content=[_tool_use_block("pinecone_search", "tu_x", {"query": "q"})],
        )
        patch_anthropic_agent.messages.create.return_value = always_tool

        with caplog.at_level(logging.WARNING, logger="agent"):
            with patch("agent.save_interaction"):
                with patch("agent.load_history", return_value=[]):
                    with patch("agent.execute_tool", return_value="result"):
                        from agent import run_agent
                        run_agent("query", "sess-warn")

        warning_text = " ".join(rec.message for rec in caplog.records)
        assert "max_iterations" in warning_text.lower() or "max_iteration" in warning_text.lower()

    def test_max_iterations_value_is_ten(self):
        from agent import MAX_ITERATIONS
        assert MAX_ITERATIONS == 10

    def test_returns_str_even_when_max_iterations_hit(self, patch_anthropic_agent):
        always_tool = _make_response(
            stop_reason="tool_use",
            content=[_tool_use_block()],
        )
        patch_anthropic_agent.messages.create.return_value = always_tool

        with patch("agent.save_interaction"):
            with patch("agent.load_history", return_value=[]):
                with patch("agent.execute_tool", return_value="r"):
                    from agent import run_agent
                    result = run_agent("q", "s1")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# API and model parameters
# ---------------------------------------------------------------------------

class TestRunAgentApiParams:

    def test_uses_tool_definitions_from_tools_module(self, patch_anthropic_agent):
        from agent import run_agent
        from tools import TOOL_DEFINITIONS
        with patch("agent.save_interaction"):
            with patch("agent.load_history", return_value=[]):
                run_agent("q", "s1")
        create_kwargs = patch_anthropic_agent.messages.create.call_args.kwargs
        assert create_kwargs["tools"] is TOOL_DEFINITIONS

    def test_uses_sonnet_model(self, patch_anthropic_agent):
        from agent import run_agent, MODEL
        with patch("agent.save_interaction"):
            with patch("agent.load_history", return_value=[]):
                run_agent("q", "s1")
        create_kwargs = patch_anthropic_agent.messages.create.call_args.kwargs
        assert create_kwargs["model"] == MODEL

    def test_system_prompt_is_set(self, patch_anthropic_agent):
        from agent import run_agent, SYSTEM
        with patch("agent.save_interaction"):
            with patch("agent.load_history", return_value=[]):
                run_agent("q", "s1")
        create_kwargs = patch_anthropic_agent.messages.create.call_args.kwargs
        assert create_kwargs["system"] == SYSTEM

    def test_max_tokens_is_set(self, patch_anthropic_agent):
        from agent import run_agent
        with patch("agent.save_interaction"):
            with patch("agent.load_history", return_value=[]):
                run_agent("q", "s1")
        create_kwargs = patch_anthropic_agent.messages.create.call_args.kwargs
        assert "max_tokens" in create_kwargs
        assert create_kwargs["max_tokens"] > 0


# ---------------------------------------------------------------------------
# Error behaviour
# ---------------------------------------------------------------------------

class TestRunAgentErrors:

    def test_model_exception_propagates(self, patch_anthropic_agent):
        """
        run_agent has no try/except — exceptions from the model propagate
        to the caller.  This is the actual documented behaviour.
        """
        patch_anthropic_agent.messages.create.side_effect = RuntimeError("API down")
        with patch("agent.save_interaction"):
            with patch("agent.load_history", return_value=[]):
                from agent import run_agent
                with pytest.raises(RuntimeError, match="API down"):
                    run_agent("q", "s1")

    def test_error_message_returned_when_no_response(self):
        """
        The early guard 'if response is None: return "ERROR:..."' is tested
        by having the loop not execute (MAX_ITERATIONS = 0 via monkeypatching).
        """
        import agent as agent_module
        client = MagicMock()
        # The loop won't run at all if MAX_ITERATIONS is 0; response stays None
        original = agent_module.MAX_ITERATIONS
        agent_module.MAX_ITERATIONS = 0
        try:
            with patch("agent.anthropic.Anthropic", return_value=client):
                with patch("agent.save_interaction"):
                    with patch("agent.load_history", return_value=[]):
                        result = agent_module.run_agent("q", "s1")
            assert result == "ERROR: No response from model."
        finally:
            agent_module.MAX_ITERATIONS = original
