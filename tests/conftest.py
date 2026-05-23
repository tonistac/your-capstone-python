"""
Shared fixtures for the research-assistant agent test suite.

All external I/O is patched here so individual test modules can import the
fixtures they need without repeating boilerplate.
"""

import os
import sys
import types
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Make the project root importable when running pytest from any directory.
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Environment variables (must be set before any project module is imported
# so that load_dotenv() in each module does not blow up on missing keys).
# ---------------------------------------------------------------------------
os.environ.setdefault("ANTHROPIC_API_KEY", "test-anthropic-key")
os.environ.setdefault("PINECONE_API_KEY", "test-pinecone-key")
os.environ.setdefault("PINECONE_INDEX", "test-index")
os.environ.setdefault("SUPABASE_URL", "https://test.supabase.co")
os.environ.setdefault("SUPABASE_KEY", "test-supabase-key")


# ---------------------------------------------------------------------------
# Helpers: build fake Anthropic response objects
# ---------------------------------------------------------------------------

def _make_text_block(text: str = "Final answer."):
    """Return a minimal fake TextBlock."""
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def _make_tool_use_block(name: str = "pinecone_search", tool_id: str = "tu_001",
                          input_data: Optional[dict] = None):
    """Return a minimal fake ToolUseBlock."""
    block = MagicMock()
    block.type = "tool_use"
    block.name = name
    block.id = tool_id
    block.input = input_data or {"query": "test query"}
    return block


def _make_anthropic_response(stop_reason: str = "end_turn", content=None,
                              input_tokens: int = 100, output_tokens: int = 50):  # type: ignore[assignment]
    """Build a fake anthropic.Message object."""
    resp = MagicMock()
    resp.stop_reason = stop_reason
    resp.content = content if content is not None else [_make_text_block()]
    resp.usage = MagicMock()
    resp.usage.input_tokens = input_tokens
    resp.usage.output_tokens = output_tokens
    return resp


# ---------------------------------------------------------------------------
# Fixtures: Supabase client chain
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_supabase_client():
    """
    Return a MagicMock that mimics the fluent supabase-py client chain:

        client.table(name).select(...).eq(...).order(...).limit(n).execute()
        client.table(name).insert({...}).execute()
    """
    client = MagicMock()

    # Build a reusable query-chain mock whose .execute() returns something
    # sensible.  Individual tests can override chain.execute.return_value.
    chain = MagicMock()
    chain.select.return_value = chain
    chain.eq.return_value = chain
    chain.order.return_value = chain
    chain.limit.return_value = chain
    chain.insert.return_value = chain
    chain.execute.return_value = MagicMock(data=[])

    client.table.return_value = chain
    return client, chain


@pytest.fixture()
def patch_supabase(mock_supabase_client):
    """
    Patch create_client in both memory.py and tools.py so they return the
    shared mock client.  Yields (client_mock, chain_mock).
    """
    client_mock, chain_mock = mock_supabase_client
    with (
        patch("memory.create_client", return_value=client_mock),
        patch("tools.create_client", return_value=client_mock),
    ):
        yield client_mock, chain_mock


# ---------------------------------------------------------------------------
# Fixtures: Pinecone index
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_pinecone_index():
    """Return a MagicMock Pinecone index with a no-hit search result."""
    index = MagicMock()
    index.search.return_value = {"result": {"hits": []}}
    return index


@pytest.fixture()
def patch_pinecone(mock_pinecone_index):
    """
    Patch the Pinecone constructor and Index call so tools._pinecone_index()
    returns the mock index.  Yields the index mock.
    """
    pinecone_instance = MagicMock()
    pinecone_instance.Index.return_value = mock_pinecone_index

    with patch("tools.Pinecone", return_value=pinecone_instance):
        yield mock_pinecone_index


# ---------------------------------------------------------------------------
# Fixtures: Anthropic client
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_anthropic_client():
    """Return a bare MagicMock Anthropic client."""
    return MagicMock()


@pytest.fixture()
def patch_anthropic_agent(mock_anthropic_client):
    """
    Patch anthropic.Anthropic inside agent.py.
    The default response is a single end_turn text reply.
    Yields the client mock so tests can reconfigure .messages.create.
    """
    mock_anthropic_client.messages.create.return_value = _make_anthropic_response()
    with patch("agent.anthropic.Anthropic", return_value=mock_anthropic_client):
        yield mock_anthropic_client


@pytest.fixture()
def patch_anthropic_evals(mock_anthropic_client):
    """
    Patch anthropic.Anthropic inside evals.py.
    Default response contains a submit_evaluation tool_use block.
    Yields the client mock.
    """
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.name = "submit_evaluation"
    tool_block.input = {"score": 4, "reason": "Good response."}

    resp = MagicMock()
    resp.content = [tool_block]

    mock_anthropic_client.messages.create.return_value = resp
    with patch("evals.anthropic.Anthropic", return_value=mock_anthropic_client):
        yield mock_anthropic_client


# ---------------------------------------------------------------------------
# Convenience re-exports so individual test modules can call the helpers
# ---------------------------------------------------------------------------
make_text_block = _make_text_block
make_tool_use_block = _make_tool_use_block
make_anthropic_response = _make_anthropic_response
