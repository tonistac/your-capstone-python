"""
test_app.py — Pytest tests for the Flask web app (app.py).

All external dependencies (run_agent, evaluate_response, load_history) are
patched with unittest.mock so no real network calls are made.
"""

import json
import sys
import os
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path (conftest.py does this too, but be
# explicit so the module can be run standalone).
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Env vars must be present before app.py is imported (it calls load_dotenv
# and logs warnings for missing keys).
os.environ.setdefault("ANTHROPIC_API_KEY", "test-anthropic-key")
os.environ.setdefault("PINECONE_API_KEY", "test-pinecone-key")
os.environ.setdefault("PINECONE_INDEX", "test-index")
os.environ.setdefault("SUPABASE_URL", "https://test.supabase.co")
os.environ.setdefault("SUPABASE_KEY", "test-supabase-key")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def app():
    """
    Return the Flask app with TESTING=True and patched external dependencies.

    run_agent, evaluate_response, and load_history are patched at the
    app.py module level for the lifetime of each test that uses this fixture.
    """
    with (
        patch("app.run_agent") as mock_run_agent,
        patch("app.evaluate_response") as mock_evaluate,
        patch("app.load_history") as mock_load_history,
    ):
        # Sensible defaults — individual tests can override these.
        mock_run_agent.return_value = "This is the agent answer."
        mock_evaluate.return_value = {"score": 4, "reason": "Good response."}
        mock_load_history.return_value = []

        import app as flask_app_module
        flask_app_module.app.config["TESTING"] = True

        # Stash mocks on the fixture value so tests can access them.
        flask_app_module.app._test_mocks = {
            "run_agent": mock_run_agent,
            "evaluate": mock_evaluate,
            "load_history": mock_load_history,
        }

        yield flask_app_module.app


@pytest.fixture()
def client(app):
    """Return a Flask test client bound to the patched app."""
    return app.test_client()


def _mocks(app):
    """Convenience accessor for the stashed mock objects."""
    return app._test_mocks


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _post_json(client, url, body):
    return client.post(
        url,
        data=json.dumps(body),
        content_type="application/json",
    )


# ---------------------------------------------------------------------------
# 1. GET /health → 200 + {"status": "ok"}
# ---------------------------------------------------------------------------

class TestHealth:
    def test_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_body_is_status_ok(self, client):
        resp = client.get("/health")
        assert resp.get_json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# 2. GET / with session_id param → 200
# ---------------------------------------------------------------------------

class TestIndex:
    def test_renders_200_with_session_id(self, client):
        resp = client.get("/?session_id=abc123")
        assert resp.status_code == 200

    # 3. GET / without session_id → falls back to "default"
    def test_renders_200_without_session_id(self, client, app):
        resp = client.get("/")
        assert resp.status_code == 200
        # load_history should have been called with "default"
        mock_load_history = _mocks(app)["load_history"]
        called_session_id = mock_load_history.call_args[0][0]
        assert called_session_id == "default"

    def test_session_id_passed_to_load_history(self, client, app):
        client.get("/?session_id=my-session")
        mock_load_history = _mocks(app)["load_history"]
        called_session_id = mock_load_history.call_args[0][0]
        assert called_session_id == "my-session"


# ---------------------------------------------------------------------------
# 4. POST /ask with valid query → 200 with answer + eval_score
# ---------------------------------------------------------------------------

class TestAsk:
    def test_valid_query_returns_200(self, client):
        resp = _post_json(client, "/ask", {"query": "What is AI?", "session_id": "s1"})
        assert resp.status_code == 200

    def test_valid_query_returns_answer(self, client):
        resp = _post_json(client, "/ask", {"query": "What is AI?", "session_id": "s1"})
        data = resp.get_json()
        assert data["answer"] == "This is the agent answer."

    def test_valid_query_returns_eval_score(self, client):
        resp = _post_json(client, "/ask", {"query": "What is AI?", "session_id": "s1"})
        data = resp.get_json()
        assert data["eval_score"] == 4

    def test_valid_query_returns_eval_reason(self, client):
        resp = _post_json(client, "/ask", {"query": "What is AI?", "session_id": "s1"})
        data = resp.get_json()
        assert data["eval_reason"] == "Good response."

    def test_valid_query_returns_session_id(self, client):
        resp = _post_json(client, "/ask", {"query": "What is AI?", "session_id": "s1"})
        data = resp.get_json()
        assert data["session_id"] == "s1"

    # 5. POST /ask with empty query → 400
    def test_empty_query_returns_400(self, client):
        resp = _post_json(client, "/ask", {"query": "", "session_id": "s1"})
        assert resp.status_code == 400

    def test_empty_query_whitespace_returns_400(self, client):
        resp = _post_json(client, "/ask", {"query": "   ", "session_id": "s1"})
        assert resp.status_code == 400

    def test_empty_query_returns_error_key(self, client):
        resp = _post_json(client, "/ask", {"query": "", "session_id": "s1"})
        data = resp.get_json()
        assert "error" in data

    # 6. POST /ask with missing query key → 400
    def test_missing_query_key_returns_400(self, client):
        resp = _post_json(client, "/ask", {"session_id": "s1"})
        assert resp.status_code == 400

    def test_missing_query_key_returns_error_key(self, client):
        resp = _post_json(client, "/ask", {"session_id": "s1"})
        data = resp.get_json()
        assert "error" in data

    # 7. POST /ask when run_agent raises → 500
    def test_run_agent_exception_returns_500(self, client, app):
        _mocks(app)["run_agent"].side_effect = RuntimeError("Model exploded")
        resp = _post_json(client, "/ask", {"query": "Tell me something", "session_id": "s1"})
        assert resp.status_code == 500

    def test_run_agent_exception_returns_error_key(self, client, app):
        _mocks(app)["run_agent"].side_effect = RuntimeError("Model exploded")
        resp = _post_json(client, "/ask", {"query": "Tell me something", "session_id": "s1"})
        data = resp.get_json()
        assert "error" in data
        assert "Model exploded" in data["error"]

    # 8. POST /ask when evaluate_response fails → 200 with score=0
    def test_evaluate_failure_still_returns_200(self, client, app):
        _mocks(app)["evaluate"].side_effect = Exception("Haiku unavailable")
        resp = _post_json(client, "/ask", {"query": "Test query", "session_id": "s1"})
        assert resp.status_code == 200

    def test_evaluate_failure_score_is_zero(self, client, app):
        _mocks(app)["evaluate"].side_effect = Exception("Haiku unavailable")
        resp = _post_json(client, "/ask", {"query": "Test query", "session_id": "s1"})
        data = resp.get_json()
        assert data["eval_score"] == 0

    def test_evaluate_failure_reason_contains_error(self, client, app):
        _mocks(app)["evaluate"].side_effect = Exception("Haiku unavailable")
        resp = _post_json(client, "/ask", {"query": "Test query", "session_id": "s1"})
        data = resp.get_json()
        assert "Haiku unavailable" in data["eval_reason"]

    def test_evaluate_failure_answer_still_returned(self, client, app):
        _mocks(app)["evaluate"].side_effect = Exception("Haiku unavailable")
        resp = _post_json(client, "/ask", {"query": "Test query", "session_id": "s1"})
        data = resp.get_json()
        assert data["answer"] == "This is the agent answer."

    # 9. POST /ask — answer starting with "ERROR:" is returned, not re-raised
    def test_error_prefixed_answer_returns_200(self, client, app):
        _mocks(app)["run_agent"].return_value = "ERROR: Something went wrong downstream."
        resp = _post_json(client, "/ask", {"query": "Test query", "session_id": "s1"})
        assert resp.status_code == 200

    def test_error_prefixed_answer_is_in_response(self, client, app):
        _mocks(app)["run_agent"].return_value = "ERROR: Something went wrong downstream."
        resp = _post_json(client, "/ask", {"query": "Test query", "session_id": "s1"})
        data = resp.get_json()
        assert data["answer"] == "ERROR: Something went wrong downstream."


# ---------------------------------------------------------------------------
# 10. POST /clear → 200 with status "cleared"
# ---------------------------------------------------------------------------

class TestClear:
    def test_returns_200(self, client):
        resp = _post_json(client, "/clear", {"session_id": "s1"})
        assert resp.status_code == 200

    def test_status_is_cleared(self, client):
        resp = _post_json(client, "/clear", {"session_id": "s1"})
        data = resp.get_json()
        assert data["status"] == "cleared"

    def test_session_id_echoed(self, client):
        resp = _post_json(client, "/clear", {"session_id": "s1"})
        data = resp.get_json()
        assert data["session_id"] == "s1"

    # 11. POST /clear without session_id → uses "default"
    def test_missing_session_id_defaults_to_default(self, client):
        resp = _post_json(client, "/clear", {})
        data = resp.get_json()
        assert data["session_id"] == "default"

    def test_null_session_id_defaults_to_default(self, client):
        resp = _post_json(client, "/clear", {"session_id": None})
        data = resp.get_json()
        assert data["session_id"] == "default"


# ---------------------------------------------------------------------------
# 12. session_id is always coerced to string
# ---------------------------------------------------------------------------

class TestSessionIdCoercion:
    def test_ask_integer_session_id_becomes_string(self, client):
        resp = _post_json(client, "/ask", {"query": "Hello", "session_id": 42})
        data = resp.get_json()
        assert data["session_id"] == "42"
        assert isinstance(data["session_id"], str)

    def test_ask_float_session_id_becomes_string(self, client):
        resp = _post_json(client, "/ask", {"query": "Hello", "session_id": 3.14})
        data = resp.get_json()
        assert isinstance(data["session_id"], str)

    def test_clear_integer_session_id_becomes_string(self, client):
        resp = _post_json(client, "/clear", {"session_id": 99})
        data = resp.get_json()
        assert data["session_id"] == "99"
        assert isinstance(data["session_id"], str)

    def test_ask_run_agent_called_with_string_session_id(self, client, app):
        _post_json(client, "/ask", {"query": "Hello", "session_id": 7})
        mock_run_agent = _mocks(app)["run_agent"]
        called_session_id = mock_run_agent.call_args[0][1]
        assert called_session_id == "7"
        assert isinstance(called_session_id, str)
