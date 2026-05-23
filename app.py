"""
app.py — Flask web UI for the Research Assistant Agent.

Routes:
  GET  /              Render chat UI with session history
  POST /ask           Run agent + evaluate → JSON response
  POST /clear         Clear UI (no-op on DB; Supabase history preserved)
  GET  /health        Health check → {"status": "ok"}

Architecture:
  Browser → Flask → run_agent() → Claude Sonnet + tools
                  → evaluate_response() → Claude Haiku judge
                  → load_history() → Supabase
"""

import logging
import os
import re
from typing import Optional

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

from agent import run_agent
from evals import evaluate_response
from memory import load_history

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)

# ── Session ID validation ──────────────────────────────────────────────────
# Alphanumeric, hyphens, underscores only — max 64 chars.
_SESSION_RE = re.compile(r"^[\w\-]{1,64}$")


def _validate_session_id(raw) -> Optional[str]:
    """Return the sanitised session_id, or None if invalid.

    Numeric types (int, float) are coerced to their integer string form
    so that session_id=1 and session_id="1" resolve to the same key.
    """
    if isinstance(raw, float):
        raw = int(raw)
    if isinstance(raw, int):
        raw = str(raw)
    value = str(raw or "").strip()
    if not value:
        return "default"
    return value if _SESSION_RE.match(value) else None


# ── App ────────────────────────────────────────────────────────────────────
app = Flask(__name__)


@app.before_request
def _check_env_vars():
    """Warn once at startup if required env vars are missing."""
    pass  # Checked in _startup_check() below, called after app creation.


def _startup_check() -> None:
    """Log warnings for missing required environment variables."""
    required = [
        "ANTHROPIC_API_KEY",
        "PINECONE_API_KEY",
        "PINECONE_INDEX",
        "SUPABASE_URL",
        "SUPABASE_KEY",
    ]
    for var in required:
        if not os.environ.get(var):
            log.warning("Missing required env var: %s — some features may fail", var)


# ── Routes ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Liveness check for deployment environments."""
    return jsonify({"status": "ok"})


@app.get("/")
def index():
    """Render the chat UI, pre-populated with the session's Supabase history."""
    raw_session = request.args.get("session_id", "default")
    session_id = _validate_session_id(raw_session)
    if session_id is None:
        session_id = "default"
    history = load_history(session_id, n=20)
    return render_template("index.html", history=history, session_id=session_id)


@app.post("/ask")
def ask():
    """
    Run the research agent and evaluate the response.

    Body (JSON): { "query": str, "session_id": str }
    Returns (JSON):
      { "answer": str, "eval_score": int, "eval_reason": str, "session_id": str }
    On error:
      { "error": str }  with HTTP 400 or 500
    """
    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()
    session_id = _validate_session_id(data.get("session_id", "default"))

    if not query:
        return jsonify({"error": "Query must not be empty."}), 400

    if session_id is None:
        return jsonify({"error": "Invalid session_id. Use alphanumeric, hyphens, or underscores (max 64 chars)."}), 400

    log.info("session=%s query=%.60s", session_id, query)

    try:
        answer = run_agent(query, session_id)
    except Exception as exc:
        log.exception("run_agent failed for session=%s", session_id)
        return jsonify({"error": f"Agent error: {exc}"}), 500

    # evaluate_response handles its own exceptions internally and always returns
    # a dict. The outer guard is a safety net in case that contract ever changes.
    try:
        eval_result = evaluate_response(answer)
        eval_score = int(eval_result.get("score", 0))
        eval_reason = eval_result.get("reason", "")
    except Exception as exc:
        log.warning("evaluate_response raised unexpectedly: %s", exc)
        eval_score = 0
        eval_reason = f"Evaluator error: {exc}"

    if eval_score == 0:
        log.warning("session=%s evaluator returned score=0: %s", session_id, eval_reason)

    return jsonify({
        "answer": answer,
        "eval_score": eval_score,
        "eval_reason": eval_reason,
        "session_id": session_id,
    })


@app.post("/clear")
def clear():
    """
    UI clear endpoint — resets the client-side chat view.
    Supabase history is intentionally preserved for continuity.
    Returns: { "status": "cleared" }
    """
    data = request.get_json(silent=True) or {}
    session_id = _validate_session_id(data.get("session_id", "default")) or "default"
    log.info("UI cleared for session=%s (Supabase history preserved)", session_id)
    return jsonify({"status": "cleared", "session_id": session_id})


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _startup_check()
    port = int(os.environ.get("PORT", 8080))
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
    )
