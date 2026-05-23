import logging
import os
from typing import Any

from dotenv import load_dotenv
from pinecone import Pinecone
from supabase import create_client, Client

TABLE = "n8n_chat_histories"

load_dotenv()

log = logging.getLogger(__name__)

# --- Tool definitions (passed to the Claude API) ---

TOOL_DEFINITIONS: list[dict] = [
    {
        "type": "web_search_20250305",
        "name": "web_search",
    },
    {
        "type": "web_fetch_20260309",
        "name": "web_fetch",
    },
    {
        "name": "pinecone_search",
        "description": "Search the knowledge base for relevant context documents.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language search query.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to return.",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "save_to_memory",
        "description": "Persist a message or insight to long-term memory in Supabase.",
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "Session identifier (must be a string).",
                },
                "role": {
                    "type": "string",
                    "enum": ["user", "assistant"],
                    "description": "Author role.",
                },
                "content": {
                    "type": "string",
                    "description": "Text to save.",
                },
            },
            "required": ["session_id", "role", "content"],
        },
    },
]

# --- Lazy clients (built on first use to avoid import-time side effects) ---

def _pinecone_index():
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    return pc.Index(os.environ["PINECONE_INDEX"])


def _supabase() -> Client:
    return create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])


# --- Executors ---

def execute_pinecone_search(query: str, top_k: int = 5) -> str:
    try:
        results = _pinecone_index().search(
            namespace="",
            query={"top_k": top_k, "inputs": {"text": query}},
        )
        hits = results.get("result", {}).get("hits", [])
        if not hits:
            return "No relevant documents found."
        lines = [
            f"[{i}] (score={h['_score']:.3f}) {h.get('fields', {}).get('text', '')}"
            for i, h in enumerate(hits, 1)
        ]
        return "\n\n".join(lines)
    except Exception as exc:
        log.exception("pinecone_search failed")
        return f"ERROR: {exc}"


def execute_save_to_memory(session_id: str, role: str, content: str) -> str:
    try:
        _supabase().table(TABLE).insert({
            "session_id": str(session_id),
            "role": role,
            "content": content,
        }).execute()
        return "Saved to memory."
    except Exception as exc:
        log.exception("save_to_memory failed")
        return f"ERROR: {exc}"


# --- Dispatcher (called by agent.py for custom tool_use blocks) ---
# web_search and web_fetch are built-in tools executed server-side by Anthropic;
# they never reach this dispatcher.

def execute_tool(name: str, tool_input: dict[str, Any]) -> str:
    if name == "pinecone_search":
        return execute_pinecone_search(**tool_input)
    if name == "save_to_memory":
        return execute_save_to_memory(**tool_input)
    return f"ERROR: Unknown tool '{name}'"
