import logging
import os

from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

log = logging.getLogger(__name__)

TABLE = "n8n_chat_histories"


def _supabase() -> Client:
    return create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])


def load_history(session_id: str, n: int = 5) -> list[dict]:
    """Return the last n messages for session_id in chronological order."""
    try:
        response = (
            _supabase()
            .table(TABLE)
            .select("role, content, created_at")
            .eq("session_id", str(session_id))
            .order("created_at", desc=True)
            .limit(n)
            .execute()
        )
        return list(reversed(response.data))
    except Exception as exc:
        log.exception("load_history failed for session %s", session_id)
        return []


def save_interaction(session_id: str, role: str, content: str) -> None:
    try:
        _supabase().table(TABLE).insert({
            "session_id": str(session_id),
            "role": role,
            "content": content,
        }).execute()
    except Exception as exc:
        log.exception("save_interaction failed for session %s", session_id)
