import logging

import anthropic

from memory import load_history, save_interaction
from tools import TOOL_DEFINITIONS, execute_tool

log = logging.getLogger(__name__)

MODEL = "claude-sonnet-4-6"
MAX_ITERATIONS = 10
SYSTEM = "You are a research assistant. Be concise, cite sources, and reason step by step."


def _text(content: list) -> str:
    return " ".join(b.text for b in content if hasattr(b, "text")).strip()


def run_agent(query: str, session_id: str) -> str:
    session_id = str(session_id)
    client = anthropic.Anthropic(timeout=60.0)

    save_interaction(session_id, "user", query)

    raw_history = load_history(session_id)
    messages = [
        {"role": r["role"], "content": r["content"]} for r in raw_history
    ]

    response = None
    for iteration in range(MAX_ITERATIONS):
        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=SYSTEM,
            tools=TOOL_DEFINITIONS,
            messages=messages,
        )

        log.info(
            "iter=%d stop=%s in_tok=%d out_tok=%d",
            iteration + 1,
            response.stop_reason,
            response.usage.input_tokens,
            response.usage.output_tokens,
        )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            break

        tool_results = [
            {
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": execute_tool(block.name, block.input),
            }
            for block in response.content
            if block.type == "tool_use"
        ]

        messages.append({"role": "user", "content": tool_results})
    else:
        log.warning(
            "max_iterations=%d reached for session=%s", MAX_ITERATIONS, session_id
        )

    if response is None:
        return "ERROR: No response from model."

    final = _text(response.content)
    if not final:
        final = "ERROR: Agent reached max iterations without a final text response."

    save_interaction(session_id, "assistant", final)
    return final
