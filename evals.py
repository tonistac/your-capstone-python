import logging

import anthropic

log = logging.getLogger(__name__)

MODEL = "claude-haiku-4-5"

_JUDGE_TOOL = {
    "name": "submit_evaluation",
    "description": "Submit a structured quality evaluation for the response.",
    "input_schema": {
        "type": "object",
        "properties": {
            "score": {
                "type": "integer",
                "description": "Quality score: 1 = poor, 3 = acceptable, 5 = excellent.",
                "minimum": 1,
                "maximum": 5,
            },
            "reason": {
                "type": "string",
                "description": "One or two sentences explaining the score.",
            },
        },
        "required": ["score", "reason"],
    },
}

_SYSTEM = (
    "You are a strict judge evaluating research assistant responses. "
    "Score on accuracy, clarity, and completeness. Be consistent and critical."
)


def evaluate_response(response_text: str) -> dict:
    """Return {'score': int, 'reason': str} for the given response_text.

    Score is 0 with an ERROR reason if the evaluation itself fails.
    """
    try:
        client = anthropic.Anthropic()
        result = client.messages.create(
            model=MODEL,
            max_tokens=256,
            system=_SYSTEM,
            tools=[_JUDGE_TOOL],
            tool_choice={"type": "tool", "name": "submit_evaluation"},
            messages=[
                {
                    "role": "user",
                    "content": f"Evaluate this research assistant response:\n\n{response_text}",
                }
            ],
        )

        for block in result.content:
            if block.type == "tool_use" and block.name == "submit_evaluation":
                return {
                    "score": int(block.input["score"]),
                    "reason": block.input["reason"],
                }

        log.error("evaluate_response: no submit_evaluation block in response")
        return {"score": 0, "reason": "ERROR: Evaluator returned no structured output."}

    except Exception as exc:
        log.exception("evaluate_response failed")
        return {"score": 0, "reason": f"ERROR: {exc}"}
