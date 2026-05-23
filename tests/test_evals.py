"""
Tests for evals.py

Coverage:
- evaluate_response: happy path, score/reason extraction, error path returns
  {'score': 0, 'reason': 'ERROR:...'}, no submit_evaluation block case,
  Anthropic API params (model, tool_choice), return type is always dict
"""

import pytest
from unittest.mock import MagicMock, patch


def _build_eval_response(score: int = 4, reason: str = "Good response."):
    """Helper: build a fake Anthropic response with submit_evaluation tool_use."""
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.name = "submit_evaluation"
    tool_block.input = {"score": score, "reason": reason}

    resp = MagicMock()
    resp.content = [tool_block]
    return resp


def _build_no_tool_response():
    """Helper: build an Anthropic response with only a text block (no tool call)."""
    text_block = MagicMock()
    text_block.type = "text"
    text_block.name = None  # text blocks have no 'name'

    resp = MagicMock()
    resp.content = [text_block]
    return resp


# ---------------------------------------------------------------------------
# evaluate_response
# ---------------------------------------------------------------------------

class TestEvaluateResponse:

    def test_returns_dict(self, patch_anthropic_evals):
        from evals import evaluate_response
        result = evaluate_response("Some response text.")
        assert isinstance(result, dict)

    def test_result_contains_score_key(self, patch_anthropic_evals):
        from evals import evaluate_response
        result = evaluate_response("Some response text.")
        assert "score" in result

    def test_result_contains_reason_key(self, patch_anthropic_evals):
        from evals import evaluate_response
        result = evaluate_response("Some response text.")
        assert "reason" in result

    def test_score_is_int(self, patch_anthropic_evals):
        from evals import evaluate_response
        result = evaluate_response("Some response text.")
        assert isinstance(result["score"], int)

    def test_reason_is_str(self, patch_anthropic_evals):
        from evals import evaluate_response
        result = evaluate_response("Some response text.")
        assert isinstance(result["reason"], str)

    def test_happy_path_score_extracted_correctly(self, patch_anthropic_evals):
        """The score from the tool block input must be returned as-is."""
        patch_anthropic_evals.messages.create.return_value = _build_eval_response(score=5, reason="Excellent.")
        from evals import evaluate_response
        result = evaluate_response("Great answer here.")
        assert result["score"] == 5
        assert result["reason"] == "Excellent."

    def test_score_values_within_range(self, patch_anthropic_evals):
        """Scores 1-5 must pass through unchanged."""
        from evals import evaluate_response
        for score in range(1, 6):
            patch_anthropic_evals.messages.create.return_value = \
                _build_eval_response(score=score, reason=f"Score {score}.")
            result = evaluate_response("text")
            assert result["score"] == score

    def test_score_is_coerced_to_int(self, patch_anthropic_evals):
        """evals.py does int(block.input['score']) so a string '3' becomes 3."""
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.name = "submit_evaluation"
        tool_block.input = {"score": "3", "reason": "Acceptable."}
        resp = MagicMock()
        resp.content = [tool_block]
        patch_anthropic_evals.messages.create.return_value = resp

        from evals import evaluate_response
        result = evaluate_response("text")
        assert result["score"] == 3
        assert isinstance(result["score"], int)

    def test_no_submit_evaluation_block_returns_zero_score(self):
        """
        When the model returns content but no submit_evaluation tool_use block,
        the fallback {'score': 0, 'reason': 'ERROR: ...'} must be returned.
        """
        client = MagicMock()
        client.messages.create.return_value = _build_no_tool_response()
        with patch("evals.anthropic.Anthropic", return_value=client):
            from evals import evaluate_response
            result = evaluate_response("text")
        assert result["score"] == 0
        assert "ERROR" in result["reason"]

    def test_api_exception_returns_error_dict(self):
        """Any exception must be caught and returned as {'score': 0, 'reason': 'ERROR:...'}."""
        client = MagicMock()
        client.messages.create.side_effect = RuntimeError("API unavailable")
        with patch("evals.anthropic.Anthropic", return_value=client):
            from evals import evaluate_response
            result = evaluate_response("text")
        assert result["score"] == 0
        assert result["reason"].startswith("ERROR:")
        assert "API unavailable" in result["reason"]

    def test_error_dict_has_correct_keys(self):
        client = MagicMock()
        client.messages.create.side_effect = Exception("oops")
        with patch("evals.anthropic.Anthropic", return_value=client):
            from evals import evaluate_response
            result = evaluate_response("text")
        assert set(result.keys()) >= {"score", "reason"}

    def test_uses_haiku_model(self, patch_anthropic_evals):
        """The judge must use claude-haiku, not claude-sonnet."""
        from evals import evaluate_response, MODEL
        evaluate_response("text")
        create_kwargs = patch_anthropic_evals.messages.create.call_args.kwargs
        assert create_kwargs["model"] == MODEL

    def test_tool_choice_forces_submit_evaluation(self, patch_anthropic_evals):
        """tool_choice must pin the model to the submit_evaluation tool."""
        from evals import evaluate_response
        evaluate_response("text")
        create_kwargs = patch_anthropic_evals.messages.create.call_args.kwargs
        tc = create_kwargs["tool_choice"]
        assert tc["type"] == "tool"
        assert tc["name"] == "submit_evaluation"

    def test_response_text_included_in_user_message(self, patch_anthropic_evals):
        """The response being evaluated must appear in the messages payload."""
        from evals import evaluate_response
        evaluate_response("The unique canary string 9f3a.")
        create_kwargs = patch_anthropic_evals.messages.create.call_args.kwargs
        messages = create_kwargs["messages"]
        combined = " ".join(str(m["content"]) for m in messages)
        assert "The unique canary string 9f3a." in combined

    def test_never_raises(self):
        """evaluate_response must never propagate an exception to the caller."""
        client = MagicMock()
        client.messages.create.side_effect = Exception("total failure")
        with patch("evals.anthropic.Anthropic", return_value=client):
            from evals import evaluate_response
            try:
                evaluate_response("text")
            except Exception:
                pytest.fail("evaluate_response raised instead of returning an error dict")
