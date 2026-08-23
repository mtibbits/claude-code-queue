"""
Tests for SessionStats dataclass and session stats extraction from JSONL logs.

Test IDs use the SS- prefix for cross-reference.
"""

import json
from unittest.mock import patch

import pytest

from claude_code_queue.models import (
    SessionStats,
    QueuedPrompt,
    PromptStatus,
    ExecutionResult,
    RateLimitInfo,
)

SESSION_ID = "11111111-2222-4333-8444-555555555555"
OTHER_SESSION_ID = "99999999-8888-4777-8666-555555555555"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_assistant_line(
    input_tokens=10,
    output_tokens=20,
    cache_creation=100,
    cache_read=200,
    message_id="msg_default",
):
    """Build a single JSONL assistant line with the given usage values."""
    return json.dumps({
        "type": "assistant",
        "message": {
            "id": message_id,
            "role": "assistant",
            "content": [{"type": "text", "text": "hello"}],
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cache_creation_input_tokens": cache_creation,
                "cache_read_input_tokens": cache_read,
            },
        },
    })


def _make_user_line():
    """Build a JSONL user line (should be ignored by stats extraction)."""
    return json.dumps({
        "type": "user",
        "message": {"role": "user", "content": "say hello"},
    })


def _make_queue_op_line():
    """Build a JSONL queue-operation line (should be ignored)."""
    return json.dumps({
        "type": "queue-operation",
        "operation": "enqueue",
        "timestamp": "2026-03-15T12:00:00.000Z",
    })


def _make_last_prompt_line():
    """Build a JSONL last-prompt line (should be ignored)."""
    return json.dumps({
        "type": "last-prompt",
        "lastPrompt": "say hello",
    })


def _write_jsonl(path, lines):
    """Write JSONL lines to a file and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for line in lines:
            f.write(line + "\n")
    return path


def _setup_jsonl(tmp_path, lines, session_id=SESSION_ID, project="project"):
    """Create one Claude session JSONL file in a test profile."""
    jsonl_file = tmp_path / "profile" / "projects" / project / f"{session_id}.jsonl"
    _write_jsonl(jsonl_file, lines)
    return jsonl_file


def _make_stats_prompt(tmp_path):
    """Create a prompt used to capture best-effort extraction warnings."""
    return QueuedPrompt(id="abc12345", content="test", working_directory=str(tmp_path))


def _use_test_profile(mocker, tmp_path):
    return mocker.patch(
        "claude_code_queue.queue_manager.claude_config_dir",
        return_value=tmp_path / "profile",
    )


# ===========================================================================
# SessionStats — basic properties
# ===========================================================================


def test_session_stats_defaults_are_zero():  # SS-001
    stats = SessionStats()
    assert stats.input_tokens == 0
    assert stats.output_tokens == 0
    assert stats.cache_creation_input_tokens == 0
    assert stats.cache_read_input_tokens == 0
    assert stats.api_turns == 0


def test_session_stats_total_input_sums_all_three():  # SS-002
    stats = SessionStats(
        input_tokens=10,
        cache_creation_input_tokens=100,
        cache_read_input_tokens=200,
    )
    assert stats.total_input_tokens == 310


def test_session_stats_total_input_zero_when_all_zero():  # SS-003
    stats = SessionStats()
    assert stats.total_input_tokens == 0


# ===========================================================================
# _extract_session_stats()
# ===========================================================================


def test_extract_stats_single_turn(manager, tmp_path, mocker):  # SS-010
    _use_test_profile(mocker, tmp_path)
    prompt = _make_stats_prompt(tmp_path)
    _setup_jsonl(tmp_path, [
        _make_user_line(),
        _make_assistant_line(input_tokens=5, output_tokens=50, cache_creation=1000, cache_read=2000),
    ])

    stats = manager._extract_session_stats(prompt, SESSION_ID)

    assert stats is not None
    assert stats.input_tokens == 5
    assert stats.output_tokens == 50
    assert stats.cache_creation_input_tokens == 1000
    assert stats.cache_read_input_tokens == 2000
    assert stats.total_input_tokens == 3005
    assert stats.api_turns == 1


def test_extract_stats_multi_turn(manager, tmp_path, mocker):  # SS-011
    _use_test_profile(mocker, tmp_path)
    prompt = _make_stats_prompt(tmp_path)
    _setup_jsonl(tmp_path, [
        _make_user_line(),
        _make_assistant_line(input_tokens=3, output_tokens=100, cache_creation=5000, cache_read=8000, message_id="msg_1"),
        _make_user_line(),
        _make_assistant_line(input_tokens=1, output_tokens=200, cache_creation=5000, cache_read=8000, message_id="msg_2"),
        _make_user_line(),
        _make_assistant_line(input_tokens=1, output_tokens=150, cache_creation=0, cache_read=10000, message_id="msg_3"),
    ])

    stats = manager._extract_session_stats(prompt, SESSION_ID)

    assert stats is not None
    assert stats.input_tokens == 5
    assert stats.output_tokens == 450
    assert stats.cache_creation_input_tokens == 10000
    assert stats.cache_read_input_tokens == 26000
    assert stats.total_input_tokens == 36005
    assert stats.api_turns == 3


def test_extract_stats_non_assistant_lines_ignored(manager, tmp_path, mocker):  # SS-012
    _use_test_profile(mocker, tmp_path)
    prompt = _make_stats_prompt(tmp_path)
    _setup_jsonl(tmp_path, [
        _make_queue_op_line(),
        _make_user_line(),
        _make_assistant_line(input_tokens=3, output_tokens=10, cache_creation=100, cache_read=200),
        _make_last_prompt_line(),
    ])

    stats = manager._extract_session_stats(prompt, SESSION_ID)

    assert stats is not None
    assert stats.input_tokens == 3
    assert stats.output_tokens == 10
    assert stats.api_turns == 1


def test_extract_stats_missing_usage_block_is_skipped(manager, tmp_path, mocker):  # SS-013
    _use_test_profile(mocker, tmp_path)
    prompt = _make_stats_prompt(tmp_path)
    line_no_usage = json.dumps({
        "type": "assistant",
        "message": {
            "id": "msg_missing_usage",
            "role": "assistant",
            "content": [{"type": "text", "text": "hi"}],
        },
    })
    _setup_jsonl(tmp_path, [
        line_no_usage,
        _make_assistant_line(input_tokens=5, output_tokens=10, cache_creation=100, cache_read=200),
    ])

    stats = manager._extract_session_stats(prompt, SESSION_ID)

    assert stats is not None
    assert stats.api_turns == 1
    assert stats.input_tokens == 5
    assert stats.output_tokens == 10


def test_extract_stats_malformed_line_skipped(manager, tmp_path, mocker):  # SS-014
    _use_test_profile(mocker, tmp_path)
    prompt = _make_stats_prompt(tmp_path)
    _setup_jsonl(tmp_path, [
        "this is not json",
        "null",
        "[]",
        json.dumps({"type": "assistant", "message": []}),
        json.dumps({"type": "assistant", "message": {"id": "bad", "usage": []}}),
        _make_assistant_line(input_tokens=7, output_tokens=30, cache_creation=500, cache_read=600),
        "{bad json",
    ])

    stats = manager._extract_session_stats(prompt, SESSION_ID)

    assert stats is not None
    assert stats.input_tokens == 7
    assert stats.output_tokens == 30
    assert stats.api_turns == 1


@pytest.mark.parametrize("value", ["10", 1.5, True, -1, None, [], {}])
def test_extract_stats_invalid_token_value_is_skipped(
    manager, tmp_path, mocker, value
):  # SS-015
    _use_test_profile(mocker, tmp_path)
    prompt = _make_stats_prompt(tmp_path)
    _setup_jsonl(tmp_path, [
        _make_assistant_line(input_tokens=value, message_id="msg_invalid"),
        _make_assistant_line(input_tokens=7, output_tokens=8, cache_creation=9, cache_read=10, message_id="msg_valid"),
    ])

    stats = manager._extract_session_stats(prompt, SESSION_ID)

    assert stats is not None
    assert stats.api_turns == 1
    assert stats.total_input_tokens == 26


def test_extract_stats_empty_file(manager, tmp_path, mocker):  # SS-016
    _use_test_profile(mocker, tmp_path)
    prompt = _make_stats_prompt(tmp_path)
    _setup_jsonl(tmp_path, [])

    stats = manager._extract_session_stats(prompt, SESSION_ID)
    assert stats is None


def test_extract_stats_directory_missing(manager, tmp_path, mocker):  # SS-017
    _use_test_profile(mocker, tmp_path)
    prompt = _make_stats_prompt(tmp_path)

    stats = manager._extract_session_stats(prompt, SESSION_ID)
    assert stats is None


def test_extract_stats_uses_active_profile(manager, tmp_path, mocker):  # SS-018
    _use_test_profile(mocker, tmp_path)
    prompt = _make_stats_prompt(tmp_path)
    _setup_jsonl(tmp_path, [
        _make_assistant_line(input_tokens=1, output_tokens=2, cache_creation=3, cache_read=4),
    ])

    stats = manager._extract_session_stats(prompt, SESSION_ID)

    assert stats is not None
    assert stats.total_input_tokens == 8


@pytest.mark.parametrize("session_id", [None, "", "../../victim", "NOT-A-UUID"])
def test_extract_stats_rejects_unknown_or_invalid_session_id(
    manager, tmp_path, mocker, session_id
):  # SS-019
    _use_test_profile(mocker, tmp_path)
    prompt = _make_stats_prompt(tmp_path)
    _setup_jsonl(tmp_path, [_make_assistant_line()])

    stats = manager._extract_session_stats(prompt, session_id)
    assert stats is None


def test_extract_stats_ignores_newer_unrelated_session(manager, tmp_path, mocker):  # SS-020
    _use_test_profile(mocker, tmp_path)
    prompt = _make_stats_prompt(tmp_path)
    _setup_jsonl(tmp_path, [
        _make_assistant_line(input_tokens=1, output_tokens=2, cache_creation=3, cache_read=4),
    ])
    _setup_jsonl(tmp_path, [
        _make_assistant_line(input_tokens=999, output_tokens=999, cache_creation=999, cache_read=999),
    ], session_id=OTHER_SESSION_ID, project="other-project")

    stats = manager._extract_session_stats(prompt, SESSION_ID)

    assert stats is not None
    assert stats.input_tokens == 1
    assert stats.output_tokens == 2


def test_extract_stats_deduplicates_streamed_assistant_events(
    manager, tmp_path, mocker
):
    _use_test_profile(mocker, tmp_path)
    prompt = _make_stats_prompt(tmp_path)
    _setup_jsonl(tmp_path, [
        _make_assistant_line(input_tokens=2, output_tokens=4, cache_creation=100, cache_read=200, message_id="msg_streamed"),
        _make_assistant_line(input_tokens=2, output_tokens=809, cache_creation=100, cache_read=200, message_id="msg_streamed"),
        _make_assistant_line(input_tokens=0, output_tokens=0, cache_creation=0, cache_read=0, message_id="msg_streamed"),
    ])

    stats = manager._extract_session_stats(prompt, SESSION_ID)

    assert stats is not None
    assert stats.api_turns == 1
    assert stats.input_tokens == 2
    assert stats.output_tokens == 809
    assert stats.cache_creation_input_tokens == 100
    assert stats.cache_read_input_tokens == 200


def test_extract_stats_exception_returns_none(manager, tmp_path):  # SS-021
    prompt = _make_stats_prompt(tmp_path)
    with patch.object(manager, "_do_extract_session_stats", side_effect=OSError("boom")):
        stats = manager._extract_session_stats(prompt, SESSION_ID)
    assert stats is None
    assert "session stats extraction failed" in prompt.execution_log


# ===========================================================================
# _format_stats_line()
# ===========================================================================


def test_format_stats_line_with_stats(manager):  # SS-030
    stats = SessionStats(
        input_tokens=100,
        output_tokens=500,
        cache_creation_input_tokens=10000,
        cache_read_input_tokens=5000,
        api_turns=3,
    )
    line = manager._format_stats_line(154.0, stats)
    assert "Duration: 2m" in line
    assert "Input: 15,100 tokens" in line
    assert "Output: 500 tokens" in line
    assert line.startswith("    ")


def test_format_stats_line_without_stats(manager):  # SS-031
    line = manager._format_stats_line(45.0, None)
    assert "Duration: 45s" in line
    assert "Input" not in line
    assert "Output" not in line
    assert line.startswith("    ")


def test_format_stats_line_pipe_separators(manager):  # SS-032
    stats = SessionStats(input_tokens=1, output_tokens=2)
    line = manager._format_stats_line(10.0, stats)
    assert " | " in line


# ===========================================================================
# _log_session_stats()
# ===========================================================================


def test_log_session_stats_detailed_breakdown(manager):  # SS-050
    prompt = QueuedPrompt(id="abc12345", content="test")
    stats = SessionStats(
        input_tokens=402,
        output_tokens=51568,
        cache_creation_input_tokens=19093602,
        cache_read_input_tokens=4255901,
        api_turns=297,
    )

    manager._log_session_stats(prompt, stats)

    assert "402 input" in prompt.execution_log
    assert "19,093,602 cache-write" in prompt.execution_log
    assert "4,255,901 cache-read" in prompt.execution_log
    assert "23,349,905 total input" in prompt.execution_log
    assert "51,568 output" in prompt.execution_log
    assert "297 API turns" in prompt.execution_log


def test_log_session_stats_none_no_log(manager):  # SS-051
    prompt = QueuedPrompt(id="abc12345", content="test")
    manager._log_session_stats(prompt, None)
    assert "Token usage" not in prompt.execution_log


def test_log_session_stats_single_turn_singular(manager):  # SS-052
    prompt = QueuedPrompt(id="abc12345", content="test")
    stats = SessionStats(input_tokens=1, output_tokens=2, api_turns=1)
    manager._log_session_stats(prompt, stats)
    assert "1 API turn)" in prompt.execution_log


# ===========================================================================
# Integration: stats printed in _process_execution_result()
# ===========================================================================


def test_result_success_prints_stats(manager, tmp_path, mocker, capsys):  # SS-040
    _use_test_profile(mocker, tmp_path)
    manager.state = manager.storage.load_queue_state()
    prompt = _make_stats_prompt(tmp_path)
    prompt.status = PromptStatus.EXECUTING
    manager.state.add_prompt(prompt)
    _setup_jsonl(tmp_path, [
        _make_assistant_line(input_tokens=5, output_tokens=50, cache_creation=1000, cache_read=2000),
    ])
    result = ExecutionResult(
        success=True,
        output="done",
        execution_time=120.5,
        session_id=SESSION_ID,
    )

    manager._process_execution_result(prompt, result)

    captured = capsys.readouterr().out
    assert "completed successfully" in captured
    assert "Duration:" in captured
    assert "Input: 3,005 tokens" in captured
    assert "Output: 50 tokens" in captured


def test_result_success_no_jsonl_prints_duration_only(manager, tmp_path, mocker, capsys):  # SS-041
    _use_test_profile(mocker, tmp_path)
    manager.state = manager.storage.load_queue_state()
    prompt = _make_stats_prompt(tmp_path)
    prompt.status = PromptStatus.EXECUTING
    manager.state.add_prompt(prompt)
    result = ExecutionResult(success=True, output="done", execution_time=30.0)

    manager._process_execution_result(prompt, result)

    captured = capsys.readouterr().out
    assert "Duration: 30s" in captured
    assert "Input" not in captured


def test_result_rate_limited_prints_stats_before_cleanup(manager, tmp_path, mocker, capsys):  # SS-042
    """Stats must be extracted BEFORE cleanup deletes the JSONL."""
    _use_test_profile(mocker, tmp_path)
    manager.state = manager.storage.load_queue_state()
    prompt = _make_stats_prompt(tmp_path)
    prompt.status = PromptStatus.EXECUTING
    manager.state.add_prompt(prompt)
    jsonl_file = _setup_jsonl(tmp_path, [
        _make_assistant_line(input_tokens=3, output_tokens=10, cache_creation=500, cache_read=600),
    ])

    rate_info = RateLimitInfo(
        is_rate_limited=True,
        limit_message="usage limit reached",
    )
    result = ExecutionResult(
        success=False,
        output="",
        error="rate limited",
        rate_limit_info=rate_info,
        execution_time=5.0,
        session_id=SESSION_ID,
    )

    manager._process_execution_result(prompt, result)

    captured = capsys.readouterr().out
    assert "rate limited" in captured
    assert "Input: 1,103 tokens" in captured
    assert "Output: 10 tokens" in captured
    assert not jsonl_file.exists()


def test_result_generic_failure_retry_prints_stats(manager, tmp_path, mocker, capsys):  # SS-043
    _use_test_profile(mocker, tmp_path)
    manager.state = manager.storage.load_queue_state()
    prompt = _make_stats_prompt(tmp_path)
    prompt.status = PromptStatus.EXECUTING
    manager.state.add_prompt(prompt)
    _setup_jsonl(tmp_path, [
        _make_assistant_line(input_tokens=2, output_tokens=30, cache_creation=100, cache_read=200),
    ])
    result = ExecutionResult(
        success=False,
        output="",
        error="something broke",
        execution_time=10.0,
        session_id=SESSION_ID,
    )

    manager._process_execution_result(prompt, result)

    captured = capsys.readouterr().out
    assert "failed" in captured
    assert "Input: 302 tokens" in captured
    assert "Output: 30 tokens" in captured


def test_result_generic_failure_permanent_prints_stats(manager, tmp_path, mocker, capsys):  # SS-044
    _use_test_profile(mocker, tmp_path)
    manager.state = manager.storage.load_queue_state()
    prompt = _make_stats_prompt(tmp_path)
    prompt.status = PromptStatus.EXECUTING
    prompt.max_retries = 1
    prompt.retry_count = 1
    manager.state.add_prompt(prompt)
    _setup_jsonl(tmp_path, [
        _make_assistant_line(input_tokens=1, output_tokens=5, cache_creation=50, cache_read=100),
    ])
    result = ExecutionResult(
        success=False,
        output="",
        error="something broke",
        execution_time=8.0,
        session_id=SESSION_ID,
    )

    manager._process_execution_result(prompt, result)

    captured = capsys.readouterr().out
    assert "failed permanently" in captured
    assert "Input: 151 tokens" in captured
    assert "Output: 5 tokens" in captured


def test_result_non_retryable_prints_stats(manager, tmp_path, mocker, capsys):  # SS-045
    _use_test_profile(mocker, tmp_path)
    manager.state = manager.storage.load_queue_state()
    prompt = _make_stats_prompt(tmp_path)
    prompt.status = PromptStatus.EXECUTING
    manager.state.add_prompt(prompt)
    _setup_jsonl(tmp_path, [
        _make_assistant_line(input_tokens=1, output_tokens=1, cache_creation=1, cache_read=1),
    ])
    result = ExecutionResult(
        success=False,
        output="",
        error="nested session",
        execution_time=1.0,
        is_non_retryable=True,
        session_id=SESSION_ID,
    )

    manager._process_execution_result(prompt, result)

    captured = capsys.readouterr().out
    assert "non-retryable" in captured
    assert "Input: 3 tokens" in captured
    assert "Output: 1 tokens" in captured
    assert "Duration: 1s" in captured
