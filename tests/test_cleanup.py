"""Integration tests for rate-limit artifact cleanup dispatch."""

from claude_code_queue.models import (
    ExecutionResult,
    PromptStatus,
    QueuedPrompt,
    QueueState,
    RateLimitInfo,
)


SESSION_UUID = "00134021-1e30-4928-b9af-e92a676ab248"


def _rate_limit_result() -> ExecutionResult:
    return ExecutionResult(
        success=False,
        output="usage limit reached",
        error="",
        rate_limit_info=RateLimitInfo(is_rate_limited=True, reset_time=None),
        execution_time=0.1,
        session_id=SESSION_UUID,
    )


def test_cleanup_not_called_on_success(manager, mocker):  # CLN-010
    prompt = QueuedPrompt(content="task")
    manager.state = manager.storage.load_queue_state()
    manager.state.add_prompt(prompt)
    spy = mocker.patch.object(manager, "_cleanup_rate_limit_artifacts")
    mocker.patch.object(
        manager.claude_interface,
        "execute_prompt",
        return_value=ExecutionResult(
            success=True,
            output="done",
            error="",
            execution_time=0.1,
            session_id=SESSION_UUID,
        ),
    )

    manager._execute_prompt(prompt)

    spy.assert_not_called()


def test_cleanup_not_called_on_generic_failure(manager, mocker):  # CLN-011
    prompt = QueuedPrompt(content="task", max_retries=3)
    manager.state = manager.storage.load_queue_state()
    manager.state.add_prompt(prompt)
    spy = mocker.patch.object(manager, "_cleanup_rate_limit_artifacts")
    mocker.patch.object(
        manager.claude_interface,
        "execute_prompt",
        return_value=ExecutionResult(
            success=False,
            output="",
            error="oops",
            execution_time=0.1,
            session_id=SESSION_UUID,
        ),
    )

    manager._execute_prompt(prompt)

    spy.assert_not_called()


def test_cleanup_called_on_rate_limit_with_session_id(manager, mocker):  # CLN-012
    prompt = QueuedPrompt(content="task", max_retries=3)
    manager.state = manager.storage.load_queue_state()
    manager.state.add_prompt(prompt)
    spy = mocker.patch.object(manager, "_cleanup_rate_limit_artifacts")
    mocker.patch.object(
        manager.claude_interface,
        "execute_prompt",
        return_value=_rate_limit_result(),
    )

    manager._execute_prompt(prompt)

    spy.assert_called_once_with(prompt, SESSION_UUID)


def test_cleanup_exception_does_not_break_result_processing(manager, mocker):  # CLN-015
    prompt = QueuedPrompt(content="task", max_retries=3)
    manager.state = QueueState(prompts=[prompt])
    mocker.patch.object(
        manager,
        "_do_cleanup_rate_limit_artifacts",
        side_effect=RuntimeError("disk unavailable"),
    )

    manager._process_execution_result(prompt, _rate_limit_result())

    assert prompt.status == PromptStatus.RATE_LIMITED
    assert manager.state.last_processed is not None
    assert "artifact cleanup failed" in prompt.execution_log
