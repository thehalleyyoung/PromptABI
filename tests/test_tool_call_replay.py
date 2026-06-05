from promptabi.tool_call_replay import (
    ReplayTurn,
    ToolCall,
    ToolReplayFindingKind,
    ToolResult,
    render_tool_replay_text,
    replay_tool_calls,
)


def test_valid_call_then_result():
    turns = (
        ReplayTurn(0, calls=(ToolCall("a", "search"),)),
        ReplayTurn(1, results=(ToolResult("a"),)),
    )
    result = replay_tool_calls(turns)
    assert result.valid
    assert result.findings == ()


def test_unanswered_call_flagged():
    turns = (ReplayTurn(0, calls=(ToolCall("a", "search"),)),)
    result = replay_tool_calls(turns)
    kinds = {f.kind for f in result.findings}
    assert ToolReplayFindingKind.UNANSWERED_CALL in kinds


def test_result_before_call_and_double_answer():
    turns = (
        ReplayTurn(0, results=(ToolResult("x"),)),
        ReplayTurn(1, calls=(ToolCall("y", "f"),)),
        ReplayTurn(2, results=(ToolResult("y"), ToolResult("y"))),
    )
    result = replay_tool_calls(turns)
    kinds = {f.kind for f in result.findings}
    assert ToolReplayFindingKind.RESULT_BEFORE_CALL in kinds
    assert ToolReplayFindingKind.DOUBLE_ANSWERED in kinds


def test_duplicate_call_id():
    turns = (
        ReplayTurn(0, calls=(ToolCall("a", "f"), ToolCall("a", "g"))),
        ReplayTurn(1, results=(ToolResult("a"),)),
    )
    result = replay_tool_calls(turns)
    kinds = {f.kind for f in result.findings}
    assert ToolReplayFindingKind.DUPLICATE_CALL_ID in kinds


def test_render_smoke():
    out = render_tool_replay_text(replay_tool_calls(()))
    assert out.endswith("\n")
