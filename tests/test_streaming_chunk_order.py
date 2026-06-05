from promptabi.streaming_chunk_order import (
    ChunkOrderFindingKind,
    StreamChunk,
    check_chunk_order,
    render_chunk_order_text,
)


def test_valid_stream_assembles_in_index_order():
    chunks = (
        StreamChunk(0, role="assistant"),
        StreamChunk(1, content_delta="Hel", content_index=0),
        StreamChunk(2, content_delta="lo", content_index=1),
        StreamChunk(3, finish_reason="stop"),
    )
    result = check_chunk_order(chunks)
    assert result.valid
    assert result.assembled == "Hello"


def test_reordered_seq_still_assembles_correctly():
    chunks = (
        StreamChunk(3, finish_reason="stop"),
        StreamChunk(1, content_delta="Hel", content_index=0),
        StreamChunk(0, role="assistant"),
        StreamChunk(2, content_delta="lo", content_index=1),
    )
    result = check_chunk_order(chunks)
    assert result.valid
    assert result.assembled == "Hello"


def test_content_after_finish_flagged():
    chunks = (
        StreamChunk(0, role="assistant"),
        StreamChunk(1, finish_reason="stop"),
        StreamChunk(2, content_delta="late", content_index=0),
    )
    result = check_chunk_order(chunks)
    kinds = {f.kind for f in result.findings}
    assert ChunkOrderFindingKind.CONTENT_AFTER_FINISH in kinds
    assert not result.valid


def test_role_not_first_flagged():
    chunks = (
        StreamChunk(0, content_delta="x", content_index=0),
        StreamChunk(1, role="assistant"),
    )
    result = check_chunk_order(chunks)
    kinds = {f.kind for f in result.findings}
    assert ChunkOrderFindingKind.ROLE_NOT_FIRST in kinds


def test_index_regression_flagged():
    chunks = (
        StreamChunk(0, role="assistant"),
        StreamChunk(1, content_delta="b", content_index=2),
        StreamChunk(2, content_delta="a", content_index=1),
    )
    result = check_chunk_order(chunks)
    kinds = {f.kind for f in result.findings}
    assert ChunkOrderFindingKind.INDEX_REGRESSION in kinds


def test_render_smoke():
    out = render_chunk_order_text(check_chunk_order(()))
    assert out.endswith("\n")
