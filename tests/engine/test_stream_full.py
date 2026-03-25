"""Tests for StreamChunk and stream_full() reasoning support."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock

from openjarvis.engine._stubs import (
    InferenceEngine,
    StreamChunk,
)


class TestStreamChunk:
    """Tests for the StreamChunk dataclass."""

    def test_default_fields(self) -> None:
        chunk = StreamChunk()
        assert chunk.content == ""
        assert chunk.reasoning is None
        assert chunk.finish_reason is None
        assert chunk.usage is None

    def test_content_only(self) -> None:
        chunk = StreamChunk(content="hello")
        assert chunk.content == "hello"
        assert chunk.reasoning is None

    def test_stream_chunk_reasoning(self) -> None:
        """StreamChunk carries reasoning alongside content."""
        chunk = StreamChunk(
            content="answer",
            reasoning="thinking step",
        )
        assert chunk.content == "answer"
        assert chunk.reasoning == "thinking step"

    def test_stream_chunk_finish(self) -> None:
        chunk = StreamChunk(
            content="",
            finish_reason="stop",
            usage={"prompt_tokens": 10, "completion_tokens": 5},
        )
        assert chunk.finish_reason == "stop"
        assert chunk.usage["prompt_tokens"] == 10


class TestDefaultStreamFull:
    """Tests for the default stream_full() on InferenceEngine."""

    def test_wraps_stream(self) -> None:
        """Default stream_full() wraps stream() tokens into
        StreamChunk with reasoning=None."""

        class DummyEngine(InferenceEngine):
            engine_id = "dummy"

            def generate(self, messages, **kw):
                return {"content": ""}

            async def stream(self, messages, **kw):
                for token in ["Hello", " ", "world"]:
                    yield token

            def list_models(self):
                return []

            def health(self):
                return True

        engine = DummyEngine()

        async def _run():
            chunks = []
            async for c in engine.stream_full(
                [], model="test", temperature=0.7
            ):
                chunks.append(c)
            return chunks

        chunks = asyncio.get_event_loop().run_until_complete(
            _run()
        )
        assert len(chunks) == 3
        assert chunks[0].content == "Hello"
        assert chunks[0].reasoning is None
        assert chunks[2].content == "world"


class TestOpenAICompatStreamFull:
    """Tests for _OpenAICompatibleEngine.stream_full() reasoning."""

    def test_openai_compat_reasoning(self) -> None:
        """SSE with delta.reasoning populates
        StreamChunk.reasoning."""
        from openjarvis.engine._openai_compat import (
            _OpenAICompatibleEngine,
        )

        # Build mock SSE lines
        chunks_data = [
            {
                "choices": [{
                    "delta": {"reasoning_content": "step 1"},
                    "finish_reason": None,
                }],
            },
            {
                "choices": [{
                    "delta": {"content": "answer"},
                    "finish_reason": None,
                }],
            },
            {
                "choices": [{
                    "delta": {},
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                },
            },
        ]
        sse_lines = []
        for c in chunks_data:
            sse_lines.append(f"data: {json.dumps(c)}")
        sse_lines.append("data: [DONE]")

        # Mock the httpx streaming response
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.iter_lines = MagicMock(
            return_value=iter(sse_lines)
        )
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=mock_resp)

        engine = _OpenAICompatibleEngine.__new__(
            _OpenAICompatibleEngine
        )
        engine._host = "http://test:8000"
        engine._api_prefix = "/v1"
        engine._client = mock_client
        engine.engine_id = "test"

        async def _run():
            result = []
            async for chunk in engine.stream_full(
                [], model="test-model", temperature=0.5
            ):
                result.append(chunk)
            return result

        result = asyncio.get_event_loop().run_until_complete(
            _run()
        )
        assert len(result) == 3
        # First chunk has reasoning
        assert result[0].reasoning == "step 1"
        assert result[0].content == ""
        # Second chunk has content
        assert result[1].content == "answer"
        assert result[1].reasoning is None
        # Third chunk has finish_reason
        assert result[2].finish_reason == "stop"

    def test_openai_compat_generate_reasoning(self) -> None:
        """generate() includes reasoning_content when present."""
        from openjarvis.engine._openai_compat import (
            _OpenAICompatibleEngine,
        )

        response_data = {
            "choices": [{
                "message": {
                    "content": "answer",
                    "reasoning_content": "my reasoning",
                },
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
            },
            "model": "test-model",
        }

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json = MagicMock(return_value=response_data)

        mock_client = MagicMock()
        mock_client.post = MagicMock(return_value=mock_resp)

        engine = _OpenAICompatibleEngine.__new__(
            _OpenAICompatibleEngine
        )
        engine._host = "http://test:8000"
        engine._api_prefix = "/v1"
        engine._client = mock_client
        engine.engine_id = "test"

        result = engine.generate(
            [], model="test-model", temperature=0.5
        )
        assert result["content"] == "answer"
        assert result["reasoning_content"] == "my reasoning"
