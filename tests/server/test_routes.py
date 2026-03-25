"""Tests for the API server routes."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from openjarvis.server.app import create_app  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_engine(content="Hello from server", models=None):
    engine = MagicMock()
    engine.engine_id = "mock"
    engine.health.return_value = True
    engine.list_models.return_value = models or ["test-model"]
    engine.generate.return_value = {
        "content": content,
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        "model": "test-model",
        "finish_reason": "stop",
    }

    # Set up async stream
    async def mock_stream(
        messages,
        *,
        model,
        temperature=0.7,
        max_tokens=1024,
        **kwargs,
    ):
        for token in ["Hello", " ", "world"]:
            yield token

    engine.stream = mock_stream
    return engine


def _make_agent(content="Hello from agent"):
    from openjarvis.agents._stubs import AgentResult

    agent = MagicMock()
    agent.agent_id = "mock"
    agent.run.return_value = AgentResult(content=content, turns=1)
    return agent


def _make_streaming_agent(
    *,
    soul_content="You are Jarvis, a helpful assistant.",
    tools=None,
    max_turns=10,
):
    """Create a mock agent with PromptBuilder, executor, and max_turns.

    Used by tests for ``_handle_agent_stream_real()`` which reads
    ``agent._prompt_builder``, ``agent._executor``, and ``agent._max_turns``.
    """
    from openjarvis.agents._stubs import AgentResult

    agent = MagicMock()
    agent.agent_id = "streaming-mock"
    agent.run.return_value = AgentResult(content="fallback", turns=1)
    agent._model = "test-model"
    agent._max_turns = max_turns

    # PromptBuilder mock
    pb = MagicMock()
    pb.build.return_value = soul_content
    agent._prompt_builder = pb

    # Executor mock
    executor = MagicMock()
    executor.get_openai_tools.return_value = tools or []
    agent._executor = executor

    return agent


def _make_streaming_engine(tokens=None):
    """Create a mock engine with ``stream_full()`` for agent streaming tests.

    Returns an engine whose ``stream_full()`` yields StreamChunk objects
    for the given token list (defaults to ["Hello", " ", "world"]).
    """
    from openjarvis.engine._stubs import StreamChunk

    engine = _make_engine()

    tok_list = tokens if tokens is not None else ["Hello", " ", "world"]

    async def mock_stream_full(messages, *, model, **kwargs):
        for token in tok_list:
            yield StreamChunk(content=token)
        yield StreamChunk(finish_reason="stop")

    engine.stream_full = mock_stream_full
    return engine


@pytest.fixture
def client():
    engine = _make_engine()
    app = create_app(engine, "test-model")
    return TestClient(app)


@pytest.fixture
def client_with_agent():
    engine = _make_engine()
    agent = _make_agent()
    app = create_app(engine, "test-model", agent=agent)
    return TestClient(app)


# ---------------------------------------------------------------------------
# Chat completions tests
# ---------------------------------------------------------------------------


class TestChatCompletions:
    def test_basic_completion(self, client):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "chat.completion"
        assert data["choices"][0]["message"]["content"] == "Hello from server"

    def test_completion_has_usage(self, client):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        data = resp.json()
        assert data["usage"]["total_tokens"] == 8

    def test_completion_has_id(self, client):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        data = resp.json()
        assert data["id"].startswith("chatcmpl-")

    def test_custom_temperature(self, client):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "temperature": 0.1,
            },
        )
        assert resp.status_code == 200

    def test_with_system_message(self, client):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [
                    {"role": "system", "content": "Be helpful"},
                    {"role": "user", "content": "Hello"},
                ],
            },
        )
        assert resp.status_code == 200

    def test_with_tools(self):
        engine = _make_engine()
        engine.generate.return_value = {
            "content": "",
            "tool_calls": [
                {"id": "c1", "name": "calc", "arguments": '{"expr":"2+2"}'},
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
            "model": "test-model",
            "finish_reason": "tool_calls",
        }
        app = create_app(engine, "test-model")
        client = TestClient(app)
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Calc"}],
                "tools": [{"type": "function", "function": {"name": "calc"}}],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["choices"][0]["message"]["tool_calls"] is not None

    def test_agent_mode(self, client_with_agent):
        resp = client_with_agent.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["choices"][0]["message"]["content"] == "Hello from agent"

    def test_agent_with_conversation(self, client_with_agent):
        resp = client_with_agent.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [
                    {"role": "system", "content": "Be helpful"},
                    {"role": "user", "content": "Hello"},
                ],
            },
        )
        assert resp.status_code == 200

    def test_streaming(self, client):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
            },
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")
        # Parse SSE events
        lines = resp.text.strip().split("\n")
        data_lines = [ln for ln in lines if ln.startswith("data:")]
        assert len(data_lines) > 0
        # Last should be [DONE]
        assert data_lines[-1].strip() == "data: [DONE]"

    def test_streaming_content(self, client):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
            },
        )
        # Collect content tokens from stream
        content = ""
        for line in resp.text.strip().split("\n"):
            if line.startswith("data:") and "[DONE]" not in line:
                data = json.loads(line[5:].strip())
                choices = data.get("choices", [{}])
                delta_content = (
                    choices[0]
                    .get(
                        "delta",
                        {},
                    )
                    .get("content")
                )
                if delta_content:
                    content += delta_content
        assert content == "Hello world"

    def test_finish_reason_default(self, client):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        data = resp.json()
        assert data["choices"][0]["finish_reason"] == "stop"


# ---------------------------------------------------------------------------
# Models endpoint tests
# ---------------------------------------------------------------------------


class TestModelsEndpoint:
    def test_list_models(self, client):
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        assert data["data"][0]["id"] == "test-model"

    def test_model_object_format(self, client):
        resp = client.get("/v1/models")
        data = resp.json()
        model = data["data"][0]
        assert model["object"] == "model"
        assert "owned_by" in model

    def test_multiple_models(self):
        engine = _make_engine(models=["model-a", "model-b", "model-c"])
        app = create_app(engine, "model-a")
        client = TestClient(app)
        resp = client.get("/v1/models")
        data = resp.json()
        assert len(data["data"]) == 3


# ---------------------------------------------------------------------------
# Health endpoint tests
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    def test_healthy(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_unhealthy(self):
        engine = _make_engine()
        engine.health.return_value = False
        app = create_app(engine, "test-model")
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# App creation tests
# ---------------------------------------------------------------------------


class TestCreateApp:
    def test_app_state(self):
        engine = _make_engine()
        app = create_app(engine, "test-model")
        assert app.state.engine is engine
        assert app.state.model == "test-model"

    def test_app_with_agent(self):
        engine = _make_engine()
        agent = _make_agent()
        app = create_app(engine, "test-model", agent=agent)
        assert app.state.agent is agent

    def test_app_without_agent(self):
        engine = _make_engine()
        app = create_app(engine, "test-model")
        assert app.state.agent is None


# ---------------------------------------------------------------------------
# Agent streaming tests (_handle_agent_stream_real)
# ---------------------------------------------------------------------------


class TestAgentStreamReal:
    """Tests for ``_handle_agent_stream_real()`` — real token streaming
    through the agent's PromptBuilder, tools, and engine.stream_full().
    """

    def test_streaming_with_agent(self):
        """Streaming with an agent returns SSE chunks with content tokens."""
        engine = _make_streaming_engine(tokens=["I", " am", " Jarvis"])
        agent = _make_streaming_agent()
        app = create_app(engine, "test-model", agent=agent)
        client = TestClient(app)

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Who are you?"}],
                "stream": True,
            },
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")

        # Parse SSE data lines
        lines = resp.text.strip().split("\n")
        data_lines = [ln for ln in lines if ln.startswith("data:")]
        assert len(data_lines) > 0
        # Last must be [DONE]
        assert data_lines[-1].strip() == "data: [DONE]"

        # Collect content from delta chunks
        content = ""
        for line in data_lines:
            raw = line[5:].strip()
            if raw == "[DONE]":
                continue
            data = json.loads(raw)
            delta_content = (
                data.get("choices", [{}])[0]
                .get("delta", {})
                .get("content")
            )
            if delta_content:
                content += delta_content
        assert content == "I am Jarvis"

    def test_streaming_with_agent_has_soul(self):
        """When PromptBuilder is active, system prompt contains SOUL.md content.

        We verify by capturing the messages passed to engine.stream_full()
        and checking the first (system) message contains the SOUL text.
        """
        from openjarvis.engine._stubs import StreamChunk

        engine = _make_streaming_engine()
        agent = _make_streaming_agent(soul_content="SOUL: Be kind and wise.")

        captured_messages = []

        original_stream_full = engine.stream_full

        async def capturing_stream_full(messages, *, model, **kwargs):
            captured_messages.extend(messages)
            async for chunk in original_stream_full(messages, model=model, **kwargs):
                yield chunk

        engine.stream_full = capturing_stream_full

        app = create_app(engine, "test-model", agent=agent)
        client = TestClient(app)

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
            },
        )
        assert resp.status_code == 200

        # The first message sent to engine should be a system message with SOUL content
        assert len(captured_messages) >= 2
        system_msg = captured_messages[0]
        assert system_msg.role.value == "system"
        assert "SOUL: Be kind and wise." in system_msg.content

    def test_streaming_with_agent_tools(self):
        """Agent with tools: tool_call events appear in the SSE stream."""
        from openjarvis.engine._stubs import StreamChunk

        tool_defs = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                },
            },
        ]
        agent = _make_streaming_agent(tools=tool_defs)

        # Mock executor.execute to return a tool result
        tool_result = MagicMock()
        tool_result.content = '{"temp": 22, "unit": "C"}'
        agent._executor.execute.return_value = tool_result

        engine = _make_engine()

        # Turn 1: model emits a tool call
        # Turn 2: model emits the final text answer
        call_count = 0

        async def multi_turn_stream_full(messages, *, model, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First turn: emit tool call fragments
                yield StreamChunk(
                    tool_calls=[{
                        "index": 0,
                        "id": "call_abc",
                        "function": {"name": "get_weather", "arguments": ""},
                    }],
                )
                yield StreamChunk(
                    tool_calls=[{
                        "index": 0,
                        "function": {"name": "", "arguments": '{"city":"Berlin"}'},
                    }],
                )
                yield StreamChunk(finish_reason="tool_calls")
            else:
                # Second turn: emit final answer after tool execution
                yield StreamChunk(content="It is 22C in Berlin.")
                yield StreamChunk(finish_reason="stop")

        engine.stream_full = multi_turn_stream_full

        app = create_app(engine, "test-model", agent=agent)
        client = TestClient(app)

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [
                    {"role": "user", "content": "What is the weather in Berlin?"},
                ],
                "stream": True,
            },
        )
        assert resp.status_code == 200

        # Check that tool_calls and tool_result SSE events are present
        assert "event: tool_calls" in resp.text
        assert "event: tool_result" in resp.text
        assert "get_weather" in resp.text

        # Check final content is in the stream
        content = ""
        for line in resp.text.strip().split("\n"):
            if line.startswith("data:") and "[DONE]" not in line:
                raw = line[5:].strip()
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                delta_content = (
                    data.get("choices", [{}])[0]
                    .get("delta", {})
                    .get("content")
                )
                if delta_content:
                    content += delta_content
        assert "22C" in content
        assert "Berlin" in content

        # Verify executor.execute was called
        agent._executor.execute.assert_called_once()

    def test_streaming_without_agent_unchanged(self):
        """Without an agent, streaming goes through the plain engine path."""
        engine = _make_engine()
        app = create_app(engine, "test-model")  # No agent
        client = TestClient(app)

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
            },
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")

        # Collect content — should come from engine.stream (not stream_full)
        content = ""
        for line in resp.text.strip().split("\n"):
            if line.startswith("data:") and "[DONE]" not in line:
                raw = line[5:].strip()
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                delta_content = (
                    data.get("choices", [{}])[0]
                    .get("delta", {})
                    .get("content")
                )
                if delta_content:
                    content += delta_content
        assert content == "Hello world"
