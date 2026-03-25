"""Route handlers for the OpenAI-compatible API server."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from openjarvis.core.types import Message, Role
from openjarvis.server.models import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ChoiceMessage,
    ComplexityInfo,
    DeltaMessage,
    ModelListResponse,
    ModelObject,
    StreamChoice,
    UsageInfo,
)

router = APIRouter()


def _to_messages(chat_messages) -> list[Message]:
    """Convert Pydantic ChatMessage objects to core Message objects."""
    messages = []
    for m in chat_messages:
        role = Role(m.role) if m.role in {r.value for r in Role} else Role.USER
        messages.append(Message(
            role=role,
            content=m.content or "",
            name=m.name,
            tool_call_id=m.tool_call_id,
        ))
    return messages


@router.post("/v1/chat/completions")
async def chat_completions(request_body: ChatCompletionRequest, request: Request):
    """Handle chat completion requests (streaming and non-streaming)."""
    engine = request.app.state.engine
    agent = getattr(request.app.state, "agent", None)
    model = request_body.model

    # Inject memory context into messages before dispatching
    config = getattr(request.app.state, "config", None)
    memory_backend = getattr(request.app.state, "memory_backend", None)
    if (
        config is not None
        and memory_backend is not None
        and config.agent.context_from_memory
        and request_body.messages
    ):
        try:
            from openjarvis.tools.storage.context import ContextConfig, inject_context

            # Extract query from the last user message
            query_text = ""
            for m in reversed(request_body.messages):
                if m.role == "user" and m.content:
                    query_text = m.content
                    break

            if query_text:
                messages = _to_messages(request_body.messages)
                ctx_cfg = ContextConfig(
                    top_k=config.memory.context_top_k,
                    min_score=config.memory.context_min_score,
                    max_context_tokens=config.memory.context_max_tokens,
                )
                enriched = inject_context(
                    query_text, messages, memory_backend, config=ctx_cfg,
                )
                # Rebuild request messages from enriched Message objects
                if len(enriched) > len(messages):
                    from openjarvis.server.models import ChatMessage

                    new_msgs = []
                    for msg in enriched:
                        new_msgs.append(ChatMessage(
                            role=msg.role.value,
                            content=msg.content,
                            name=msg.name,
                            tool_call_id=getattr(msg, "tool_call_id", None),
                        ))
                    request_body.messages = new_msgs
        except Exception:
            logging.getLogger("openjarvis.server").debug(
                "Memory context injection failed", exc_info=True,
            )

    # Run complexity analysis on the last user message
    complexity_info = None
    query_text_for_complexity = ""
    for m in reversed(request_body.messages):
        if m.role == "user" and m.content:
            query_text_for_complexity = m.content
            break
    if query_text_for_complexity:
        try:
            from openjarvis.learning.routing.complexity import (
                adjust_tokens_for_model,
                score_complexity,
            )

            cr = score_complexity(query_text_for_complexity)
            suggested = adjust_tokens_for_model(
                cr.suggested_max_tokens, model,
            )
            complexity_info = ComplexityInfo(
                score=cr.score,
                tier=cr.tier,
                suggested_max_tokens=suggested,
            )
            # Bump max_tokens when complexity suggests more than what
            # the client requested — never reduce below the request value.
            if suggested > request_body.max_tokens:
                request_body.max_tokens = suggested
        except Exception:
            logging.getLogger("openjarvis.server").debug(
                "Complexity analysis failed", exc_info=True,
            )

    if request_body.stream:
        # Prefer real agent streaming (PromptBuilder + tools + multi-turn)
        # whenever an agent is loaded.  Falls back to plain engine streaming
        # when no agent is configured.
        if agent is not None:
            return await _handle_agent_stream_real(
                agent, engine, model, request_body,
            )
        return await _handle_stream(engine, model, request_body, complexity_info)

    # Non-streaming: use agent if available, otherwise direct engine call
    if agent is not None:
        return _handle_agent(agent, model, request_body, complexity_info)

    bus = getattr(request.app.state, "bus", None)
    return _handle_direct(
        engine, model, request_body,
        bus=bus, complexity_info=complexity_info,
    )


def _handle_direct(
    engine, model: str, req: ChatCompletionRequest, bus=None,
    complexity_info=None,
) -> ChatCompletionResponse:
    """Direct engine call without agent."""
    messages = _to_messages(req.messages)
    kwargs: dict[str, Any] = {}
    if req.tools:
        kwargs["tools"] = req.tools
    if bus:
        from openjarvis.telemetry.wrapper import instrumented_generate

        result = instrumented_generate(
            engine, messages, model=model, bus=bus,
            temperature=req.temperature, max_tokens=req.max_tokens,
            **kwargs,
        )
    else:
        result = engine.generate(
            messages,
            model=model,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            **kwargs,
        )
    content = result.get("content", "")
    usage = result.get("usage", {})

    choice_msg = ChoiceMessage(role="assistant", content=content)
    # Include tool calls if present
    tool_calls = result.get("tool_calls")
    if tool_calls:
        choice_msg.tool_calls = [
            {
                "id": tc.get("id", ""),
                "type": "function",
                "function": {
                    "name": tc.get("name", ""),
                    "arguments": tc.get("arguments", "{}"),
                },
            }
            for tc in tool_calls
        ]

    return ChatCompletionResponse(
        model=model,
        choices=[Choice(
            message=choice_msg,
            finish_reason=result.get("finish_reason", "stop"),
        )],
        usage=UsageInfo(
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
        ),
        complexity=complexity_info,
    )


def _handle_agent(
    agent, model: str, req: ChatCompletionRequest,
    complexity_info=None,
) -> ChatCompletionResponse:
    """Run through agent."""
    from openjarvis.agents._stubs import AgentContext

    # Build context from prior messages
    ctx = AgentContext()
    if len(req.messages) > 1:
        prior = _to_messages(req.messages[:-1])
        for m in prior:
            ctx.conversation.add(m)

    # Last message is the input
    input_text = req.messages[-1].content if req.messages else ""

    # Override agent model for this request if the caller specified one
    original_model = agent._model
    if model:
        agent._model = model
    try:
        result = agent.run(input_text, context=ctx)
    finally:
        agent._model = original_model

    usage = UsageInfo(
        prompt_tokens=result.metadata.get("prompt_tokens", 0),
        completion_tokens=result.metadata.get("completion_tokens", 0),
        total_tokens=result.metadata.get("total_tokens", 0),
    )

    return ChatCompletionResponse(
        model=model,
        choices=[Choice(
            message=ChoiceMessage(role="assistant", content=result.content),
            finish_reason="stop",
        )],
        usage=usage,
        complexity=complexity_info,
    )


async def _handle_agent_stream(agent, bus, model, req):
    """Stream agent response with EventBus events via SSE."""
    from openjarvis.server.stream_bridge import create_agent_stream

    return await create_agent_stream(agent, bus, model, req)


async def _handle_agent_stream_real(
    agent,
    engine,
    model: str,
    req: ChatCompletionRequest,
) -> StreamingResponse:
    """Stream agent response with real token-by-token output via SSE.

    Uses ``engine.stream_full()`` for true streaming while applying the
    agent's PromptBuilder (SOUL.md, USER.md, etc.) and executing any
    tool calls the model emits in a multi-turn loop.
    """
    import json as _json

    from openjarvis.core.types import Message, Role
    from openjarvis.core.types import ToolCall as MsgToolCall

    logger = logging.getLogger("openjarvis.server")

    # -- Build the system prompt via agent's PromptBuilder (if available) --
    system_prompt: str | None = None
    pb = getattr(agent, "_prompt_builder", None)
    if pb is not None:
        try:
            system_prompt = pb.build()
        except Exception:
            logger.debug("PromptBuilder.build() failed, falling back", exc_info=True)

    # Fall back to the system message from the request (if any)
    if not system_prompt:
        for m in req.messages:
            if m.role == "system" and m.content:
                system_prompt = m.content
                break

    # -- Assemble LLM messages: system + request messages (excluding
    #    the original system message to avoid duplication) --
    llm_messages: list[Message] = []
    if system_prompt:
        llm_messages.append(Message(role=Role.SYSTEM, content=system_prompt))
    for m in req.messages:
        if m.role == "system":
            # Already handled above via PromptBuilder or fallback
            continue
        role = Role(m.role) if m.role in {r.value for r in Role} else Role.USER
        llm_messages.append(Message(
            role=role,
            content=m.content or "",
            name=m.name,
            tool_call_id=m.tool_call_id,
        ))

    # -- Collect tool definitions from the agent --
    openai_tools: list[dict] = []
    executor = getattr(agent, "_executor", None)
    if executor is not None:
        try:
            openai_tools = executor.get_openai_tools()
        except Exception:
            logger.debug("Failed to get agent tools", exc_info=True)

    # Also include tools from the request (client-supplied)
    if req.tools:
        openai_tools = openai_tools + list(req.tools)

    stream_kwargs: dict = {}
    if openai_tools:
        stream_kwargs["tools"] = openai_tools

    temperature = req.temperature
    max_tokens = req.max_tokens
    max_turns = getattr(agent, "_max_turns", 10)

    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    async def generate():
        """Async generator yielding SSE chunks with real token streaming."""
        messages_for_llm = list(llm_messages)
        turns = 0

        while turns < max_turns:
            turns += 1
            turn_content = ""
            tool_call_fragments: dict[int, dict] = {}
            current_finish_reason = None

            try:
                async for chunk in engine.stream_full(
                    messages_for_llm,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **stream_kwargs,
                ):
                    # Stream content tokens to the client immediately
                    if chunk.content:
                        turn_content += chunk.content
                        chunk_data = ChatCompletionChunk(
                            id=chunk_id,
                            model=model,
                            choices=[StreamChoice(
                                delta=DeltaMessage(content=chunk.content),
                            )],
                        )
                        yield f"data: {chunk_data.model_dump_json()}\n\n"

                    # Accumulate tool_call fragments
                    if chunk.tool_calls:
                        _merge_tool_call_fragments(
                            tool_call_fragments, chunk.tool_calls,
                        )

                    if chunk.finish_reason:
                        current_finish_reason = chunk.finish_reason

            except Exception as exc:
                logger.error("Agent stream error: %s", exc, exc_info=True)
                error_chunk = ChatCompletionChunk(
                    id=chunk_id,
                    model=model,
                    choices=[StreamChoice(
                        delta=DeltaMessage(
                            content=f"\n\nError during generation: {exc}",
                        ),
                        finish_reason="stop",
                    )],
                )
                yield f"data: {error_chunk.model_dump_json()}\n\n"
                yield "data: [DONE]\n\n"
                return

            # -- Handle tool calls: execute and loop for the next turn --
            if tool_call_fragments and current_finish_reason == "tool_calls":
                sorted_tcs = [
                    tool_call_fragments[i]
                    for i in sorted(tool_call_fragments.keys())
                ]

                # Emit tool_calls metadata as SSE event (informational)
                tool_meta = [
                    {
                        "tool_name": tc["function"]["name"],
                        "arguments": tc["function"]["arguments"],
                    }
                    for tc in sorted_tcs
                ]
                yield (
                    f"event: tool_calls\n"
                    f"data: {_json.dumps({'calls': tool_meta})}\n\n"
                )

                # Append assistant message with tool_calls to the conversation
                assistant_msg = Message(
                    role=Role.ASSISTANT,
                    content=turn_content or None,
                    tool_calls=[
                        MsgToolCall(
                            id=tc["id"],
                            name=tc["function"]["name"],
                            arguments=tc["function"]["arguments"],
                        )
                        for tc in sorted_tcs
                    ],
                )
                messages_for_llm.append(assistant_msg)

                # Execute each tool call and feed results back
                for tc in sorted_tcs:
                    tool_name = tc["function"]["name"]
                    tool_args = tc["function"]["arguments"]
                    tool_result_content = f"Tool '{tool_name}' not available"

                    try:
                        if executor is not None:
                            result = executor.execute(MsgToolCall(
                                id=tc["id"],
                                name=tool_name,
                                arguments=tool_args,
                            ))
                            tool_result_content = result.content
                        else:
                            logger.warning(
                                "No executor available for tool '%s'", tool_name,
                            )
                    except Exception as tool_exc:
                        logger.error(
                            "Tool execution error for %s: %s",
                            tool_name, tool_exc, exc_info=True,
                        )
                        tool_result_content = (
                            f"Error executing {tool_name}: {tool_exc}"
                        )

                    # Emit tool result as SSE event (informational)
                    tr_data = _json.dumps({
                        "tool_name": tool_name,
                        "output": tool_result_content,
                    })
                    yield f"event: tool_result\ndata: {tr_data}\n\n"

                    # Append tool result to conversation for next turn
                    messages_for_llm.append(Message(
                        role=Role.TOOL,
                        content=tool_result_content,
                        tool_call_id=tc["id"],
                        name=tool_name,
                    ))

                # Continue to next turn (loop back to stream_full)
                continue

            # No tool calls — final response turn
            break

        # Send finish chunk
        finish_chunk = ChatCompletionChunk(
            id=chunk_id,
            model=model,
            choices=[StreamChoice(
                delta=DeltaMessage(),
                finish_reason="stop",
            )],
        )
        yield f"data: {finish_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


def _merge_tool_call_fragments(
    accumulated: dict[int, dict],
    fragments: list[dict],
) -> None:
    """Merge incremental tool_call delta fragments into accumulated state.

    OpenAI-compatible APIs send tool_calls as incremental fragments keyed
    by ``index``. Each fragment may contain partial ``function.name`` and/or
    ``function.arguments`` strings that must be concatenated.
    """
    for frag in fragments:
        idx = frag.get("index", 0)
        if idx not in accumulated:
            accumulated[idx] = {
                "id": frag.get("id", ""),
                "type": "function",
                "function": {"name": "", "arguments": ""},
            }
        entry = accumulated[idx]
        if frag.get("id"):
            entry["id"] = frag["id"]
        fn = frag.get("function", {})
        if fn.get("name"):
            entry["function"]["name"] += fn["name"]
        if fn.get("arguments"):
            entry["function"]["arguments"] += fn["arguments"]


async def _handle_stream(
    engine, model: str, req: ChatCompletionRequest,
    complexity_info=None,
):
    """Stream response using SSE format."""
    messages = _to_messages(req.messages)
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    async def generate():
        # Send role chunk first
        first_chunk = ChatCompletionChunk(
            id=chunk_id,
            model=model,
            choices=[StreamChoice(
                delta=DeltaMessage(role="assistant"),
            )],
        )
        yield f"data: {first_chunk.model_dump_json()}\n\n"

        try:
            # Stream content
            async for token in engine.stream(
                messages,
                model=model,
                temperature=req.temperature,
                max_tokens=req.max_tokens,
            ):
                chunk = ChatCompletionChunk(
                    id=chunk_id,
                    model=model,
                    choices=[StreamChoice(
                        delta=DeltaMessage(content=token),
                    )],
                )
                yield f"data: {chunk.model_dump_json()}\n\n"
        except Exception as exc:
            # Surface errors as a content chunk so the frontend can
            # display them instead of silently failing.
            import logging
            logging.getLogger("openjarvis.server").error(
                "Stream error: %s", exc, exc_info=True,
            )
            error_chunk = ChatCompletionChunk(
                id=chunk_id,
                model=model,
                choices=[StreamChoice(
                    delta=DeltaMessage(
                        content=f"\n\nError during generation: {exc}",
                    ),
                    finish_reason="stop",
                )],
            )
            yield f"data: {error_chunk.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"
            return

        # Send finish chunk with usage data if available
        import json as _json
        finish_data = ChatCompletionChunk(
            id=chunk_id,
            model=model,
            choices=[StreamChoice(
                delta=DeltaMessage(),
                finish_reason="stop",
            )],
        )
        finish_dict = _json.loads(finish_data.model_dump_json())

        # Pull usage from the engine if it tracked it during streaming
        raw_engine = engine
        # Unwrap InstrumentedEngine if present
        if hasattr(raw_engine, "_inner"):
            raw_engine = raw_engine._inner
        # Unwrap MultiEngine if present
        if hasattr(raw_engine, "_engine_for"):
            raw_engine = raw_engine._engine_for(model)
        stream_usage = getattr(raw_engine, "_last_stream_usage", None)
        if isinstance(stream_usage, dict) and stream_usage.get("total_tokens", 0) > 0:
            finish_dict["usage"] = stream_usage

        if complexity_info is not None:
            finish_dict["complexity"] = complexity_info.model_dump()

        yield f"data: {_json.dumps(finish_dict)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@router.get("/v1/models")
async def list_models(request: Request) -> ModelListResponse:
    """List available models from the engine."""
    engine = request.app.state.engine
    model_ids = engine.list_models()
    return ModelListResponse(
        data=[ModelObject(id=mid) for mid in model_ids],
    )


@router.post("/v1/models/pull")
async def pull_model(request: Request):
    """Pull / download a model from the Ollama registry."""
    body = await request.json()
    model_name = body.get("model", "").strip()
    if not model_name:
        raise HTTPException(status_code=400, detail="'model' field is required")

    engine = request.app.state.engine
    engine_name = getattr(request.app.state, "engine_name", "")
    # Only Ollama supports pulling
    if engine_name != "ollama" and getattr(engine, "engine_id", "") != "ollama":
        raise HTTPException(
            status_code=501,
            detail="Model pulling is only supported with the Ollama engine",
        )

    import httpx as _httpx

    host = getattr(engine, "_host", "http://localhost:11434")
    client = _httpx.Client(base_url=host, timeout=600.0)
    try:
        resp = client.post(
            "/api/pull",
            json={"name": model_name, "stream": False},
        )
        resp.raise_for_status()
    except (_httpx.ConnectError, _httpx.TimeoutException) as exc:
        raise HTTPException(status_code=502, detail=f"Ollama unreachable: {exc}")
    except _httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=exc.response.status_code,
            detail=f"Ollama error: {exc.response.text[:300]}",
        )
    finally:
        client.close()

    return {"status": "ok", "model": model_name}


@router.delete("/v1/models/{model_name:path}")
async def delete_model(model_name: str, request: Request):
    """Delete a model from Ollama."""
    engine = request.app.state.engine
    engine_name = getattr(request.app.state, "engine_name", "")
    if engine_name != "ollama" and getattr(engine, "engine_id", "") != "ollama":
        raise HTTPException(status_code=501, detail="Only supported with Ollama engine")

    import httpx as _httpx

    host = getattr(engine, "_host", "http://localhost:11434")
    client = _httpx.Client(base_url=host, timeout=30.0)
    try:
        resp = client.request(
            "DELETE",
            "/api/delete",
            json={"name": model_name},
        )
        resp.raise_for_status()
    except (_httpx.ConnectError, _httpx.TimeoutException) as exc:
        raise HTTPException(status_code=502, detail=f"Ollama unreachable: {exc}")
    except _httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=exc.response.status_code,
            detail=f"Ollama error: {exc.response.text[:300]}",
        )
    finally:
        client.close()

    return {"status": "deleted", "model": model_name}


@router.get("/v1/savings")
async def savings(request: Request):
    """Return savings summary compared to cloud providers.

    Only includes telemetry from the current server session so that
    counters start at zero each time a new model + agent is launched.
    """
    from openjarvis.core.config import DEFAULT_CONFIG_DIR
    from openjarvis.server.savings import compute_savings, savings_to_dict
    from openjarvis.telemetry.aggregator import TelemetryAggregator

    db_path = DEFAULT_CONFIG_DIR / "telemetry.db"
    if not db_path.exists():
        empty = compute_savings(0, 0, 0)
        return savings_to_dict(empty)

    session_start = getattr(request.app.state, "session_start", None)

    agg = TelemetryAggregator(db_path)
    try:
        summary = agg.summary(since=session_start)
        # Exclude cloud model tokens from savings — only local
        # inference counts toward cost savings.
        _cloud_prefixes = (
            "gpt-", "o1-", "o3-", "o4-",
            "claude-", "gemini-", "openrouter/",
        )
        local_models = [
            m for m in summary.per_model
            if not any(m.model_id.startswith(p) for p in _cloud_prefixes)
        ]
        result = compute_savings(
            prompt_tokens=sum(m.prompt_tokens for m in local_models),
            completion_tokens=sum(
                m.completion_tokens for m in local_models
            ),
            total_calls=sum(m.call_count for m in local_models),
            session_start=session_start if session_start else 0.0,
        )
        return savings_to_dict(result)
    finally:
        agg.close()


@router.get("/v1/info")
async def server_info(request: Request):
    """Return server configuration: model, agent, engine."""
    agent = getattr(request.app.state, "agent", None)
    agent_id = getattr(agent, "agent_id", None) if agent else None
    # Fall back to configured agent name if agent didn't instantiate
    if agent_id is None:
        agent_id = getattr(request.app.state, "agent_name", None)
    return {
        "model": getattr(request.app.state, "model", ""),
        "agent": agent_id,
        "engine": getattr(request.app.state, "engine_name", ""),
    }


@router.get("/health")
async def health(request: Request):
    """Health check endpoint."""
    engine = request.app.state.engine
    healthy = engine.health()
    if not healthy:
        raise HTTPException(status_code=503, detail="Engine unhealthy")
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Channel endpoints
# ---------------------------------------------------------------------------


@router.get("/v1/channels")
async def list_channels(request: Request):
    """List available messaging channels."""
    bridge = getattr(request.app.state, "channel_bridge", None)
    if bridge is None:
        return {"channels": [], "message": "Channel bridge not configured"}
    channels = bridge.list_channels()
    return {"channels": channels, "status": bridge.status().value}


@router.post("/v1/channels/send")
async def channel_send(request: Request):
    """Send a message to a channel."""
    bridge = getattr(request.app.state, "channel_bridge", None)
    if bridge is None:
        raise HTTPException(status_code=503, detail="Channel bridge not configured")

    body = await request.json()
    channel_name = body.get("channel", "")
    content = body.get("content", "")
    conversation_id = body.get("conversation_id", "")

    if not channel_name or not content:
        raise HTTPException(
            status_code=400, detail="'channel' and 'content' are required",
        )

    ok = bridge.send(channel_name, content, conversation_id=conversation_id)
    if not ok:
        raise HTTPException(status_code=502, detail="Failed to send message")
    return {"status": "sent", "channel": channel_name}


@router.get("/v1/channels/status")
async def channel_status(request: Request):
    """Return channel bridge connection status."""
    bridge = getattr(request.app.state, "channel_bridge", None)
    if bridge is None:
        return {"status": "not_configured"}
    return {"status": bridge.status().value}


__all__ = ["router"]
