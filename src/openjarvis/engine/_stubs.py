"""ABC for inference engine backends.

Adapted from IPW's ``InferenceClient`` at ``src/ipw/clients/base.py``.
Phase 1 will provide concrete implementations (vLLM, Ollama, etc.).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from openjarvis.core.types import Message


@dataclass(slots=True)
class StreamChunk:
    """A single chunk emitted by ``stream_full()``.

    Attributes:
        content: Visible content token (may be empty).
        reasoning: Reasoning / thinking token (may be ``None``).
        finish_reason: Set on the final chunk (e.g. ``"stop"``).
        usage: Token usage dict, typically on the final chunk.
    """

    content: str = ""
    reasoning: Optional[str] = None
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, Any]] = field(default=None)


@dataclass(slots=True)
class ResponseFormat:
    """Structured output configuration for inference engines.

    Attributes:
        type: The response format type. ``"json_object"`` enables JSON mode
            (the model returns valid JSON). ``"json_schema"`` enables structured
            output constrained to a specific JSON Schema.
        schema: A JSON Schema dict used when *type* is ``"json_schema"``.
            Ignored for ``"json_object"`` mode.
    """

    type: str = "json_object"
    schema: Optional[Dict[str, Any]] = field(default=None)


class InferenceEngine(ABC):
    """Base class for all inference engine backends.

    Subclasses must be registered via
    ``@EngineRegistry.register("name")`` to become discoverable.
    """

    engine_id: str

    @abstractmethod
    def generate(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Synchronous completion — returns a dict with ``content`` and ``usage``."""

    @abstractmethod
    async def stream(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Yield token strings as they are generated."""
        # NOTE: must contain a yield to satisfy the type checker
        yield ""  # pragma: no cover

    @abstractmethod
    def list_models(self) -> List[str]:
        """Return identifiers of models available on this engine."""

    @abstractmethod
    def health(self) -> bool:
        """Return ``True`` when the engine is reachable and healthy."""

    def close(self) -> None:
        """Release resources (HTTP clients, connections, threads, etc.)."""

    def prepare(self, model: str) -> None:
        """Optional warm-up hook called before the first request."""

    async def stream_full(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Yield :class:`StreamChunk` objects with content + reasoning.

        Default implementation wraps ``stream()`` so that engines
        which only implement ``stream()`` still work.
        """
        async for token in self.stream(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        ):
            yield StreamChunk(content=token, reasoning=None)


__all__ = ["InferenceEngine", "ResponseFormat", "StreamChunk"]
