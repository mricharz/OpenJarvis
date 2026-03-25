"""Streaming filter that extracts <think>...</think> blocks from content.

Used for engines that embed reasoning in content (Ollama, LlamaCpp, etc.)
rather than providing a separate reasoning field.
"""

from __future__ import annotations


class ThinkTagFilter:
    """State machine that separates ``<think>`` reasoning from visible content.

    Handles:
    - Explicit ``<think>...</think>`` blocks (one or more).
    - Bare ``</think>`` without opening ``<think>`` (distilled models
      that start reasoning immediately).
    - Tags split across multiple ``feed()`` calls.
    """

    _OPEN_TAG = "<think>"
    _CLOSE_TAG = "</think>"

    def __init__(self) -> None:
        self._in_think: bool = False
        self._buffer: str = ""
        self._seen_any_tag: bool = False

    def feed(self, text: str) -> tuple[str, str]:
        """Feed a text chunk.

        Returns:
            ``(visible_content, reasoning_content)`` extracted so far
            from this chunk.  Either or both may be empty strings.
        """
        if not text:
            return ("", "")

        self._buffer += text
        visible: list[str] = []
        reasoning: list[str] = []

        while self._buffer:
            if self._in_think:
                # Look for closing tag
                close_idx = self._buffer.find(self._CLOSE_TAG)
                if close_idx == -1:
                    # Might be a partial tag at the end
                    partial = self._partial_tag_suffix(
                        self._buffer, self._CLOSE_TAG
                    )
                    if partial > 0:
                        # Keep the potential partial tag
                        reasoning.append(
                            self._buffer[: -partial]
                        )
                        self._buffer = self._buffer[-partial:]
                    else:
                        reasoning.append(self._buffer)
                        self._buffer = ""
                    break
                else:
                    reasoning.append(
                        self._buffer[:close_idx]
                    )
                    self._buffer = self._buffer[
                        close_idx + len(self._CLOSE_TAG):
                    ]
                    self._in_think = False
                    self._seen_any_tag = True
            else:
                # Not in think mode — look for open tag or
                # bare close tag
                open_idx = self._buffer.find(self._OPEN_TAG)
                close_idx = self._buffer.find(self._CLOSE_TAG)

                # Bare </think> before any <think>
                if (
                    not self._seen_any_tag
                    and close_idx != -1
                    and (open_idx == -1 or close_idx < open_idx)
                ):
                    # Everything before </think> was reasoning
                    reasoning.append(
                        self._buffer[:close_idx]
                    )
                    self._buffer = self._buffer[
                        close_idx + len(self._CLOSE_TAG):
                    ]
                    self._seen_any_tag = True
                    continue

                if open_idx == -1:
                    if not self._seen_any_tag:
                        # No tags seen yet — keep entire buffer
                        # (might be reasoning before bare
                        # </think>). Check for partial close tag
                        # to avoid breaking on split tags.
                        partial_close = (
                            self._partial_tag_suffix(
                                self._buffer, self._CLOSE_TAG
                            )
                        )
                        partial_open = (
                            self._partial_tag_suffix(
                                self._buffer, self._OPEN_TAG
                            )
                        )
                        if partial_close == 0 and partial_open == 0:
                            # No partial tag at end — but keep
                            # buffering anyway for potential bare
                            # </think> in a future feed() call.
                            pass
                        # Either way, keep the full buffer
                        break
                    # Tags were already seen — emit as visible
                    partial = self._partial_tag_suffix(
                        self._buffer, self._OPEN_TAG
                    )
                    if partial > 0:
                        visible.append(
                            self._buffer[: -partial]
                        )
                        self._buffer = self._buffer[-partial:]
                    else:
                        visible.append(self._buffer)
                        self._buffer = ""
                    break
                else:
                    # Found <think> tag
                    before = self._buffer[:open_idx]
                    if before:
                        if self._seen_any_tag:
                            visible.append(before)
                        else:
                            # Content before first <think> is
                            # visible (not reasoning)
                            visible.append(before)
                            self._seen_any_tag = True
                    self._buffer = self._buffer[
                        open_idx + len(self._OPEN_TAG):
                    ]
                    self._in_think = True
                    self._seen_any_tag = True

        return ("".join(visible), "".join(reasoning))

    def flush(self) -> tuple[str, str]:
        """Flush remaining buffer at stream end.

        Returns:
            ``(visible_content, reasoning_content)``.

        If no tags were ever seen, the entire buffer is returned
        as visible content (the model was not thinking).
        If still inside a think block, the buffer is reasoning.
        """
        remaining = self._buffer
        self._buffer = ""
        if not remaining:
            return ("", "")
        if self._in_think:
            return ("", remaining)
        if not self._seen_any_tag:
            # Never saw any tag — everything is visible content
            return (remaining, "")
        return (remaining, "")

    @staticmethod
    def _partial_tag_suffix(text: str, tag: str) -> int:
        """Check if ``text`` ends with a partial prefix of ``tag``.

        Returns the length of the matching suffix (0 if none).
        """
        # Check decreasing suffix lengths of text against
        # prefixes of tag
        max_check = min(len(text), len(tag) - 1)
        for length in range(max_check, 0, -1):
            if text[-length:] == tag[:length]:
                return length
        return 0


__all__ = ["ThinkTagFilter"]
