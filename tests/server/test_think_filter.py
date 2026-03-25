"""Tests for the ThinkTagFilter streaming state machine."""

from __future__ import annotations

from openjarvis.server.think_filter import ThinkTagFilter


class TestThinkTagFilter:
    """Unit tests for ThinkTagFilter."""

    def test_no_think_tags(self) -> None:
        """Plain text without tags is returned as visible content."""
        f = ThinkTagFilter()
        vis, reas = f.feed("Hello world")
        fv, fr = f.flush()
        assert vis + fv == "Hello world"
        assert reas + fr == ""

    def test_explicit_think_block(self) -> None:
        """Standard <think>reason</think>answer flow."""
        f = ThinkTagFilter()
        vis, reas = f.feed(
            "<think>reason</think>answer"
        )
        fv, fr = f.flush()
        assert vis + fv == "answer"
        assert reas + fr == "reason"

    def test_bare_think(self) -> None:
        """Bare </think> without opening <think> tag."""
        f = ThinkTagFilter()
        vis, reas = f.feed("reason</think>answer")
        fv, fr = f.flush()
        assert vis + fv == "answer"
        assert reas + fr == "reason"

    def test_split_open_tag(self) -> None:
        """<think> tag split across two feed() calls."""
        f = ThinkTagFilter()
        v1, r1 = f.feed("<thi")
        v2, r2 = f.feed("nk>reason</think>answer")
        fv, fr = f.flush()
        visible = v1 + v2 + fv
        reasoning = r1 + r2 + fr
        assert visible == "answer"
        assert reasoning == "reason"

    def test_split_close_tag(self) -> None:
        """</think> tag split across two feed() calls."""
        f = ThinkTagFilter()
        v1, r1 = f.feed("<think>reason</thi")
        v2, r2 = f.feed("nk>answer")
        fv, fr = f.flush()
        visible = v1 + v2 + fv
        reasoning = r1 + r2 + fr
        assert visible == "answer"
        assert reasoning == "reason"

    def test_flush_no_tags(self) -> None:
        """feed() + flush() without any tags."""
        f = ThinkTagFilter()
        v1, r1 = f.feed("hello")
        fv, fr = f.flush()
        assert v1 + fv == "hello"
        assert r1 + fr == ""

    def test_flush_in_think_block(self) -> None:
        """Stream ends while still inside a think block."""
        f = ThinkTagFilter()
        v1, r1 = f.feed("<think>reason")
        fv, fr = f.flush()
        assert v1 + fv == ""
        assert r1 + fr == "reason"

    def test_multiple_blocks(self) -> None:
        """Multiple think blocks in one feed."""
        f = ThinkTagFilter()
        vis, reas = f.feed(
            "<think>r1</think>a1<think>r2</think>a2"
        )
        fv, fr = f.flush()
        assert vis + fv == "a1a2"
        assert reas + fr == "r1r2"

    def test_empty_input(self) -> None:
        """Empty string feed returns empty results."""
        f = ThinkTagFilter()
        vis, reas = f.feed("")
        assert vis == ""
        assert reas == ""

    def test_incremental_token_stream(self) -> None:
        """Simulate a realistic token-by-token stream."""
        f = ThinkTagFilter()
        tokens = [
            "<", "think", ">",
            "Let me ", "think...",
            "</", "think", ">",
            "The ", "answer ", "is 42.",
        ]
        all_vis = []
        all_reas = []
        for token in tokens:
            v, r = f.feed(token)
            all_vis.append(v)
            all_reas.append(r)
        fv, fr = f.flush()
        all_vis.append(fv)
        all_reas.append(fr)
        assert "".join(all_vis) == "The answer is 42."
        assert "".join(all_reas) == "Let me think..."

    def test_bare_think_split(self) -> None:
        """Bare </think> split across calls."""
        f = ThinkTagFilter()
        v1, r1 = f.feed("reasoning</thi")
        v2, r2 = f.feed("nk>visible")
        fv, fr = f.flush()
        assert v1 + v2 + fv == "visible"
        assert r1 + r2 + fr == "reasoning"
