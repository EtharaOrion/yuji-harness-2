"""_text_of must strip NUL bytes before forwarding to kiro-cli.

kiro-cli's JSON transport rejects \x00 in message content with a parse error
that drops the whole conversation. The agent routinely sends raw binary (a
.docx cat'd through a tool result) that carries NULs, so every return path of
_text_of has to sanitise before handing text off downstream.
"""

from __future__ import annotations

import io

from kiro_bridge.messages_api import _text_of


def test_none_input_returns_empty_string():
    assert _text_of(None) == ""


def test_bare_string_strips_nuls():
    assert _text_of("hello\x00world") == "helloworld"
    assert _text_of("\x00\x00clean\x00") == "clean"


def test_list_of_text_blocks_strips_nuls():
    msg = [{"type": "text", "text": "a\x00b"},
           {"type": "text", "text": "\x00c\x00d\x00"}]
    assert _text_of(msg) == "ab\ncd"


def test_list_with_image_and_document_blocks_omits_them():
    msg = [{"type": "text", "text": "before"},
           {"type": "image", "source": {"data": "..."}},
           {"type": "document", "source": {"data": "..."}},
           {"type": "text", "text": "after"}]
    out = _text_of(msg)
    assert "[image omitted]" in out
    assert "[document omitted]" in out
    assert "before" in out and "after" in out


def test_list_with_raw_string_entries_handled():
    assert _text_of(["a\x00", "\x00b"]) == "a\nb"


def test_fallback_str_path_strips_nuls():
    assert "\x00" not in _text_of(io.StringIO("x\x00y"))


def test_pure_nul_string_becomes_empty():
    assert _text_of("\x00\x00\x00") == ""
