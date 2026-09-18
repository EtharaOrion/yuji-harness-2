#!/usr/bin/env python3
"""Tests for grader_compress.

Run on the HOST, where `headroom` is not installed. That is deliberate: the
no-headroom path is the one every grading run takes today, so it is the path
that most needs proving safe. The tests that need a real compressor install a
fake one, so the suite never depends on a private package being available.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import grader_compress  # noqa: E402


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    """Each test starts with a clean probe, clean stats, compression ON.

    The module caches the import probe and the warning set process-wide, which
    is right in production (one stderr line per run, not one per criterion) and
    wrong across tests.
    """
    monkeypatch.setattr(grader_compress, "_HEADROOM_AVAILABLE", None)
    monkeypatch.setattr(grader_compress, "_compress", None)
    monkeypatch.setattr(grader_compress, "_CompressConfig", None)
    monkeypatch.setattr(grader_compress, "_WARNED", set())
    # Force the not-installed path as the default for every test. It used to be
    # the default for free, because no host had headroom-ai. Once it is
    # installed -- as it now is for the host-side judge -- the fail-open tests
    # would quietly start exercising the real library instead of the path they
    # were written to prove, and would pass or fail depending on whether the
    # machine happened to have it. Tests that want a compressor install a fake
    # one, which bypasses the probe entirely.
    monkeypatch.setitem(sys.modules, "headroom", None)
    grader_compress.reset_stats()
    monkeypatch.setenv("GRADER_HEADROOM_ENABLED", "true")
    yield
    grader_compress.reset_stats()


class _Result:
    def __init__(self, messages, saved, before=1000, after=600):
        self.messages = messages
        self.tokens_saved = saved
        self.tokens_before = before
        self.tokens_after = after


def _install_fake(monkeypatch, fn):
    """Stand in for headroom.compress without needing the real package."""
    monkeypatch.setattr(grader_compress, "_HEADROOM_AVAILABLE", True)
    monkeypatch.setattr(grader_compress, "_compress", fn)
    monkeypatch.setattr(grader_compress, "_CompressConfig", lambda **kw: kw)


MESSAGES = [
    {"role": "system", "content": "return only JSON"},
    {"role": "user", "content": "a" * 4000},
]

EVIDENCE = "tool=x args={} -> ok\n" * 400


# --- default-off ------------------------------------------------------------

def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv("GRADER_HEADROOM_ENABLED", raising=False)
    assert grader_compress.enabled() is False
    assert grader_compress.compress_messages("gpt-5.6-sol", MESSAGES) is MESSAGES
    assert grader_compress.compress_evidence("gpt-5.6-sol", EVIDENCE) is EVIDENCE


def test_disabled_does_not_even_probe(monkeypatch):
    """An opted-out run must not attempt the import or emit a warning."""
    monkeypatch.delenv("GRADER_HEADROOM_ENABLED", raising=False)
    called = []
    monkeypatch.setattr(grader_compress, "_probe", lambda: called.append(1) or True)
    grader_compress.compress_messages("gpt-5.6-sol", MESSAGES)
    grader_compress.compress_evidence("gpt-5.6-sol", EVIDENCE)
    assert called == []


@pytest.mark.parametrize("value", ["1", "true", "TRUE", " yes ", "on"])
def test_truthy_spellings(monkeypatch, value):
    monkeypatch.setenv("GRADER_HEADROOM_ENABLED", value)
    assert grader_compress.enabled() is True


@pytest.mark.parametrize("value", ["", "0", "false", "no", "maybe"])
def test_falsy_spellings(monkeypatch, value):
    monkeypatch.setenv("GRADER_HEADROOM_ENABLED", value)
    assert grader_compress.enabled() is False


# --- fail-open --------------------------------------------------------------

def test_missing_package_is_a_passthrough():
    """The real host state: headroom is not installed, grading proceeds."""
    assert grader_compress.compress_messages("gpt-5.6-sol", MESSAGES) is MESSAGES
    assert grader_compress.stats()["calls_skipped"] == 1


def test_missing_package_warns_once(capsys):
    for _ in range(3):
        grader_compress.compress_messages("gpt-5.6-sol", MESSAGES)
    err = capsys.readouterr().err
    assert err.count("headroom not importable") == 1


def test_compressor_raising_is_a_passthrough(monkeypatch):
    def boom(*a, **kw):
        raise RuntimeError("kaboom")

    _install_fake(monkeypatch, boom)
    assert grader_compress.compress_messages("gpt-5.6-sol", MESSAGES) is MESSAGES
    assert grader_compress.compress_evidence("gpt-5.6-sol", EVIDENCE) is EVIDENCE


def test_zero_saving_is_a_passthrough(monkeypatch):
    _install_fake(monkeypatch, lambda msgs, **kw: _Result([dict(m) for m in msgs], 0))
    assert grader_compress.compress_messages("gpt-5.6-sol", MESSAGES) is MESSAGES


def test_length_mismatch_is_a_passthrough(monkeypatch):
    _install_fake(monkeypatch, lambda msgs, **kw: _Result(list(msgs)[:1], 400))
    assert grader_compress.compress_messages("gpt-5.6-sol", MESSAGES) is MESSAGES


def test_non_list_result_is_a_passthrough(monkeypatch):
    _install_fake(monkeypatch, lambda msgs, **kw: _Result("nope", 400))
    assert grader_compress.compress_messages("gpt-5.6-sol", MESSAGES) is MESSAGES


def test_empty_input_is_a_passthrough():
    assert grader_compress.compress_messages("gpt-5.6-sol", []) == []
    assert grader_compress.compress_evidence("gpt-5.6-sol", "") == ""
    assert grader_compress.compress_evidence("gpt-5.6-sol", "   ") == "   "


def test_non_string_evidence_is_a_passthrough():
    assert grader_compress.compress_evidence("gpt-5.6-sol", None) is None
    assert grader_compress.compress_evidence("gpt-5.6-sol", {"a": 1}) == {"a": 1}


# --- the cache_control interlock -------------------------------------------

def test_cache_control_on_message_stands_down(monkeypatch):
    _install_fake(monkeypatch, lambda msgs, **kw: pytest.fail("must not be called"))
    cached = [{"role": "user", "content": "x" * 4000, "cache_control": {"type": "ephemeral"}}]
    assert grader_compress.compress_messages("gpt-5.6-sol", cached) is cached


def test_cache_control_on_block_stands_down(monkeypatch):
    _install_fake(monkeypatch, lambda msgs, **kw: pytest.fail("must not be called"))
    cached = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "x" * 4000, "cache_control": {"type": "ephemeral"}}
            ],
        }
    ]
    assert grader_compress.compress_messages("gpt-5.6-sol", cached) is cached


def test_cache_control_warns_once(capsys):
    cached = [{"role": "user", "content": "x", "cache_control": {"type": "ephemeral"}}]
    for _ in range(3):
        grader_compress.compress_messages("gpt-5.6-sol", cached)
    assert capsys.readouterr().err.count("stood down") == 1


def test_interlock_beats_the_probe(monkeypatch):
    """A cached list must stand down even when headroom is importable."""
    _install_fake(monkeypatch, lambda msgs, **kw: pytest.fail("must not be called"))
    cached = [{"role": "user", "content": "x", "cache_control": {}}]
    grader_compress.compress_messages("gpt-5.6-sol", cached)
    assert grader_compress.stats()["calls_compressed"] == 0


# --- the one-shot default ---------------------------------------------------

def test_protect_recent_defaults_to_zero(monkeypatch):
    """Every grader here sends one turn; a tail of 2 would protect all of it."""
    seen = {}

    def fake(msgs, **kw):
        seen.update(kw["config"])
        return _Result([dict(m) for m in msgs], 0)

    _install_fake(monkeypatch, fake)
    grader_compress.compress_messages("gpt-5.6-sol", MESSAGES)
    assert seen["protect_recent"] == 0


def test_protect_recent_is_overridable(monkeypatch):
    seen = {}

    def fake(msgs, **kw):
        seen.update(kw["config"])
        return _Result([dict(m) for m in msgs], 0)

    _install_fake(monkeypatch, fake)
    monkeypatch.setenv("GRADER_HEADROOM_PROTECT_RECENT", "2")
    grader_compress.compress_messages("gpt-5.6-sol", MESSAGES)
    assert seen["protect_recent"] == 2


def test_evidence_pins_protect_recent_to_zero(monkeypatch):
    """compress_evidence must ignore the env knob, or it switches itself off."""
    seen = {}

    def fake(msgs, **kw):
        seen.update(kw["config"])
        return _Result([{"role": "user", "content": "small"}], 400)

    _install_fake(monkeypatch, fake)
    monkeypatch.setenv("GRADER_HEADROOM_PROTECT_RECENT", "5")
    grader_compress.compress_evidence("gpt-5.6-sol", EVIDENCE)
    assert seen["protect_recent"] == 0


def test_system_messages_are_never_compressed(monkeypatch):
    seen = {}

    def fake(msgs, **kw):
        seen.update(kw["config"])
        return _Result([dict(m) for m in msgs], 0)

    _install_fake(monkeypatch, fake)
    grader_compress.compress_messages("gpt-5.6-sol", MESSAGES)
    assert seen["compress_system_messages"] is False
    assert seen["compress_user_messages"] is True


# --- tuning knobs -----------------------------------------------------------

def test_min_tokens_and_ratio_defaults(monkeypatch):
    seen = {}

    def fake(msgs, **kw):
        seen.update(kw["config"])
        return _Result([dict(m) for m in msgs], 0)

    _install_fake(monkeypatch, fake)
    grader_compress.compress_messages("gpt-5.6-sol", MESSAGES)
    assert seen["min_tokens_to_compress"] == 500
    assert seen["target_ratio"] == 0.4


def test_garbage_env_falls_back_to_defaults(monkeypatch):
    seen = {}

    def fake(msgs, **kw):
        seen.update(kw["config"])
        return _Result([dict(m) for m in msgs], 0)

    _install_fake(monkeypatch, fake)
    monkeypatch.setenv("GRADER_HEADROOM_MIN_TOKENS", "not-a-number")
    monkeypatch.setenv("GRADER_HEADROOM_TARGET_RATIO", "")
    grader_compress.compress_messages("gpt-5.6-sol", MESSAGES)
    assert seen["min_tokens_to_compress"] == 500
    assert seen["target_ratio"] == 0.4


# --- model hint -------------------------------------------------------------

@pytest.mark.parametrize(
    "given,expected",
    [
        ("claude-sonnet-4-6", "anthropic/claude-opus-4-20250514"),
        ("gpt-5.6-sol", "gpt-5.6-sol"),
        ("gpt-4o", "gpt-4o"),
        ("o3-mini", "o3-mini"),
        ("gemini/gemini-3.1-pro-preview", "gpt-4o"),
        ("", "gpt-4o"),
        (None, "gpt-4o"),
    ],
)
def test_model_hint(given, expected):
    assert grader_compress._model_hint(given) == expected


# --- text-block flattening --------------------------------------------------

def test_flatten_collapses_pure_text_blocks():
    msg = {"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}
    assert grader_compress._flatten_text_blocks(msg)["content"] == "a\nb"


@pytest.mark.parametrize(
    "content",
    [
        "already a string",
        [],
        [{"type": "tool_use", "id": "1", "name": "x", "input": {}}],
        [{"type": "tool_result", "tool_use_id": "1", "content": "y"}],
        [{"type": "image", "source": {}}],
        [{"type": "thinking", "thinking": "..."}],
        [{"type": "text", "text": "a", "citations": []}],
    ],
)
def test_flatten_refuses_structured_content(content):
    msg = {"role": "user", "content": content}
    assert grader_compress._flatten_text_blocks(msg) is msg


def test_untouched_messages_keep_their_original_object(monkeypatch):
    """A no-op rewrite must not silently reshape a message on the wire."""
    blocks = [{"type": "text", "text": "x" * 4000}]
    messages = [{"role": "user", "content": blocks}, {"role": "assistant", "content": "ok"}]
    # Compressor rewrites only the second message.
    _install_fake(
        monkeypatch,
        lambda msgs, **kw: _Result(
            [dict(msgs[0]), {"role": "assistant", "content": "k"}], 400
        ),
    )
    out = grader_compress.compress_messages("gpt-5.6-sol", messages)
    assert out[0]["content"] is blocks
    assert out[1]["content"] == "k"


# --- accounting -------------------------------------------------------------

def test_stats_accumulate(monkeypatch):
    _install_fake(
        monkeypatch,
        lambda msgs, **kw: _Result([dict(m) for m in msgs], 400, before=1000, after=600),
    )
    grader_compress.compress_messages("gpt-5.6-sol", MESSAGES)
    grader_compress.compress_messages("gpt-5.6-sol", MESSAGES)
    s = grader_compress.stats()
    assert s["calls_compressed"] == 2
    assert s["tokens_saved"] == 800
    assert s["tokens_before"] == 2000
    assert s["tokens_after"] == 1200


def test_stats_is_a_copy(monkeypatch):
    grader_compress.stats()["tokens_saved"] = 99999
    assert grader_compress.stats()["tokens_saved"] == 0


def test_success_logs_the_only_positive_signal(monkeypatch, capsys):
    """"No errors" and "it worked" look identical without this line."""
    _install_fake(
        monkeypatch,
        lambda msgs, **kw: _Result([dict(m) for m in msgs], 400, before=1000, after=600),
    )
    grader_compress.compress_messages("gpt-5.6-sol", MESSAGES)
    err = capsys.readouterr().err
    assert "[grader_compress] 1000 -> 600 tokens (saved 400, 40.0%)" in err


def test_evidence_returns_compressed_text(monkeypatch):
    _install_fake(
        monkeypatch,
        lambda msgs, **kw: _Result([{"role": "user", "content": "shrunk"}], 400),
    )
    assert grader_compress.compress_evidence("gpt-5.6-sol", EVIDENCE) == "shrunk"


def test_evidence_rejects_non_string_content(monkeypatch):
    _install_fake(
        monkeypatch,
        lambda msgs, **kw: _Result([{"role": "user", "content": ["blocks"]}], 400),
    )
    assert grader_compress.compress_evidence("gpt-5.6-sol", EVIDENCE) is EVIDENCE


# --- the retrieval-stub interlock -------------------------------------------
# headroom-ai 0.24.0 does not summarise; above roughly 5K tokens it deletes the
# content and returns "[N lines compressed to 0. Retrieve more: hash=...]".
# A one-shot judge has no retrieve tool, so a stub means grading a hash.

STUB = "\n[120 lines compressed to 0. Retrieve more: hash=30dcf3f73786d50b102f6ebf]"


@pytest.mark.parametrize(
    "stub",
    [
        STUB,
        "[12 items compressed to 3. Retrieve more: hash=abc123]",
        "[5 messages compressed. hash=deadbeef]",
        "  [99 lines compressed to 0. Retrieve more: hash=x]  ",
    ],
)
def test_stub_shapes_are_detected(stub):
    assert grader_compress._is_retrieval_stub(stub) is True


@pytest.mark.parametrize(
    "text",
    [
        "tool=gmail_send args={} -> ok",
        "The agent acknowledged alarm AL-4029 [see line 12 compressed notes]",
        "",
        None,
        123,
        "[not a stub at all]",
    ],
)
def test_real_content_is_not_mistaken_for_a_stub(text):
    assert grader_compress._is_retrieval_stub(text) is False


def test_messages_stub_is_refused(monkeypatch):
    _install_fake(
        monkeypatch,
        lambda msgs, **kw: _Result(
            [dict(msgs[0]), {"role": "user", "content": STUB}], 11557, 11594, 37
        ),
    )
    assert grader_compress.compress_messages("gpt-5.6-sol", MESSAGES) is MESSAGES
    assert grader_compress.stats()["calls_refused"] == 1
    assert grader_compress.stats()["calls_compressed"] == 0


def test_evidence_stub_is_refused(monkeypatch):
    _install_fake(
        monkeypatch,
        lambda msgs, **kw: _Result([{"role": "user", "content": STUB}], 11557, 11594, 37),
    )
    assert grader_compress.compress_evidence("gpt-5.6-sol", EVIDENCE) is EVIDENCE
    assert grader_compress.stats()["calls_refused"] == 1


def test_stub_inside_a_text_block_is_refused(monkeypatch):
    _install_fake(
        monkeypatch,
        lambda msgs, **kw: _Result(
            [dict(msgs[0]), {"role": "user", "content": [{"type": "text", "text": STUB}]}],
            11557, 11594, 37,
        ),
    )
    assert grader_compress.compress_messages("gpt-5.6-sol", MESSAGES) is MESSAGES
    assert grader_compress.stats()["calls_refused"] == 1


def test_stub_refusal_warns_once(capsys, monkeypatch):
    _install_fake(
        monkeypatch,
        lambda msgs, **kw: _Result([{"role": "user", "content": STUB}], 11557, 11594, 37),
    )
    for _ in range(3):
        grader_compress.compress_evidence("gpt-5.6-sol", EVIDENCE)
    assert capsys.readouterr().err.count("retrieval stub") == 1


def test_genuine_compression_still_passes_the_guard(monkeypatch):
    """The interlock must not block real compressed prose, only stubs."""
    _install_fake(
        monkeypatch,
        lambda msgs, **kw: _Result(
            [{"role": "user", "content": "agent acked AL-4029 as false_alarm"}], 400
        ),
    )
    out = grader_compress.compress_evidence("gpt-5.6-sol", EVIDENCE)
    assert out == "agent acked AL-4029 as false_alarm"
    assert grader_compress.stats()["calls_refused"] == 0
    assert grader_compress.stats()["calls_compressed"] == 1
