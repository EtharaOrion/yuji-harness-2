#!/usr/bin/env python3
"""Proves each grading call site actually reaches the compressor.

grader_compress fails open everywhere, which means a call site that was never
wired and a call site whose compression declined look identical from the
outside -- no error either way. These tests close that gap by installing a fake
`headroom` that stamps a marker into whatever it is handed, then asserting the
marker reaches the prompt the judge would actually send.

The fake is installed into sys.modules rather than pip-installed, so the suite
runs on a host that has never seen headroom-ai.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

_SCORING = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SCORING))
sys.path.insert(0, str(_SCORING.parents[1]))

import grader_compress  # noqa: E402

MARKER = "<<COMPRESSED>>"


class _FakeResult:
    def __init__(self, messages, before, after):
        self.messages = messages
        self.tokens_before = before
        self.tokens_after = after
        self.tokens_saved = before - after


def _fake_compress(messages, model=None, config=None):
    """Replace every compressible user message with a short marker.

    Honours compress_system_messages=False and protect_recent the way the real
    library does, so a call site that protects the wrong thing fails here.
    """
    protect = (config or {}).get("protect_recent", 0)
    do_system = (config or {}).get("compress_system_messages", False)
    cutoff = len(messages) - protect if protect else len(messages)
    out = []
    for i, m in enumerate(messages):
        if i < cutoff and (m.get("role") != "system" or do_system):
            m = dict(m)
            m["content"] = MARKER
        out.append(m)
    return _FakeResult(out, before=1000, after=10)


@pytest.fixture
def headroom(monkeypatch):
    """Install the fake package and turn compression on."""
    module = types.ModuleType("headroom")
    module.compress = _fake_compress
    module.CompressConfig = lambda **kw: kw
    monkeypatch.setitem(sys.modules, "headroom", module)
    monkeypatch.setattr(grader_compress, "_HEADROOM_AVAILABLE", None)
    monkeypatch.setattr(grader_compress, "_compress", None)
    monkeypatch.setattr(grader_compress, "_CompressConfig", None)
    monkeypatch.setattr(grader_compress, "_WARNED", set())
    monkeypatch.setenv("GRADER_HEADROOM_ENABLED", "true")
    grader_compress.reset_stats()
    yield
    grader_compress.reset_stats()


# --- rubric_judge_cli: the judge that runs inside the task container --------

def _drive_cli(monkeypatch):
    import rubric_judge_cli as cli

    captured = {}

    def fake_exec(model, prompt, timeout):
        captured["prompt"] = prompt
        return json.dumps({"results": [{"number": 1, "satisfied": True, "justification": "x"}]}), {}

    monkeypatch.setattr(cli, "_codex_cli", lambda: "/usr/bin/true")
    monkeypatch.setattr(cli, "_codex_exec", fake_exec)
    monkeypatch.setattr(cli, "_record_usage_codex", lambda usage, model: None)
    cli._run_judge_codex(
        [{"number": 1, "criterion": "did the thing", "is_positive": True}],
        "tool=create_listing args={} -> ok\n" * 200,
        "FINAL ANSWER: the price is 42",
        "gpt-5.6-sol",
    )
    return captured["prompt"]


def test_cli_compresses_the_trajectory(headroom, monkeypatch):
    prompt = _drive_cli(monkeypatch)
    assert MARKER in prompt
    assert "tool=create_listing" not in prompt


def test_cli_leaves_the_final_message_alone(headroom, monkeypatch):
    """final_ctx is the artifact under judgement, never a compression target."""
    prompt = _drive_cli(monkeypatch)
    assert "FINAL ANSWER: the price is 42" in prompt


def test_cli_leaves_the_verdict_contract_alone(headroom, monkeypatch):
    prompt = _drive_cli(monkeypatch)
    assert "Return JSON: {results:" in prompt
    assert "did the thing" in prompt


def test_cli_is_untouched_when_disabled(monkeypatch):
    monkeypatch.delenv("GRADER_HEADROOM_ENABLED", raising=False)
    prompt = _drive_cli(monkeypatch)
    assert MARKER not in prompt
    assert "tool=create_listing" in prompt


# --- rubric_judge: the host-side judge --------------------------------------

def _drive_judge(monkeypatch):
    from services.scoring import rubric_judge as rj

    captured = {}

    def fake_exec(model, prompt, timeout):
        captured["prompt"] = prompt
        return json.dumps({"score": 1.0, "reason": "ok"}), {}

    monkeypatch.setattr(rj._codex_cli, "_codex_exec", fake_exec)
    messages = [
        {"role": "system", "content": "RETURN ONLY JSON"},
        {"role": "user", "content": "Trajectory (JSON):\n" + "x" * 4000},
    ]
    rj._call_judge_once("gpt-5.6-sol", messages)
    return captured["prompt"]


def test_judge_compresses_the_evidence(headroom, monkeypatch):
    prompt = _drive_judge(monkeypatch)
    assert MARKER in prompt


def test_judge_protects_the_system_contract(headroom, monkeypatch):
    """Concatenation erases the system role, so this must happen before it."""
    prompt = _drive_judge(monkeypatch)
    assert "RETURN ONLY JSON" in prompt


def test_judge_is_untouched_when_disabled(monkeypatch):
    monkeypatch.delenv("GRADER_HEADROOM_ENABLED", raising=False)
    prompt = _drive_judge(monkeypatch)
    assert MARKER not in prompt
    assert "x" * 4000 in prompt


# --- single_model_diagnostic: the failure classifier ------------------------

class _FakeResponse:
    status = 200

    async def json(self):
        return {
            "choices": [{"message": {"content": json.dumps({"primary_failure": "wrong_tool"})}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2},
        }

    async def text(self):
        return ""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def _drive_diagnostic(monkeypatch, captured):
    import asyncio

    sys.path.insert(0, str(_SCORING.parents[0] / "diagnostics"))
    import single_model_diagnostic as smd

    class _FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        def post(self, url, headers=None, json=None, timeout=None):
            captured["messages"] = json["messages"]
            return _FakeResponse()

    monkeypatch.setattr(smd.aiohttp, "ClientSession", lambda *a, **kw: _FakeSession())
    monkeypatch.setenv("EVAL_LLM_BASE_URL", "http://stub")
    monkeypatch.setenv("EVAL_LLM_API_KEY", "stub")

    cfg = smd.DiagnosisConfig(request_delay=0.0)
    client = smd.AsyncLiteLLMClient(cfg)
    messages = [{"role": "user", "content": "ENRICHED TRAJECTORY\n" + "y" * 4000}]
    asyncio.get_event_loop().run_until_complete(client.generate_content(messages))


def test_diagnostic_compresses_the_trajectory(headroom, monkeypatch):
    captured = {}
    _drive_diagnostic(monkeypatch, captured)
    assert captured["messages"][0]["content"] == MARKER


def test_diagnostic_is_untouched_when_disabled(monkeypatch):
    monkeypatch.delenv("GRADER_HEADROOM_ENABLED", raising=False)
    captured = {}
    _drive_diagnostic(monkeypatch, captured)
    assert "ENRICHED TRAJECTORY" in captured["messages"][0]["content"]


# --- the deliberate exclusion ----------------------------------------------

def test_score_claims_is_not_wired():
    """The response it judges is the artifact, not evidence about the run.

    A future refactor that routes score_claims through grader_compress would
    change what the benchmark measures. If that becomes deliberate, delete this
    test and the WHY NOT section it guards -- do not silence it.
    """
    source = (_SCORING / "score_claims.py").read_text()
    assert "grader_compress" not in source
    assert "compress_messages" not in source
