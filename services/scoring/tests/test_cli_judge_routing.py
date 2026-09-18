"""The rubric judge reaches Codex through the local CLI, and nothing else.

This file used to test `_point_sdk_at_the_bridge` and then the codex-bridge
HTTP preflight. Both transports are gone: the judge now shells out to
`codex exec`, so preflight is the CLI's own `codex login status` and there is
no base URL, key, or server left to misconfigure.

What remains is worth asserting, because the absence of a thing is easy to
regress into: the judge must refuse a non-Codex model rather than quietly
finding some other way to grade, and it must fail before grading rather than
per-criterion, since rubric_judge_cli writes a score file from whatever it gets
back.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

cli = pytest.importorskip("services.scoring.rubric_judge_cli")


def test_the_claude_transport_is_gone():
    """No Agent SDK, no output-schema contract, no SDK usage walker."""
    src = Path(cli.__file__).read_text()
    for gone in ("ClaudeAgentOptions", "ClaudeSDKClient", "structured_output"):
        assert gone not in src, gone
    for attr in ("_OUTPUT_SCHEMA", "_record_usage", "_point_sdk_at_the_bridge"):
        assert not hasattr(cli, attr), attr


def test_no_claude_credential_juggling_remains():
    """The direct transport carries its key on the request, so none of the
    `claude` CLI's credential precedence is this module's concern any more."""
    src = Path(cli.__file__).read_text()
    for gone in ("ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"):
        assert f'os.environ["{gone}"]' not in src, gone


def test_a_non_codex_model_is_refused():
    import asyncio
    with pytest.raises(cli.JudgeResponseError) as e:
        asyncio.run(cli._run_judge([], "t", "f", "claude-sonnet-4-6"))
    assert "gpt-5.6-sol" in str(e.value)


def test_preflight_passes_when_codex_is_logged_in(monkeypatch):
    monkeypatch.setattr(cli, "_codex_credential_error", lambda: "")
    assert cli._preflight("gpt-5.6-sol") is None


def test_preflight_reports_a_missing_cli(monkeypatch):
    """No `codex` on PATH must be caught before grading, not per-criterion."""
    monkeypatch.setattr(cli, "_codex_cli", lambda: "")
    reason = cli._preflight("gpt-5.6-sol")
    assert reason is not None and "codex" in reason


def test_preflight_asks_the_cli_itself(monkeypatch):
    """`codex login status` is the check — no credential store is read."""
    monkeypatch.setattr(cli, "_codex_cli", lambda: "/usr/local/bin/codex")
    seen: dict = {}

    class _Proc:
        returncode = 1
        stdout = ""
        stderr = "Not logged in"

    def fake(argv, capture_output=None, text=None, timeout=None):
        seen["argv"] = argv
        return _Proc()

    monkeypatch.setattr(cli.subprocess, "run", fake)
    reason = cli._preflight("gpt-5.6-sol")
    assert seen["argv"] == ["/usr/local/bin/codex", "login", "status"]
    assert reason is not None and "codex login" in reason


def test_preflight_ignores_a_non_codex_model(monkeypatch):
    """The CLI is not consulted there; _run_judge does the refusing."""
    monkeypatch.setattr(cli, "_codex_cli", lambda: "")
    assert cli._preflight("claude-sonnet-4-6") is None


def test_the_cli_default_model_is_the_codex_model(monkeypatch):
    """On a host with codex installed, the pinned Codex grader still wins.

    Was a source-text assertion on the literal argparse default. The default is
    now resolved at runtime, because tests/test.sh invokes this CLI with no
    --model and the task container has no codex binary -- so a fixed default
    meant the bundle rubric channel could never be graded at all. Asserting the
    behaviour instead of the source keeps the guarantee that mattered.
    """
    monkeypatch.delenv("JUDGE_MODEL", raising=False)
    monkeypatch.setattr(cli, "_codex_cli", lambda: "/usr/local/bin/codex")
    assert cli._default_judge_model() == cli.CODEX_MODELS[0]


def test_an_explicit_judge_model_is_never_overridden(monkeypatch):
    """A pinned grader is a benchmark property; resolution must not touch it."""
    monkeypatch.setenv("JUDGE_MODEL", "gpt-5.6-sol")
    monkeypatch.setattr(cli, "_codex_cli", lambda: "")
    monkeypatch.setattr(cli, "_claude_cli", lambda: "/root/.local/bin/claude")
    assert cli._default_judge_model() == "gpt-5.6-sol"


def test_container_without_codex_falls_back_to_claude(monkeypatch):
    """The case this transport exists for: the task container."""
    monkeypatch.delenv("JUDGE_MODEL", raising=False)
    monkeypatch.setattr(cli, "_codex_cli", lambda: "")
    monkeypatch.setattr(cli, "_claude_cli", lambda: "/root/.local/bin/claude")
    assert cli._default_judge_model() == cli.CLAUDE_MODELS[0]


def test_neither_cli_keeps_the_codex_default(monkeypatch):
    monkeypatch.delenv("JUDGE_MODEL", raising=False)
    monkeypatch.setattr(cli, "_codex_cli", lambda: "")
    monkeypatch.setattr(cli, "_claude_cli", lambda: "")
    assert cli._default_judge_model() == cli.CODEX_MODELS[0]


def test_claude_preflight_requires_a_credential(monkeypatch):
    monkeypatch.setattr(cli, "_claude_cli", lambda: "/root/.local/bin/claude")
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    reason = cli._preflight("claude-sonnet-4-5")
    assert reason is not None and "CLAUDE_CODE_OAUTH_TOKEN" in reason
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "tok")
    assert cli._preflight("claude-sonnet-4-5") is None


def test_claude_preflight_reports_a_missing_cli(monkeypatch):
    monkeypatch.setattr(cli, "_claude_cli", lambda: "")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "tok")
    reason = cli._preflight("claude-sonnet-4-5")
    assert reason is not None and "claude CLI not found" in reason
