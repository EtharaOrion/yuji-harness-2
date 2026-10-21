"""Coverage for the argv run_cli_agent builds.

Review found that the suite caught 4 of 8 sabotages but missed four whose root
cause was that run_cli_agent's constructed argv is never asserted anywhere.
These tests assert the argv shape instead of trusting a happy-path end-to-end:

  * --model <id> is always passed (so a sabotage that drops it is noisy)
  * tools=[] produces --trust-tools= (confining the judge; a sabotage that
    disables confinement is noisy)
  * --effort <id> is passed when effort is set (a sabotage that drops it
    cannot ride on the test being green)
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from kiro_bridge import cli_backend
from kiro_bridge.cli_backend import AgentResult, Usage, run_cli_agent


class _FakeConfig:
    model = "claude-sonnet-4.5"
    run_timeout_sec = 30.0
    reasoning_effort = ""
    agent_engine = ""

    def require_cli(self) -> str:
        return "/usr/local/bin/kiro-cli"


class _FakeBridge:
    def __init__(self) -> None:
        self.config = _FakeConfig()


def _ok_result() -> AgentResult:
    return AgentResult(
        status="finished", text="ok", agent_id="s", run_id="s",
        model="claude-sonnet-4.5", runtime="local", duration_sec=0.0,
        usage=Usage(credits=0.0, request_count=1), events=[], error="",
    )


def _capture_argv(effort: str | None = None, tools=None):
    """Call run_cli_agent with _drive stubbed so we can inspect the argv."""
    captured: dict[str, list[str]] = {}

    def _fake_drive(argv, workspace, config, model, timeout, on_event, cancel_signal):
        captured["argv"] = list(argv)
        captured["workspace"] = str(workspace)
        return _ok_result()

    with patch.object(cli_backend, "_drive", _fake_drive):
        run_cli_agent(_FakeBridge(), prompt="hello", effort=effort, tools=tools)
    return captured


def test_argv_always_contains_model():
    cap = _capture_argv()
    assert "--model" in cap["argv"]
    i = cap["argv"].index("--model")
    assert cap["argv"][i + 1] == "claude-sonnet-4.5"


def test_argv_contains_trust_tools_empty_when_tools_empty():
    cap = _capture_argv(tools=[])
    # the exact token the review named: --trust-tools= (empty rhs when
    # there are no MCP custom tools either)
    assert any(a.startswith("--trust-tools=") for a in cap["argv"]), cap["argv"]
    # confinement also implies a scratch workspace, not cwd
    assert "/kiro-cli-" in cap["workspace"]


def test_argv_passes_effort_when_set():
    cap = _capture_argv(effort="high")
    assert "--effort" in cap["argv"]
    i = cap["argv"].index("--effort")
    assert cap["argv"][i + 1] == "high"


def test_argv_omits_effort_when_unset():
    cap = _capture_argv(effort=None)
    assert "--effort" not in cap["argv"]
