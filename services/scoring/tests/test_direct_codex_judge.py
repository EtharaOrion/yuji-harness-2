"""The rubric judge driving the local Codex CLI, with no server in the path.

The judge used to post at codex-bridge's /v1/responses; before that it went
through claude_agent_sdk and the `claude` CLI. Both are gone. The transport is
now the one devops-projects' judge council uses for its `codex/` seat:
`codex exec --json --sandbox read-only`, prompt on stdin, verdict parsed out
of the CLI's JSONL event stream. It spends a ChatGPT subscription, so the
only credential is a `codex login` the machine already holds.

The tests that matter here are the negative ones. `_run_judge_codex` must
raise rather than return an empty result set, because rubric_judge_cli writes
a score file from whatever it gets back: a judge that quietly returns nothing
produces a rubric of zeros that reads exactly like a graded run.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

cli = pytest.importorskip("services.scoring.rubric_judge_cli")

_CRITERIA = [{"number": "1", "criterion": "did the thing", "is_positive": True}]
_REPLY = {"results": [{"number": "1", "satisfied": True, "justification": "yes"}]}


def _jsonl(*events: dict) -> str:
    return "\n".join(json.dumps(e) for e in events)


def _verdict_stream(reply_text: str, usage: dict | None = None) -> str:
    """A codex exec event stream: narration, then the verdict, then usage."""
    return _jsonl(
        {"item": {"type": "agent_message", "text": "Let me review the trajectory."}},
        {"item": {"type": "agent_message", "text": reply_text}},
        {"type": "turn.completed", "usage": usage or {}},
    )


class _Proc:
    def __init__(self, stdout: str = "", returncode: int = 0, stderr: str = ""):
        self.stdout = stdout
        self.returncode = returncode
        self.stderr = stderr


@pytest.fixture
def codex(monkeypatch):
    """A codex CLI that exists; each test decides what running it does."""
    monkeypatch.setattr(cli, "_codex_cli", lambda: "/usr/local/bin/codex")
    # Retries sleep for real; tests never should.
    monkeypatch.setattr(cli.time, "sleep", lambda s: None)


def _stub_run(monkeypatch, proc: _Proc, capture: dict | None = None):
    def fake(argv, input=None, capture_output=None, text=None, timeout=None, cwd=None):
        if capture is not None:
            capture["argv"] = argv
            capture["input"] = input
            capture["cwd"] = cwd
        return proc

    monkeypatch.setattr(cli.subprocess, "run", fake)


def test_the_transport_is_the_codex_cli(monkeypatch, codex):
    """`codex exec --json` in a read-only sandbox, prompt on stdin, scratch cwd."""
    seen: dict = {}
    _stub_run(monkeypatch, _Proc(_verdict_stream(json.dumps(_REPLY))), seen)
    out = cli._run_judge_codex(_CRITERIA, "traj-ctx", "final-ctx", "gpt-5.6-sol")
    assert out == _REPLY["results"]
    assert seen["argv"] == [
        "/usr/local/bin/codex", "exec", "--json", "--sandbox", "read-only",
        "--skip-git-repo-check", "-m", "gpt-5.6-sol",
    ]
    # The prompt is the same one every previous transport sent, on stdin —
    # `codex exec` has no system role.
    assert seen["input"] == cli._judge_prompt(_CRITERIA, "traj-ctx", "final-ctx")
    # A scratch cwd, so the read-only agent cannot see the trial it judges.
    assert seen["cwd"] == tempfile.gettempdir()


def test_the_last_agent_message_wins(monkeypatch, codex):
    """The agent may narrate before delivering the verdict block."""
    stream = _jsonl(
        {"item": {"type": "agent_message", "text": "Thinking about it."}},
        {"item": {"type": "agent_message", "text": json.dumps(_REPLY)}},
        {"type": "turn.completed", "usage": {}},
    )
    _stub_run(monkeypatch, _Proc(stream))
    assert cli._run_judge_codex(_CRITERIA, "t", "f", "gpt-5.6-sol") == _REPLY["results"]


def test_json_wrapped_in_prose_is_recovered(monkeypatch, codex):
    """No structured-output guarantee on this path, so the reply may be fenced."""
    fenced = "Here you go:\n```json\n" + json.dumps(_REPLY) + "\n```"
    _stub_run(monkeypatch, _Proc(_verdict_stream(fenced)))
    assert cli._run_judge_codex(_CRITERIA, "t", "f", "gpt-5.6-sol") == _REPLY["results"]


def test_non_json_stdout_lines_are_ignored(monkeypatch, codex):
    stream = "warning: something\n" + _verdict_stream(json.dumps(_REPLY))
    _stub_run(monkeypatch, _Proc(stream))
    assert cli._run_judge_codex(_CRITERIA, "t", "f", "gpt-5.6-sol") == _REPLY["results"]


# ----- the failures that must not be silent -----


def test_an_unparseable_reply_raises(monkeypatch, codex):
    _stub_run(monkeypatch, _Proc(_verdict_stream("I cannot grade this.")))
    with pytest.raises(cli.JudgeResponseError):
        cli._run_judge_codex(_CRITERIA, "t", "f", "gpt-5.6-sol")


def test_no_agent_message_raises_after_retries(monkeypatch, codex):
    monkeypatch.setattr(cli, "_MAX_ATTEMPTS", 1)
    _stub_run(monkeypatch, _Proc(_jsonl({"type": "turn.completed", "usage": {}})))
    with pytest.raises(cli.JudgeResponseError) as e:
        cli._run_judge_codex(_CRITERIA, "t", "f", "gpt-5.6-sol")
    assert "no agent message" in str(e.value)


def test_a_nonzero_exit_surfaces_stderr(monkeypatch, codex):
    monkeypatch.setattr(cli, "_MAX_ATTEMPTS", 1)
    _stub_run(monkeypatch, _Proc("", returncode=2, stderr="stream error: 429"))
    with pytest.raises(cli.JudgeResponseError) as e:
        cli._run_judge_codex(_CRITERIA, "t", "f", "gpt-5.6-sol")
    assert "exited 2" in str(e.value) and "429" in str(e.value)


def test_a_missing_cli_refuses_before_any_subprocess(monkeypatch):
    monkeypatch.setattr(cli, "_codex_cli", lambda: "")

    def explode(*a, **k):
        raise AssertionError("ran a subprocess without a CLI")

    monkeypatch.setattr(cli.subprocess, "run", explode)
    with pytest.raises(cli.JudgeResponseError):
        cli._run_judge_codex(_CRITERIA, "t", "f", "gpt-5.6-sol")


def test_a_transient_failure_is_retried_then_succeeds(monkeypatch, codex):
    calls = {"n": 0}

    def flaky(argv, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return _Proc("", returncode=1, stderr="upstream hiccup")
        return _Proc(_verdict_stream(json.dumps(_REPLY)))

    monkeypatch.setattr(cli.subprocess, "run", flaky)
    out = cli._run_judge_codex(_CRITERIA, "t", "f", "gpt-5.6-sol")
    assert out == _REPLY["results"]
    assert calls["n"] == 2


# ----- usage accounting -----


def test_usage_is_recorded_in_the_finance_shape(monkeypatch, codex, tmp_path):
    """Codex reports OpenAI-style inclusive counts; the cached portion is
    subtracted so the fields stay disjoint, and the call is valued at the
    judge model's list price so judge effort is visible in Finance rather
    than reading $0.00 on every subscription-quota run."""
    out = tmp_path / "judge_tokens.json"
    monkeypatch.setattr(cli, "_token_out", out)
    usage = {"input_tokens": 15_000, "cached_input_tokens": 11_000,
             "output_tokens": 30, "cache_write_input_tokens": 0}
    _stub_run(monkeypatch, _Proc(_verdict_stream(json.dumps(_REPLY), usage)))
    cli._run_judge_codex(_CRITERIA, "t", "f", "gpt-5.6-sol")
    line = json.loads(out.read_text())[0]
    assert line["judge_input_tokens"] == 4_000
    assert line["judge_input_cache_tokens"] == 11_000
    assert line["judge_output_tokens"] == 30
    # 4_000 in @ $5/Mtok + 11_000 cache-read @ $0.50/Mtok + 30 out @ $30/Mtok
    assert line["judge_cost_usd"] == pytest.approx(0.0264)


def test_cost_is_zero_for_a_model_with_no_known_rate(monkeypatch, codex, tmp_path):
    """An unrecognised judge prices at nothing rather than borrowing a rate."""
    out = tmp_path / "judge_tokens.json"
    monkeypatch.setattr(cli, "_token_out", out)
    monkeypatch.setattr(cli, "_usage_lines", [])
    cli._record_usage_codex(
        {"input_tokens": 15_000, "cached_input_tokens": 11_000,
         "output_tokens": 30, "cache_write_input_tokens": 0}, "not-a-real-model")
    assert json.loads(out.read_text())[0]["judge_cost_usd"] == 0.0


def test_rates_can_be_zeroed_to_stop_valuing_judge_calls(monkeypatch, codex, tmp_path):
    """Setting every JUDGE_PRICE_* to 0 restores plain zero-cost reporting."""
    for var in cli._RATE_ENV.values():
        monkeypatch.setenv(var, "0")
    out = tmp_path / "judge_tokens.json"
    monkeypatch.setattr(cli, "_token_out", out)
    monkeypatch.setattr(cli, "_usage_lines", [])
    cli._record_usage_codex(
        {"input_tokens": 15_000, "cached_input_tokens": 11_000,
         "output_tokens": 30, "cache_write_input_tokens": 0}, "gpt-5.6-sol")
    assert json.loads(out.read_text())[0]["judge_cost_usd"] == 0.0
