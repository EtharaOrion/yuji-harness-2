import base64
import re

import pytest

from kiro_bridge import ExecResult, KiroBridgeError, KiroConfig, sandbox_tools
from kiro_bridge.tools import clip_output


class FakeShell:
    """A tiny in-memory filesystem that understands the commands the tools emit."""

    def __init__(self):
        self.files: dict[str, str] = {"/w/a.txt": "one\ntwo\nthree\n"}
        self.commands: list[str] = []

    def __call__(self, cmd: str, timeout):
        self.commands.append(cmd)
        if m := re.fullmatch(r"cat -- (\S+)", cmd):
            path = m.group(1).strip("'")
            if path not in self.files:
                return ExecResult(1, f"cat: {path}: No such file")
            return ExecResult(0, self.files[path])
        if m := re.fullmatch(r": > (\S+)", cmd):
            self.files[m.group(1).strip("'")] = ""
            return ExecResult(0, "")
        if m := re.fullmatch(r"printf %s (\S+) \| base64 -d >> (\S+)", cmd):
            self.files[m.group(2).strip("'")] += base64.b64decode(m.group(1)).decode()
            return ExecResult(0, "")
        if cmd.startswith("mkdir -p"):
            return ExecResult(0, "")
        return ExecResult(0, f"ran: {cmd}")


def _call(tools, name, **args):
    return tools[name].execute(args, None)


def test_shell_runs_in_workdir_and_reports_exit_code():
    sh = FakeShell()
    out = _call(sandbox_tools(sh, workdir="/w"), "shell", command="pytest -q")
    assert out.startswith("exit_code: 0") and sh.commands[-1] == "cd /w && pytest -q"


def test_read_file_is_line_numbered_and_windowed():
    out = _call(sandbox_tools(FakeShell(), workdir="/w"), "read_file", path="a.txt", offset=2, limit=1)
    assert out == "     2|two"


def test_write_then_edit_file_round_trip():
    sh = FakeShell()
    tools = sandbox_tools(sh, workdir="/w")
    _call(tools, "write_file", path="b.py", content="x = 1\ny = 'ünï'\n")
    assert sh.files["/w/b.py"] == "x = 1\ny = 'ünï'\n"
    _call(tools, "edit_file", path="b.py", old_string="x = 1", new_string="x = 2")
    assert sh.files["/w/b.py"].startswith("x = 2")


def test_edit_file_rejects_ambiguous_match():
    sh = FakeShell()
    sh.files["/w/c"] = "a a"
    with pytest.raises(ValueError, match="2 times"):
        _call(sandbox_tools(sh, workdir="/w"), "edit_file", path="c", old_string="a", new_string="b")


def test_custom_read_write_channels_are_used():
    store = {"/f": "hello"}
    tools = sandbox_tools(lambda c, t: ExecResult(0, ""), read_fn=store.__getitem__,
                          write_fn=store.__setitem__)
    _call(tools, "edit_file", path="/f", old_string="hello", new_string="bye")
    assert store["/f"] == "bye"


def test_clip_output_keeps_head_and_tail():
    clipped = clip_output("A" * 50 + "B" * 50, 20)
    assert clipped.startswith("A" * 10) and clipped.endswith("B" * 10) and "cut" in clipped


def test_config_from_env():
    cfg = KiroConfig.from_env({
        "KIRO_MODEL": "claude-sonnet-5", "KIRO_JUDGE_MODEL": "claude-opus-5",
        "KIRO_AGENT_ENGINE": "v3", "KIRO_REASONING_EFFORT": "high",
        "KIRO_RUN_TIMEOUT": "60",
    })
    assert cfg.model == "claude-sonnet-5"
    assert cfg.effective_judge_model == "claude-opus-5"
    assert cfg.agent_engine == "v3" and cfg.reasoning_effort == "high"
    assert cfg.run_timeout_sec == 60


def test_config_default_model_is_auto():
    cfg = KiroConfig.from_env({})
    assert cfg.model == "auto" and cfg.effective_judge_model == "auto"


def test_config_rejects_bad_engine():
    with pytest.raises(KiroBridgeError):
        KiroConfig.from_env({"KIRO_AGENT_ENGINE": "v9"})


def test_config_rejects_bad_timeout():
    with pytest.raises(KiroBridgeError):
        KiroConfig.from_env({"KIRO_RUN_TIMEOUT": "soon"})


def test_config_rejects_bad_reasoning_effort():
    with pytest.raises(KiroBridgeError):
        KiroConfig.from_env({"KIRO_REASONING_EFFORT": "extreme"})
