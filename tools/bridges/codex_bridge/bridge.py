from __future__ import annotations

import atexit
import subprocess
import threading
from dataclasses import dataclass

from codex_bridge.backend import CodexBackend
from codex_bridge.config import CodexBridgeError, CodexConfig


@dataclass(frozen=True)
class DoctorReport:
    ok: bool
    cli_present: bool
    judge_model: str = ""
    cli_path: str = ""
    error: str = ""


def doctor(config: CodexConfig | None = None) -> DoctorReport:
    config = config or CodexConfig.from_env()
    try:
        cli = config.require_cli()
    except CodexBridgeError as err:
        return DoctorReport(False, False, error=str(err))
    problem = _login_problem(cli)
    if problem:
        return DoctorReport(False, True, cli_path=cli, error=problem)
    return DoctorReport(True, True, judge_model=config.judge_model, cli_path=cli)


def credential_problem(config: CodexConfig | None = None) -> str:
    config = config or CodexConfig.from_env()
    try:
        cli = config.require_cli()
    except CodexBridgeError as err:
        return str(err)
    return _login_problem(cli)


def _login_problem(cli: str) -> str:
    """Why `codex exec` would fail here, or "" when it would run.

    `codex login status` returns 0 and prints the account on a logged-in host.
    Any other outcome is reported with its stdout+stderr as the message so the
    caller can show the operator exactly what codex said.
    """
    status = subprocess.run([cli, "login", "status"], capture_output=True, text=True)
    if status.returncode != 0:
        combined = (status.stdout + status.stderr).strip()
        return combined or f"codex login status exit {status.returncode}"
    return ""


_SHARED: CodexBackend | None = None
_SHARED_LOCK = threading.Lock()


def shared_bridge(config: CodexConfig | None = None) -> CodexBackend:
    global _SHARED
    with _SHARED_LOCK:
        wanted = config or CodexConfig.from_env()
        if _SHARED is None or _SHARED.config != wanted:
            _SHARED = CodexBackend(wanted)
        return _SHARED


def close_shared_bridge() -> None:
    global _SHARED
    with _SHARED_LOCK:
        _SHARED = None


atexit.register(close_shared_bridge)


__all__ = [
    "CodexBridgeError",
    "DoctorReport",
    "close_shared_bridge",
    "credential_problem",
    "doctor",
    "shared_bridge",
]
