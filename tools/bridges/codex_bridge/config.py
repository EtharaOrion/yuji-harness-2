"""Settings for the codex bridge, read from the environment or passed explicitly.

The Codex CLI (`codex`) authenticates through its own login (`codex login`,
ChatGPT subscription) -- there is no API key passed through the bridge. Every
request becomes one `codex exec --json` process on the operator's login, so a
container only ever holds the minted keys this bridge issues, which are valid
for the lifetime of the process.

    CODEX_BIN                 path to the `codex` CLI (found on PATH)
    CODEX_JUDGE_MODEL         model the judge key is pinned to  (gpt-5.6-sol)
    CODEX_REASONING_EFFORT    none|minimal|low|medium|high|xhigh|max (request's own)
    CODEX_BRIDGE_CONCURRENCY  max parallel `codex exec` processes    (8)
    CODEX_RUN_TIMEOUT         seconds before a run is cancelled   (1800)
    CODEX_MAX_ATTEMPTS        attempts on retryable startup errors    (3)
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

CLI_NAME = "codex"
_CLI_FALLBACKS = (
    Path.home() / ".local" / "bin" / CLI_NAME,
    Path("/usr/local/bin") / CLI_NAME,
    Path("/opt/homebrew/bin") / CLI_NAME,
)

DEFAULT_JUDGE_MODEL = "gpt-5.6-sol"
DEFAULT_RUN_TIMEOUT_SEC = 1800
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_CONCURRENCY = 8
# What `codex exec -c model_reasoning_effort=` accepts. gpt-5.6-sol takes none,
# low, medium, high, xhigh, max and REJECTS minimal (checked 2026-10-27);
# minimal stays for the gpt-5.x-codex models that do take it.
VALID_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")

_AUTH_PATH = Path.home() / ".codex" / "auth.json"


class CodexBridgeError(Exception):
    """A failure the bridge raises deliberately (bad config, run never started)."""


@dataclass(frozen=True)
class CodexConfig:
    judge_model: str = DEFAULT_JUDGE_MODEL
    reasoning_effort: str = ""
    run_timeout_sec: float = DEFAULT_RUN_TIMEOUT_SEC
    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    concurrency: int = DEFAULT_CONCURRENCY
    cli_path: str = ""

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None, **overrides) -> "CodexConfig":
        e = os.environ if env is None else env
        effort = e.get("CODEX_REASONING_EFFORT", "").strip().lower()
        if effort and effort not in VALID_EFFORTS:
            raise CodexBridgeError(
                f"CODEX_REASONING_EFFORT={effort!r} -- use one of {', '.join(VALID_EFFORTS)}"
            )
        config = cls(
            judge_model=e.get("CODEX_JUDGE_MODEL", "").strip() or DEFAULT_JUDGE_MODEL,
            reasoning_effort=effort,
            run_timeout_sec=_number(e, "CODEX_RUN_TIMEOUT", DEFAULT_RUN_TIMEOUT_SEC),
            max_attempts=max(1, int(_number(e, "CODEX_MAX_ATTEMPTS", DEFAULT_MAX_ATTEMPTS))),
            concurrency=max(1, int(_number(e, "CODEX_BRIDGE_CONCURRENCY", DEFAULT_CONCURRENCY))),
            cli_path=e.get("CODEX_BIN", "").strip(),
        )
        return replace(config, **overrides) if overrides else config

    def find_cli(self) -> str | None:
        if self.cli_path:
            return self.cli_path if Path(self.cli_path).is_file() else None
        found = shutil.which(CLI_NAME)
        if found:
            return found
        return next((str(p) for p in _CLI_FALLBACKS if p.is_file()), None)

    def require_cli(self) -> str:
        cli = self.find_cli()
        if cli is None:
            where = self.cli_path or f"{CLI_NAME} on PATH"
            raise CodexBridgeError(
                f"no Codex CLI ({where}) -- install the Codex CLI and run `{CLI_NAME} login`, "
                f"or set CODEX_BIN to its path"
            )
        return cli

    def auth_path(self) -> Path:
        return _AUTH_PATH

    def require_credentials(self) -> None:
        """Raise unless the CLI is present (login is checked separately)."""
        self.require_cli()


def _number(env, key: str, default: float) -> float:
    raw = env.get(key, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as err:
        raise CodexBridgeError(f"{key}={raw!r} is not a number") from err
