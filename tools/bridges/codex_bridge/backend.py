"""Judge runs through the `codex exec --json` CLI, on the operator's login.

Each request becomes one headless

    codex exec --json --ephemeral --skip-git-repo-check --ignore-user-config \\
        --ignore-rules -s read-only -m <model> [-c model_reasoning_effort=<e>] -

process. Its JSON-Lines stream is parsed into text, usage and errors so the
HTTP layer never branches on the transport.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import threading
from typing import Any

from codex_bridge.config import CodexBridgeError, CodexConfig, VALID_EFFORTS


class CodexBackend:
    def __init__(self, config: CodexConfig | None = None) -> None:
        self.config = config or CodexConfig.from_env()
        self._slots = threading.BoundedSemaphore(max(1, self.config.concurrency))

    @property
    def codex_bin(self) -> str:
        return self.config.require_cli()

    def models(self) -> list[dict[str, str]]:
        model = self.config.judge_model
        return [{"id": model, "name": model}]

    def complete(self, system: str, user: str, **kwargs: Any) -> tuple[str, dict[str, Any]]:
        text, usage, _ = self.complete_with_reasoning(system, user, **kwargs)
        return text, usage

    def complete_with_reasoning(self, system: str, user: str, *, model: str,
                                timeout_sec: float | None = None,
                                effort: str = "") -> tuple[str, dict[str, Any], str]:
        prompt = f"{system}\n\n{user}" if system else user
        cmd = [self.codex_bin, "exec", "--json", "--ephemeral",
               "--skip-git-repo-check", "--ignore-user-config", "--ignore-rules",
               "-s", "read-only", "-m", model]
        effort = (effort or self.config.reasoning_effort or "").lower()
        # _requested_effort returns "none" when the caller disabled thinking.
        # Codex takes `none` as-is for gpt-5.6-sol (and rejects `minimal` there
        # with a 400), so it is passed through, never translated.
        if effort in set(VALID_EFFORTS):
            cmd += ["-c", f"model_reasoning_effort={effort}"]
        cmd.append("-")

        with self._slots, tempfile.TemporaryDirectory(prefix="codex-judge-") as cwd:
            try:
                proc = subprocess.run(cmd, input=prompt, capture_output=True,
                                      text=True, cwd=cwd, timeout=timeout_sec)
            except FileNotFoundError as err:
                raise CodexBridgeError(f"codex did not start: {err}") from err
            except subprocess.TimeoutExpired as err:
                raise CodexBridgeError(
                    f"codex exec timed out after {timeout_sec}s"
                ) from err

        texts: list[str] = []
        usage: dict[str, Any] = {}
        errors: list[str] = []
        for line in proc.stdout.splitlines():
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            kind = ev.get("type")
            item = ev.get("item") or {}
            if kind == "item.completed" and item.get("type") == "agent_message":
                texts.append(str(item.get("text") or ""))
            elif kind == "turn.completed":
                u = ev.get("usage") or {}
                cached = int(u.get("cached_input_tokens") or 0)
                # OpenAI counts cached tokens inside input_tokens; Anthropic
                # (and this bridge's wire format) does not. Subtract so the
                # emitted input_tokens matches Anthropic's semantics.
                usage = {
                    "input_tokens": max(0, int(u.get("input_tokens") or 0) - cached),
                    "output_tokens": int(u.get("output_tokens") or 0),
                    "cache_read_tokens": cached,
                    "cache_write_tokens": int(u.get("cache_write_input_tokens") or 0),
                    "reasoning_tokens": int(u.get("reasoning_output_tokens") or 0),
                    "request_count": 1,
                }
            elif kind in ("turn.failed", "error"):
                errors.append(
                    json.dumps(ev.get("error") or ev.get("message") or ev)[:500]
                )

        if proc.returncode != 0 or errors or not texts:
            detail = (
                "; ".join(errors) or proc.stderr.strip()[-500:] or "no agent_message"
            )
            raise CodexBridgeError(
                f"codex exec failed (rc={proc.returncode}): {detail}"
            )
        return texts[-1], usage, ""
