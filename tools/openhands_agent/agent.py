"""OpenHands as this harness's agent, on the host's Claude subscription.

Harbor imports this class from the harness checkout:

    PYTHONPATH=<harness> harbor run --agent tools.openhands_agent.agent:OpenHandsAgent ...

scripts/run_task.sh does that when AGENT=openhands (the default). It replaces
Harbor's claude-code agent the way the reference harness replaced its `claude
-p` lane: the OpenHands SDK drives the loop, and the model is reached through
the ccbridge, a local Anthropic-compatible proxy that signs every request with
the Claude Code OAuth login already on this machine.

    main container                         host
    ─────────────────────────────          ──────────────────────────────────
    runner.py (OpenHands SDK)              tools/bridges/ccbridge  :8765
      LiteLLM  anthropic/claude-opus-5 ─►    checks the shared secret,
      base_url http://host.docker.internal     swaps in the OAuth bearer token,
      api_key  <bridge secret>                 adds the Claude Code system
                                               prefix + billing attribution  ─► api.anthropic.com

WHY THIS SHAPE

  * No credential for the subscription enters the container. The agent holds
    only the bridge's shared secret, which can do one thing: ask the bridge for
    a completion. Harbor's claude-code agent needs the OAuth token itself inside
    the container the agent has root in.
  * Nothing is installed at trial time. The agent phase runs with the network
    closed (tools/network/egress-proxy/overlay.yaml), so the SDK comes from the
    openhands-runtime image, mounted read-only at /opt/openhands-runtime by
    tools/openhands_agent/overlay.yaml. install() only checks it is there and
    uploads the runner. A missing runtime is a refusal, never a pip install.
  * The instruction travels as a file. Harbor's own openhands-sdk agent puts it
    on the command line, which is the ARG_MAX failure scripts/patch_harbor.py
    had to fix for claude-code.
"""

from __future__ import annotations

import json
import os
import shlex
import tempfile
from pathlib import Path
from typing import Any, ClassVar, Mapping, override

from pydantic import Field

from harbor.agents.capabilities import AgentCapabilities
from harbor.agents.installed.base import (
    AgentAuthenticationError,
    ApiUsageLimitError,
    BaseInstalledAgent,
    ErrorPattern,
    NetworkConnectionError,
    NonZeroAgentExitCodeError,
    with_prompt_template,
)
from harbor.agents.options import InstalledAgentOptions
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

RUNTIME_ROOT = "/opt/openhands-runtime"
RUNTIME_PYTHON = f"{RUNTIME_ROOT}/venv/bin/python"
RUNNER_REMOTE = "/installed-agent/openhands_runner.py"
INSTRUCTION_REMOTE = "/installed-agent/instruction.md"
RUNNER_LOCAL = Path(__file__).with_name("runner.py")

# Where the model is, and the secret that opens it. Read from harbor's own
# environment (or --ae), set by scripts/run_task.sh from the running bridge.
BASE_URL_ENV = "OPENHANDS_LLM_BASE_URL"
API_KEY_ENV = "OPENHANDS_LLM_API_KEY"

# The runner writes these under /logs/agent; populate_context_post_run reads
# the trajectory back through harbor's bind mount.
TRAJECTORY_FILENAME = "trajectory.json"
RUNNER_LOG_FILENAME = "openhands-runner.log"

# Importing the SDK imports LiteLLM, which fetches its price map from
# raw.githubusercontent.com unless told not to (see runner.py main()).
_QUIET_IMPORT = ("OPENHANDS_SUPPRESS_BANNER=1 LITELLM_LOCAL_MODEL_COST_MAP=True "
                 "LITELLM_LOCAL_ANTHROPIC_BETA_HEADERS=True")


def litellm_model(model: str) -> str:
    """The model id LiteLLM routes on.

    run_task.sh keeps MODEL=claude-opus-5 -- that name is how every report,
    trajectory directory and finance record in this harness is keyed -- and the
    bridge speaks Anthropic's wire protocol, so a bare id is an Anthropic one.
    An id that already names a provider is passed through untouched.
    """
    model = model.strip()
    return model if "/" in model else f"anthropic/{model}"


class OpenHandsAgentOptions(InstalledAgentOptions):
    max_iterations: int = Field(
        default=500, ge=1,
        description="Agent steps per conversation run (the SDK's max_iteration_per_run).",
    )
    max_continuations: int = Field(
        default=6, ge=0,
        description="Times an agent that ended on a message is told to carry on.",
    )
    reasoning_effort: str | None = Field(
        default=None, description="Passed to the SDK LLM; unset keeps the SDK default.",
    )
    max_output_tokens: int | None = Field(
        default=32000, ge=1,
        description="Output ceiling per model call. The SDK's own default is 16384.",
    )
    llm_timeout: int = Field(
        default=1800, ge=1, description="Seconds one model call may take.",
    )
    num_retries: int = Field(
        default=5, ge=0, description="SDK retries per model call.",
    )
    temperature: float | None = Field(
        default=None, description="Sampling temperature; dropped by the SDK for thinking models.",
    )
    thinking_display: str = Field(
        default="summarized", pattern="^(omitted|summarized)$",
        description="Adaptive thinking display: 'summarized' puts readable thinking in "
                    "the trajectory; 'omitted' withholds the text (README, Thinking).",
    )


class OpenHandsAgent(BaseInstalledAgent):
    capabilities = AgentCapabilities(atif=True)
    options_model = OpenHandsAgentOptions
    options: OpenHandsAgentOptions

    # Harbor's generic patterns plus the bridge's own failure wordings. The
    # match that ends LAST in the output wins, so order only breaks ties.
    ERROR_PATTERNS: ClassVar[list[ErrorPattern]] = [
        *BaseInstalledAgent.ERROR_PATTERNS,
        ErrorPattern(r"APIConnectionError", NetworkConnectionError),
        ErrorPattern(r"ccbridge: missing/invalid bridge secret", AgentAuthenticationError),
        ErrorPattern(r"credentials_unavailable|OAuth refresh failed", AgentAuthenticationError),
        ErrorPattern(r"subscription_cap|\ball \d+ accounts exhausted\b", ApiUsageLimitError),
        ErrorPattern(r"OpenHands runtime is not mounted", NonZeroAgentExitCodeError),
    ]

    # How the process died outranks what its output mentions (the same rule
    # scripts/patch_harbor.py installs into harbor's claude-code agent).
    _FATAL_SIGNALS: ClassVar[dict[int, str]] = {
        137: "killed by SIGKILL (128+9) -- in a container this is normally the "
             "OOM killer; compare the task's memory_mb with what docker has",
        139: "killed by SIGSEGV (128+11) -- agent process crashed",
        143: "killed by SIGTERM (128+15) -- stopped by an external signal",
        124: "exit 124 -- command timed out (coreutils timeout)",
    }

    @staticmethod
    @override
    def name() -> str:
        return "openhands"

    @classmethod
    @override
    def preflight(cls, kwargs: dict[str, Any] | None = None,
                  env: Mapping[str, str] | None = None) -> None:
        super().preflight(kwargs, env)
        env = env or {}
        missing = [k for k in (BASE_URL_ENV, API_KEY_ENV)
                   if not (env.get(k) or os.environ.get(k))]
        if missing:
            raise ValueError(
                f"the openhands agent needs {' and '.join(missing)}: the ccbridge URL "
                "as the container sees it and the bridge's shared secret. "
                "scripts/run_task.sh sets both after starting the bridge."
            )

    @override
    def get_version_command(self) -> str | None:
        return (f"{_QUIET_IMPORT} {RUNTIME_PYTHON} -c "
                "'import openhands.sdk as s; print(s.__version__)' 2>/dev/null")

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        probe = await environment.exec(
            command=(f"[ -x {RUNTIME_PYTHON} ] && {_QUIET_IMPORT} "
                     f"{RUNTIME_PYTHON} -c 'import openhands.sdk, openhands.tools'"),
        )
        if probe.return_code != 0:
            raise RuntimeError(
                f"OpenHands runtime is not mounted at {RUNTIME_ROOT} "
                f"(probe exit {probe.return_code}: {self._truncate_output(probe.stderr, 400)}). "
                "It comes from the openhands-runtime image through "
                "tools/openhands_agent/overlay.yaml, which scripts/run_task.sh adds "
                "for AGENT=openhands. It is never pip-installed here: the agent "
                "phase has no route to an index."
            )
        await environment.upload_file(source_path=RUNNER_LOCAL, target_path=RUNNER_REMOTE)
        await environment.exec(command=f"chmod 0755 {RUNNER_REMOTE}", user="root")

    @override
    def populate_context_post_run(self, context: AgentContext) -> None:
        path = self.logs_dir / TRAJECTORY_FILENAME
        try:
            metrics = (json.loads(path.read_text()) or {}).get("final_metrics") or {}
        except (OSError, ValueError) as exc:
            self.logger.debug(f"no readable trajectory at {path}: {exc}")
            return
        # total_prompt_tokens is LiteLLM's figure, which already includes the
        # cache reads and writes -- the same "total input" harbor records for
        # claude-code in n_input_tokens.
        context.n_input_tokens = metrics.get("total_prompt_tokens") or 0
        context.n_output_tokens = metrics.get("total_completion_tokens") or 0
        context.n_cache_tokens = metrics.get("total_cached_tokens") or 0
        context.cost_usd = metrics.get("total_cost_usd")

    @override
    def _classify_exec_error(self, command: str, result: Any) -> NonZeroAgentExitCodeError:
        note = self._FATAL_SIGNALS.get(getattr(result, "return_code", None))
        if note:
            return NonZeroAgentExitCodeError(
                f"{note}\nCommand failed (exit {result.return_code}): "
                f"{self._redact_command(command)}\n"
                f"stdout: {self._truncate_output(result.stdout)}\n"
                f"stderr: {self._truncate_output(result.stderr)}"
            )
        return super()._classify_exec_error(command, result)

    def _runner_env(self) -> dict[str, str]:
        base_url = self._get_env(BASE_URL_ENV)
        api_key = self._get_env(API_KEY_ENV)
        if not base_url or not api_key:
            raise ValueError(f"{BASE_URL_ENV} and {API_KEY_ENV} must be set")
        if not self.model_name:
            raise ValueError("no model: pass --model to harbor")
        opts = self.options
        env: dict[str, str] = {
            "LLM_MODEL": litellm_model(self.model_name),
            "LLM_BASE_URL": base_url,
            "LLM_API_KEY": api_key,
            "LLM_TIMEOUT": str(opts.llm_timeout),
            "LLM_NUM_RETRIES": str(opts.num_retries),
            "MAX_ITERATIONS": str(opts.max_iterations),
            "MAX_CONTINUATIONS": str(opts.max_continuations),
            "AGENT_LOGS_DIR": self.environment_logs_dir.as_posix(),
            "OPENHANDS_SUPPRESS_BANNER": "1",
        }
        if opts.reasoning_effort:
            env["LLM_REASONING_EFFORT"] = opts.reasoning_effort
        if opts.max_output_tokens:
            env["LLM_MAX_OUTPUT_TOKENS"] = str(opts.max_output_tokens)
        if opts.temperature is not None:
            env["LLM_TEMPERATURE"] = str(opts.temperature)
        env["LLM_THINKING_DISPLAY"] = opts.thinking_display
        if self.mcp_servers:
            servers: list[dict[str, Any]] = []
            for s in self.mcp_servers:
                entry: dict[str, Any] = {"name": s.name, "transport": s.transport}
                if s.transport == "stdio":
                    if s.command:
                        entry["command"] = s.command
                    if s.args:
                        entry["args"] = s.args
                elif s.url:
                    entry["url"] = s.url
                servers.append(entry)
            env["MCP_SERVERS_JSON"] = json.dumps(servers)
        return env

    @with_prompt_template
    @override
    async def run(self, instruction: str, environment: BaseEnvironment,
                  context: AgentContext) -> None:
        env = self._runner_env()
        with tempfile.TemporaryDirectory(prefix="openhands-instruction-") as tmp:
            local = Path(tmp) / "instruction.md"
            local.write_text(instruction, encoding="utf-8")
            await environment.upload_file(source_path=local, target_path=INSTRUCTION_REMOTE)

        logs = self.environment_logs_dir.as_posix()
        command = (
            f"{RUNTIME_PYTHON} -u {RUNNER_REMOTE} "
            f"--instruction-file {INSTRUCTION_REMOTE} --logs-dir {shlex.quote(logs)} "
            f"2>&1 | tee {shlex.quote(logs)}/{RUNNER_LOG_FILENAME}"
        )
        await self.exec_as_agent(environment, command=command, env=env)


__all__ = ["OpenHandsAgent", "OpenHandsAgentOptions", "litellm_model"]
