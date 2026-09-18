#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from decimal import ROUND_DOWN, Decimal
from pathlib import Path


def _cut(value: float, dp: int = 2) -> float:
    """Truncate to `dp` places, mirroring harbor_to_output.norm_reward.

    Defined here rather than imported: this module is mounted into the task
    container at /harness/scoring, where tools/delivery does not exist. The
    same reason ctrf_pytest_plugin.py carries its own copy. All three must
    agree, or the container and the host publish different numbers for the
    same run.

    Decimal on repr(), not int(value * 100) / 100: the float nearest 0.29 is
    0.28999999999999998, which would truncate to 0.28.
    """
    return float(
        Decimal(repr(float(value))).quantize(Decimal(1).scaleb(-dp),
                                             rounding=ROUND_DOWN)
    )

# Optional Headroom prompt compression, default OFF (see grader_compress.py).
# This file runs inside the task container, where the scoring tree is mounted
# read-only at /harness/scoring rather than installed, so the module sits next
# to it on disk but not necessarily on sys.path. A missing or broken module
# must never stop a trial being graded, so the last fallback is an explicit
# identity function rather than an error.
try:
    from grader_compress import compress_evidence  # type: ignore
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        from grader_compress import compress_evidence  # type: ignore
    except ImportError:
        def compress_evidence(model, text):  # type: ignore[misc]
            return text

_out_path: Path | None = None
_token_out: Path | None = None


def _load_criteria(rubric_path: Path) -> list[dict]:
    """Load the bundle rubric, or refuse.

    An unrecognised shape used to return an empty list. That is the worst
    available outcome: zero criteria grade cleanly, _compute_scores divides a
    zero pool, and the run reports a finished rubric channel that judged
    nothing. Raising turns a silent wrong answer into a loud one, and matches
    rubric_judge._load_rubric, which refuses rather than returning empty.
    """
    raw = json.loads(rubric_path.read_text())
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        if isinstance(raw.get("criteria"), list):
            return raw["criteria"]
        if isinstance(raw.get("rubric"), list):
            return raw["rubric"]
    raise ValueError(
        f"{rubric_path} must be a list, {{'criteria': [...]}}, or {{'rubric': [...]}}, "
        f"got {type(raw).__name__}"
    )


# Hard char limit the codex-exec transport rejects prompts over.
_TRANSPORT_MAX_CHARS = int(os.environ.get("JUDGE_TRANSPORT_MAX_CHARS", "1048576"))
_SAFETY_MARGIN_CHARS = int(os.environ.get("JUDGE_SAFETY_MARGIN_CHARS", "8192"))

_EVIDENCE_BUDGET_CHARS = int(os.environ.get("JUDGE_EVIDENCE_BUDGET_CHARS", "0")) or None


def _evidence_budget(criteria: list[dict], final_message: str) -> int:
    """Evidence chars this call can afford: cap minus measured prompt overhead."""
    if _EVIDENCE_BUDGET_CHARS:
        return _EVIDENCE_BUDGET_CHARS
    fixed = len(_judge_prompt(criteria, "", "")) + len(final_message or "")
    return max(0, _TRANSPORT_MAX_CHARS - _SAFETY_MARGIN_CHARS - fixed)


def _render_trajectory(traj: dict, budget: int | None = None) -> str:
    """Render steps whole; when the budget binds, drop whole steps from the middle."""
    if budget is None:
        raise ValueError("_render_trajectory now requires an explicit budget; "
                         "use _evidence_budget(criteria, final_message)")
    steps = traj.get("steps", []) or []

    rendered: list[str] = []
    for step in steps:
        args = step.get("arguments", {})
        resp = step.get("response", "")
        args_s = json.dumps(args) if not isinstance(args, str) else args
        resp_s = json.dumps(resp) if not isinstance(resp, str) else str(resp)
        rendered.append(f"tool={step.get('tool')} args={args_s} → {resp_s}")

    # Final message is carried by _judge_prompt's own section, not duplicated here.
    tail = ""

    step_budget = max(0, budget - len(tail))
    total = sum(len(p) + 1 for p in rendered)

    if total > step_budget and rendered:
        head: list[str] = []
        tail_steps: list[str] = []
        used = 0
        i, j = 0, len(rendered) - 1
        take_head = True
        while i <= j:
            cand = rendered[i] if take_head else rendered[j]
            if used + len(cand) + 1 > step_budget:
                break
            used += len(cand) + 1
            if take_head:
                head.append(cand); i += 1
            else:
                tail_steps.append(cand); j -= 1
            take_head = not take_head
        dropped = len(rendered) - len(head) - len(tail_steps)
        rendered = head + (
            [f"...[{dropped} of {len(steps)} steps omitted: evidence budget "
             f"{budget:,} chars exceeded]"] if dropped else []
        ) + list(reversed(tail_steps))

    parts = rendered + ([tail] if tail else [])
    return "\n".join(parts)


# _OUTPUT_SCHEMA lived here. It was the Agent SDK's structured-output
# contract; the direct transport parses the reply with _parse_judge_json,
# which already handled bare, fenced, and prose-wrapped JSON.


# The models this judge grades with, spelled the way `codex exec -m` wants
# them. The transport (ported from devops-projects' judge council `codex/`
# seat) is the locally-installed Codex CLI run as a subprocess. It spends the
# operator's ChatGPT subscription quota rather than metered API credit, and
# needs no OPENAI_API_KEY, no bridge server, and no generated key — only a
# `codex login` this machine already holds.
CODEX_MODELS = ("gpt-5.6-sol",)

CODEX_EXEC_ARGS = ("exec", "--json", "--sandbox", "read-only", "--skip-git-repo-check")

# The models this judge grades with over the Claude Code CLI. This transport
# exists because the Codex one cannot run where the bundle judge actually runs.
# `tests/test.sh` executes inside the TASK CONTAINER, which is python:3.12-slim
# plus pytest and claude-agent-sdk -- there is no `codex` binary in it and no
# `codex login` credential, so every bundled trial logged "rubric judge ran but
# failed" and scored the rubric channel at nothing.
#
# Mounting the operator's ChatGPT credential into a container that just ran an
# untrusted agent is not an acceptable way to fix that. Claude needs no such
# thing here: task.toml's [verifier.env] already passes CLAUDE_CODE_OAUTH_TOKEN
# in, and because environment_mode = "shared" the verifier runs in the same
# container where Harbor bootstrapped the Claude Code CLI for the agent phase.
# The credential and the binary are both already present and already trusted.
CLAUDE_MODELS = ("claude-sonnet-4-5", "claude-opus-4-5", "claude-haiku-4-5")

CLAUDE_EXEC_ARGS = (
    "-p", "--output-format", "json", "--max-turns", "1",
    # A judge reads and returns a verdict. It gets no tools: nothing to grep
    # the trial tree with, nothing to edit, no way to act on what it grades.
    "--allowed-tools", "",
)

# Measured 2026-09-01 on codex-cli 0.146.0: a trivial one-line prompt still
# reported input_tokens=14564 (cached_input_tokens=11008). That is Codex's own
# agent scaffold, carried on EVERY call on top of the grading prompt. It is
# quota, not dollars, but it makes this judge's token counts non-comparable
# to an HTTP judge's.
CODEX_SCAFFOLD_TOKENS_OBSERVED = 14_564

# Retry policy for the subprocess transport. Transient failures here look
# like nonzero exits or an empty event stream rather than HTTP 429s, but the
# cause is usually the same hot rate-limit window right after an agent run.
_BACKOFF_BASE_SEC = 15.0
_BACKOFF_CAP_SEC = 120.0
_MAX_ATTEMPTS = 4


def _codex_cli() -> str:
    """Path to the Codex CLI, or "" when it is not installed."""
    return shutil.which("codex") or ""


def _claude_cli() -> str:
    """Path to the Claude Code CLI, or "" when it is not installed.

    PATH alone is not enough. Harbor bootstraps the CLI into ~/.local/bin and
    appends that directory to ~/.bashrc -- but ~/.bashrc is only sourced by an
    interactive shell, and tests/test.sh is not one. So the binary is present
    and invisible to `shutil.which` at exactly the moment the judge needs it.
    The explicit fallback is what makes this transport work in-container.
    """
    found = shutil.which("claude")
    if found:
        return found
    candidate = Path(os.path.expanduser("~/.local/bin/claude"))
    return str(candidate) if candidate.is_file() and os.access(str(candidate), os.X_OK) else ""


def _codex_credential_error() -> str:
    """Why the Codex CLI is unusable as a judge, or "" if it is ready.

    Asks the CLI itself (`codex login status`) rather than reading its
    credential store: the answer is what actually matters, and no secret need
    be touched to get it.
    """
    cli = _codex_cli()
    if not cli:
        return (
            "codex CLI not found on PATH; the rubric judge shells out to "
            "`codex exec`, so it must run where the CLI is installed and "
            "logged in (the host, not the task container)"
        )
    try:
        proc = subprocess.run(
            [cli, "login", "status"], capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError) as err:
        return f"could not run `codex login status`: {err}"
    out = f"{proc.stdout} {proc.stderr}".strip()
    # A plain substring test for "logged in" would also match "Not logged in",
    # so a nonzero exit or an explicit denial each fail on their own.
    if proc.returncode != 0 or "not logged in" in out.lower():
        return f"codex CLI is not logged in (run `codex login`): {out[:120]}"
    return ""


def _preflight(model: str) -> str | None:
    """Confirm the judge can reach its backend before grading starts.

    A backend discovered per-criterion arrives as a rubric full of failures
    that still writes a score file, which reads like a graded run.

    For a Codex model the whole check is the CLI's own `codex login status`.
    For a Claude model it is the binary plus a credential: Harbor injects
    CLAUDE_CODE_OAUTH_TOKEN through task.toml's [verifier.env], and without it
    the CLI exits 0 having written a prose auth error into `result` -- which
    would otherwise surface as an unparseable verdict four retries later
    instead of as the configuration fact it is.

    Any other model returns None here and is refused by _run_judge instead.
    """
    if model in CODEX_MODELS:
        return _codex_credential_error() or None
    if model in CLAUDE_MODELS:
        if not _claude_cli():
            return (
                "claude CLI not found on PATH or in ~/.local/bin; the rubric "
                "judge shells out to `claude -p`, so it must run where the CLI "
                "is installed"
            )
        if not (os.getenv("CLAUDE_CODE_OAUTH_TOKEN") or os.getenv("ANTHROPIC_API_KEY")):
            return (
                "neither CLAUDE_CODE_OAUTH_TOKEN nor ANTHROPIC_API_KEY is set; "
                "check task.toml's [verifier.env] passes the token into the "
                "verifier container"
            )
        return None
    return None


def _judge_prompt(criteria: list[dict], traj_ctx: str, final_ctx: str) -> str:
    """The grading prompt. Shared, so the two transports grade identically."""
    criteria_payload = json.dumps(
        [
            {
                "number": c.get("number"),
                "criterion": c.get("criterion"),
                "evaluation_target": c.get("evaluation_target", "trajectory"),
                "is_positive": c.get("is_positive", True),
            }
            for c in criteria
        ],
        indent=2,
    )

    prompt = (
        "You are grading an AI agent's trial output against a rubric.\n\n"
        "## Trajectory\n"
        + traj_ctx
        + "\n\n## Final Message\n"
        + (final_ctx or "(none)")
        + "\n\n## Criteria\n"
        + criteria_payload
        + "\n\nFor each criterion, set satisfied=true when the criterion's statement "
        "is TRUE of the agent's behavior, and false when it is not.\n"
        "Read is_positive carefully. A criterion with is_positive=false states a "
        "mistake, so satisfied=true there means the agent MADE that mistake and "
        "satisfied=false means it avoided it. Judge the statement itself, never "
        "whether the agent did well: on a is_positive=false criterion an agent that "
        "behaved correctly gets satisfied=false.\n"
        "Use trajectory for evaluation_target=trajectory or trajectory_and_state.\n"
        "Use final message for evaluation_target=final_answer.\n"
        "Return JSON: {results: [{number, satisfied (bool), justification (one sentence)}]}"
    )
    return prompt


def _codex_exec(model: str, prompt: str, timeout: float) -> tuple[str, dict]:
    """One `codex exec` call: prompt in on stdin, (reply_text, usage) out.

    The single implementation of the subprocess plumbing — `_run_judge_codex`
    here and `rubric_judge._call_judge_once` both grade through it, so the
    argv, the sandbox pinning, and the JSONL event parsing cannot drift apart.
    Raises JudgeResponseError on a nonzero exit or an empty event stream;
    retry policy belongs to the caller.
    """
    cli = _codex_cli()
    if not cli:
        raise JudgeResponseError("codex CLI not found on PATH")
    proc = subprocess.run(
        [cli, *CODEX_EXEC_ARGS, "-m", model],
        input=prompt, capture_output=True, text=True, timeout=timeout,
        # A scratch cwd, so the read-only agent cannot even see the trial
        # tree it is judging or mistake it for its own workspace.
        cwd=tempfile.gettempdir(),
    )
    if proc.returncode != 0:
        raise JudgeResponseError(
            f"codex exec exited {proc.returncode}: "
            f"{(proc.stderr or proc.stdout).strip()[:300]}"
        )
    message, usage_obj = "", {}
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        item = event.get("item") or {}
        if item.get("type") == "agent_message" and isinstance(item.get("text"), str):
            # Keep the LAST message: the agent may narrate before it
            # delivers the verdict JSON.
            message = item["text"]
        if event.get("type") == "turn.completed":
            usage_obj = event.get("usage") or {}
    if not message.strip():
        raise JudgeResponseError("codex exec produced no agent message")
    return message, usage_obj


def _claude_exec(model: str, prompt: str, timeout: float) -> tuple[str, dict]:
    """One `claude -p` call: prompt in on stdin, (reply_text, usage) out.

    Mirrors _codex_exec's contract exactly -- same return shape, same
    JudgeResponseError on transport failure, retry policy left to the caller --
    so _run_judge_claude and _run_judge_codex stay structurally identical and
    cannot drift.

    The CLI answers with a single JSON envelope: {"type": "result", "subtype":
    "success", "is_error": false, "result": "<reply text>", "usage": {...},
    "total_cost_usd": N}. The reply commonly arrives fenced in ```json, which
    _parse_judge_json already unwraps.
    """
    cli = _claude_cli()
    if not cli:
        raise JudgeResponseError("claude CLI not found on PATH or in ~/.local/bin")
    proc = subprocess.run(
        [cli, *CLAUDE_EXEC_ARGS, "--model", model],
        input=prompt, capture_output=True, text=True, timeout=timeout,
        # Same reasoning as the Codex path: a scratch cwd so the judge cannot
        # see the trial tree it is grading or mistake it for its own workspace.
        cwd=tempfile.gettempdir(),
    )
    if proc.returncode != 0:
        raise JudgeResponseError(
            f"claude exited {proc.returncode}: "
            f"{(proc.stderr or proc.stdout).strip()[:300]}"
        )
    try:
        envelope = json.loads(proc.stdout)
    except json.JSONDecodeError as err:
        raise JudgeResponseError(
            f"claude produced no JSON envelope: {proc.stdout.strip()[:300]}"
        ) from err
    # is_error is the CLI's own signal that the turn failed (auth, rate limit,
    # refusal). It still exits 0 in that case, so checking the exit code alone
    # would hand a prose apology to _parse_judge_json and fail one layer later
    # with a much less useful message.
    if envelope.get("is_error"):
        raise JudgeResponseError(
            f"claude reported is_error: {str(envelope.get('result'))[:300]}"
        )
    message = envelope.get("result") or ""
    if not str(message).strip():
        raise JudgeResponseError("claude produced no result text")
    return str(message), (envelope.get("usage") or {}) | {
        "total_cost_usd": envelope.get("total_cost_usd", 0.0)
    }


def _run_judge_claude(
    criteria: list[dict], traj_ctx: str, final_ctx: str, model: str
) -> list[dict]:
    """Grade by running the local Claude Code CLI. Same shape as the Codex path."""
    if not _claude_cli():
        raise JudgeResponseError("claude CLI not found on PATH or in ~/.local/bin")
    prompt = _judge_prompt(criteria, compress_evidence(model, traj_ctx), final_ctx)
    timeout = float(os.environ.get("JUDGE_CLAUDE_TIMEOUT", os.environ.get("JUDGE_CODEX_TIMEOUT", "600")))

    last: Exception | None = None
    text, usage = "", {}
    for n in range(1, _MAX_ATTEMPTS + 1):
        try:
            text, usage = _claude_exec(model, prompt, timeout)
            break
        except (JudgeResponseError, OSError, subprocess.SubprocessError) as err:
            last = err
            # Over-cap prompts fail deterministically; don't waste retries.
            if "input_too_large" in str(err) or "maximum length" in str(err):
                raise JudgeResponseError(
                    f"prompt over transport cap ({_TRANSPORT_MAX_CHARS:,} chars); "
                    f"not retrying: {err}") from err
            if n >= _MAX_ATTEMPTS:
                raise JudgeResponseError(f"claude judge failed: {err}") from err
            delay = min(_BACKOFF_BASE_SEC * 2 ** (n - 1), _BACKOFF_CAP_SEC)
            print(
                f"[judge] claude/{model}: {str(err)[:120]} "
                f"(attempt {n}/{_MAX_ATTEMPTS}), retrying in {delay:.0f}s",
                file=sys.stderr,
            )
            time.sleep(delay)
    else:  # pragma: no cover - the loop always breaks or raises
        raise JudgeResponseError(f"claude judge failed: {last}")

    _record_usage_claude(usage, model)

    parsed = _parse_judge_json(text) if text else None
    if isinstance(parsed, dict) and "results" in parsed:
        return parsed["results"]
    if isinstance(parsed, list):
        return parsed
    raise JudgeResponseError(
        "claude judge returned no results array; reply began: "
        + (text[:400] if text else "(empty)")
    )


def _record_usage_claude(usage: dict, model: str) -> None:
    """Usage line for the Claude transport. Best-effort; never raises.

    Anthropic reports input_tokens EXCLUSIVE of the cached counts, unlike
    Codex's inclusive convention -- so there is deliberately no subtraction
    here. Getting that backwards would double-count the cache read.
    """
    if _token_out is None:
        return
    try:
        name = model or os.getenv("JUDGE_MODEL", "")
        fresh = int(usage.get("input_tokens", 0) or 0)
        output = int(usage.get("output_tokens", 0) or 0)
        cache_read = int(usage.get("cache_read_input_tokens", 0) or 0)
        cache_write = int(usage.get("cache_creation_input_tokens", 0) or 0)
        # `claude -p` reports a real total_cost_usd on an API key and 0 on a
        # subscription. Trust the reported figure whenever there is one, and
        # only value the call at list price when the CLI priced it at nothing.
        reported = float(usage.get("total_cost_usd", 0.0) or 0.0)
        line = {
            "model_name": name,
            "judge_input_tokens": fresh,
            "judge_output_tokens": output,
            "judge_input_cache_tokens": cache_read,
            "judge_output_cache_tokens": cache_write,
            "judge_cost_usd": reported or _judge_cost(name, fresh, cache_read, cache_write, output),
        }
        _emit_usage(line)
    except Exception:
        pass


def _run_judge_codex(
    criteria: list[dict], traj_ctx: str, final_ctx: str, model: str
) -> list[dict]:
    """Grade by running the local Codex CLI. No server, no key, no bridge.

    Ported from devops-projects' judge council `_call_judge_codex`. It drives
    the CLI the way it is meant to be driven (`codex exec`) rather than
    extracting the stored credential and replaying it against api.openai.com:
    that credential is scoped to Codex traffic, and impersonating the Codex
    client to get around that is neither reliable nor ours to do.

    Two consequences worth knowing at the call site:

      * Codex is an AGENT, not a completion endpoint. It is pinned to a
        read-only sandbox and run in a scratch cwd so the judge cannot mutate
        the trial it is grading, but it may still take a turn or two to
        answer, so this is slower than an HTTP call.
      * Every call carries Codex's own ~14.5K-token scaffold
        (CODEX_SCAFFOLD_TOKENS_OBSERVED) on top of the grading prompt.
    """
    # A missing CLI is a configuration fact, not a transient failure: refuse
    # before the retry loop rather than backing off three times into it.
    if not _codex_cli():
        raise JudgeResponseError("codex CLI not found on PATH")
    # Only the trajectory goes through compression. The criteria and the
    # surrounding instructions are the verdict contract, and final_ctx is the
    # artifact being judged -- see grader_compress's module docstring for why
    # neither is a legitimate target.
    prompt = _judge_prompt(criteria, compress_evidence(model, traj_ctx), final_ctx)
    timeout = float(os.environ.get("JUDGE_CODEX_TIMEOUT", "600"))

    last: Exception | None = None
    text, usage = "", {}
    for n in range(1, _MAX_ATTEMPTS + 1):
        try:
            text, usage = _codex_exec(model, prompt, timeout)
            break
        except (JudgeResponseError, OSError, subprocess.SubprocessError) as err:
            last = err
            # Over-cap prompts fail deterministically; don't waste retries.
            if "input_too_large" in str(err) or "maximum length" in str(err):
                raise JudgeResponseError(
                    f"prompt over transport cap ({_TRANSPORT_MAX_CHARS:,} chars); "
                    f"not retrying: {err}") from err
            if n >= _MAX_ATTEMPTS:
                raise JudgeResponseError(f"codex judge failed: {err}") from err
            delay = min(_BACKOFF_BASE_SEC * 2 ** (n - 1), _BACKOFF_CAP_SEC)
            print(
                f"[judge] codex/{model}: {str(err)[:120]} "
                f"(attempt {n}/{_MAX_ATTEMPTS}), retrying in {delay:.0f}s",
                file=sys.stderr,
            )
            time.sleep(delay)
    else:  # pragma: no cover - the loop always breaks or raises
        raise JudgeResponseError(f"codex judge failed: {last}")

    _record_usage_codex(usage, model)

    parsed = _parse_judge_json(text) if text else None
    if isinstance(parsed, dict) and "results" in parsed:
        return parsed.get("results", [])
    raise JudgeResponseError(
        "judge returned no parseable JSON; reply began: "
        + (text[:400] if text else "(empty)")
    )


_usage_lines: list[dict] = []


def _emit_usage(line: dict) -> None:
    """Append one judge line and rewrite the file with every line so far.

    Each recorder used to write_text([line]), so when a grading made more than
    one judge call only the last one survived and the rest of the tokens went
    unbilled. Accumulating in memory (rather than reading the file back) keeps
    a re-driven run dir from stacking a previous process's lines on top.
    """
    if _token_out is None:
        return
    _usage_lines.append(line)
    _token_out.parent.mkdir(parents=True, exist_ok=True)
    _token_out.write_text(json.dumps(_usage_lines, indent=2))


def _price(var: str) -> float:
    """A per-million-token rate from the environment; 0.0 when unset or junk."""
    try:
        return float(os.getenv(var, "") or 0.0)
    except ValueError:
        return 0.0


# USD per million tokens, from litellm's model_prices_and_context_window --
# the same table Harbor prices the trajectory side with, so a judge line and a
# trajectory line in Finance are denominated the same way. Rates are list
# price; see _judge_cost for what that does and does not mean.
_JUDGE_RATES: dict[str, dict[str, float]] = {
    "gpt-5.6-sol":       {"input": 5.0, "output": 30.0, "cache_read": 0.5,  "cache_write": 6.25},
    "claude-opus-4-5":   {"input": 5.0, "output": 25.0, "cache_read": 0.5,  "cache_write": 6.25},
    "claude-sonnet-4-5": {"input": 3.0, "output": 15.0, "cache_read": 0.3,  "cache_write": 3.75},
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0, "cache_read": 0.3,  "cache_write": 3.75},
    "claude-haiku-4-5":  {"input": 1.0, "output": 5.0,  "cache_read": 0.1,  "cache_write": 1.25},
}

_RATE_ENV = {
    "input":       "JUDGE_PRICE_INPUT_PER_MTOK",
    "output":      "JUDGE_PRICE_OUTPUT_PER_MTOK",
    "cache_read":  "JUDGE_PRICE_CACHED_INPUT_PER_MTOK",
    "cache_write": "JUDGE_PRICE_CACHE_WRITE_PER_MTOK",
}


def _rates_for(model: str) -> dict[str, float]:
    """Per-Mtok rates for one judge model: table first, env overrides on top.

    Keyed by model rather than taken as four flat numbers because the judge
    runs on two transports -- gpt-5.6-sol on the host, a Claude model in the
    task container -- and one global rate set would misprice whichever one is
    not in use. An unknown model prices at zero rather than borrowing another
    model's rate.
    """
    base = dict(_JUDGE_RATES.get(model, {}))
    for key, var in _RATE_ENV.items():
        if os.getenv(var, "").strip():
            base[key] = _price(var)
    return base


def _judge_cost(model: str, fresh: int, cached: int, cache_write: int, output: int) -> float:
    """Imputed list-price cost of one judge call.

    This is a VALUATION, not an invoice. `codex exec` and `claude -p` spend the
    operator's subscription quota, so no card is charged for these tokens; the
    number here is what the same work would have cost at metered list price, so
    judge effort is visible in Finance instead of silently reading $0.00.
    Do not add it to a provider bill -- it would double count.

    Set any JUDGE_PRICE_*_PER_MTOK to override a rate, or set them all to 0 to
    go back to reporting nothing.
    """
    r = _rates_for(model)
    if not r:
        return 0.0
    return round(
        fresh       / 1e6 * r.get("input", 0.0)
        + output      / 1e6 * r.get("output", 0.0)
        + cached      / 1e6 * r.get("cache_read", 0.0)
        + cache_write / 1e6 * r.get("cache_write", 0.0),
        6,
    )


def _record_usage_codex(usage: dict, model: str) -> None:
    """Same file and field names as the old bridge path, from `codex exec` usage.

    Codex reports OpenAI-style INCLUSIVE input counts: both cached_input_tokens
    and cache_write_input_tokens are subsets of input_tokens. Both are
    subtracted so the buckets stay disjoint and sum back to input_tokens --
    the same mapping finance_reporter.py applies to the trajectory side.

    `reasoning_output_tokens` is deliberately not added to judge_output_tokens:
    it is a breakdown of output_tokens, not a sibling of it, so folding it in
    would double count.

    Written best-effort so tools/finance/finance_reporter.py always sees the usage
    that was incurred: never raises, never blocks grading.
    """
    if _token_out is None:
        return
    try:
        total_in = int(usage.get("input_tokens", 0) or 0)
        cached = int(usage.get("cached_input_tokens", 0) or 0)
        # Reported by codex as cache_write_input_tokens; this was hardcoded to
        # 0 before, which silently dropped every cache write the judge paid for.
        cache_write = int(usage.get("cache_write_input_tokens", 0) or 0)
        output = int(usage.get("output_tokens", 0) or 0)
        fresh = max(0, total_in - cached - cache_write)
        line = {
            "model_name": model or os.getenv("JUDGE_MODEL", ""),
            "judge_input_tokens": fresh,
            "judge_output_tokens": output,
            "judge_input_cache_tokens": cached,
            # Input-side cache WRITES. Named for the field the Finance API
            # accepts; finance_reporter.py mirrors it to judge_cache_write_tokens.
            "judge_output_cache_tokens": cache_write,
            "judge_cost_usd": _judge_cost(
                model or os.getenv("JUDGE_MODEL", ""), fresh, cached, cache_write, output),
        }
        _emit_usage(line)
    except Exception:
        pass


def _default_judge_model() -> str:
    """Pick a judge model the machine can actually reach.

    An explicit JUDGE_MODEL always wins -- a pinned grader is a benchmark
    property, and this must never silently override one.

    Otherwise the default follows the transport that exists here. A fixed
    default of gpt-5.6-sol is right on the host and unrunnable inside the task
    container, and every bundle's tests/test.sh invokes this CLI with no
    --model. Resolving at runtime is what lets the container grade its rubric
    channel without each bundle having to know which CLI it was built with.
    """
    pinned = os.getenv("JUDGE_MODEL", "").strip()
    if pinned:
        return pinned

    # Nothing pinned: the model now depends on which CLI this machine happens
    # to have, so the same rubric can be graded by different judges across
    # trials of one job. That is a benchmark property changing underfoot, and
    # it has already happened -- one recorded run was graded by
    # claude-sonnet-4-5 while the other 42 used gpt-5.6-sol, with nothing in
    # the logs to say so. The fallback stays, because the container genuinely
    # has no codex binary and grading nothing would be worse; but it announces
    # itself so the divergence is visible at the point it is decided.
    chosen = CODEX_MODELS[0] if _codex_cli() else (
        CLAUDE_MODELS[0] if _claude_cli() else CODEX_MODELS[0])
    print(f"[rubric-judge] WARNING: JUDGE_MODEL is not set; falling back to "
          f"'{chosen}' based on which CLI is installed here. Pin JUDGE_MODEL "
          f"to keep the grader constant across trials.", file=sys.stderr)
    return chosen


async def _run_judge(
    criteria: list[dict], traj_ctx: str, final_ctx: str, model: str
) -> list[dict]:
    """Grade one trial over whichever local CLI can actually run here.

    Two transports, chosen by the model id, because the judge runs in two very
    different places and only one CLI exists in each:

      * On the HOST (scripts, re-grades) `codex` is installed and logged in,
        and gpt-5.6-sol grades through it.
      * In the TASK CONTAINER, where tests/test.sh runs, there is no codex
        binary and no codex credential -- but Harbor has already bootstrapped
        the Claude Code CLI there for the agent, and task.toml's
        [verifier.env] passes CLAUDE_CODE_OAUTH_TOKEN in.

    Before this, the container case had no working transport at all: every
    bundled trial logged "rubric judge ran but failed" and scored the rubric
    channel at nothing, while still producing a plausible-looking reward from
    the other channels. A grader that silently drops a weighted channel is
    worse than one that refuses, so an unusable model now names both lists and
    says which CLI is missing.
    """
    if model in CODEX_MODELS:
        return _run_judge_codex(criteria, traj_ctx, final_ctx, model)
    if model in CLAUDE_MODELS:
        return _run_judge_claude(criteria, traj_ctx, final_ctx, model)
    raise JudgeResponseError(
        f"{model!r} is not a model this judge grades with. Pass --model, or set "
        f"JUDGE_MODEL, to one of {list(CODEX_MODELS)} (local `codex` CLI, "
        f"available={bool(_codex_cli())}) or {list(CLAUDE_MODELS)} "
        f"(local `claude` CLI, available={bool(_claude_cli())})."
    )


class JudgeResponseError(RuntimeError):
    """The judge ran but its answer could not be read as rubric JSON."""


def _parse_judge_json(text: str):
    """Best-effort JSON out of a model reply: bare, fenced, or embedded in prose."""
    text = text.strip()
    if not text:
        return None
    if text.startswith("```"):
        lines = text.split("\n")[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except ValueError:
        pass
    start, end = text.find("{"), text.rfind("}")      # JSON wrapped in prose
    if 0 <= start < end:
        try:
            return json.loads(text[start:end + 1])
        except ValueError:
            pass
    return None


def _compute_scores(criteria: list[dict], results: list[dict]) -> dict:
    by_num = {str(r.get("number", "")): r for r in results}
    pos_total = pos_earned = neg_total = neg_hit = 0.0
    per_criterion: list[dict] = []

    for c in criteria:
        num = str(c.get("number", ""))
        score = float(c.get("score", 1))
        is_pos = bool(c.get("is_positive", True))
        r = by_num.get(num, {})
        satisfied = bool(r.get("satisfied", False))

        if is_pos:
            pos_total += score
            if satisfied:
                pos_earned += score
        else:
            neg_total += score
            if satisfied:
                neg_hit += score

        per_criterion.append(
            {
                "number": num,
                "criterion": c.get("criterion"),
                "type": c.get("type"),
                "evaluation_target": c.get("evaluation_target"),
                "importance": c.get("importance"),
                "score": score,
                "is_positive": is_pos,
                "satisfied": satisfied,
                "justification": r.get("justification", ""),
            }
        )

    rc = (pos_earned / pos_total) if pos_total else 1.0
    rb = (neg_hit / neg_total) if neg_total else 0.0
    return {
        # Cut, never rounded, and cut only on the way out: the score is taken
        # from the full-precision rc and rb, so truncating the two inputs does
        # not compound into the number the ledger reads. These three published
        # at four places was the last thing in the tree still disagreeing with
        # the two-place rule -- rubric_breakdown.json carried 0.5716 beside a
        # reward.json reading 0.57, which is exactly the "which precision is
        # the real one" problem the rule exists to end.
        "score": _cut(rc * (1.0 - rb)),
        "rc": _cut(rc),
        "rb": _cut(rb),
        # Pass/fail is decided on the FULL values, not the cut ones. A run that
        # earned 0.999 must not be promoted to a pass by truncation.
        "rubric_passed": rc >= 0.999999 and rb <= 0.000001,
        "per_criterion": per_criterion,
    }


def main() -> None:
    global _out_path, _token_out

    ap = argparse.ArgumentParser()
    ap.add_argument("--rubric", required=True)
    ap.add_argument("--trajectory", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--token-output", default=None,
                    help="write the judge's own token usage here, in the finance "
                         "API's judge_lines shape (read by tools/finance/finance_reporter.py)")
    ap.add_argument("--model", default=None,
                    help="judge model; defaults to JUDGE_MODEL, else whichever "
                         "of the codex / claude CLIs is installed here")
    a = ap.parse_args()
    if not a.model:
        a.model = _default_judge_model()

    _out_path = Path(a.output)
    _token_out = Path(a.token_output) if a.token_output else None
    rubric_path = Path(a.rubric)
    traj_path = Path(a.trajectory)

    if not rubric_path.exists():
        print(f"rubric not found: {rubric_path}", file=sys.stderr)
        sys.exit(1)
    if not traj_path.exists():
        print(f"trajectory not found: {traj_path}", file=sys.stderr)
        sys.exit(1)

    # Route to the judge backend before any grading starts. A misconfigured
    # backend that is discovered per-criterion arrives as a rubric full of
    # failures that still writes a score file, which reads like a graded run.
    routing_error = _preflight(a.model)
    if routing_error:
        print(routing_error, file=sys.stderr)
        _out_path.parent.mkdir(parents=True, exist_ok=True)
        _out_path.write_text(
            json.dumps(
                {"score": 0.0, "rc": 0.0, "rb": 0.0, "rubric_passed": False, "per_criterion": []},
                indent=2,
            )
        )
        sys.exit(0)

    criteria = _load_criteria(rubric_path)
    if not criteria:
        print("no LLM-graded criteria found in rubric", file=sys.stderr)
        _out_path.parent.mkdir(parents=True, exist_ok=True)
        _out_path.write_text(
            json.dumps(
                {"score": 0.0, "rc": 0.0, "rb": 0.0, "rubric_passed": False, "per_criterion": []},
                indent=2,
            )
        )
        sys.exit(0)

    traj = json.loads(traj_path.read_text())
    final_ctx = traj.get("final_message", "")
    _budget = _evidence_budget(criteria, final_ctx)
    traj_ctx = _render_trajectory(traj, budget=_budget)

    # Say how much evidence actually went in. Without this the only way to know
    # the judge was reading a fraction of the trajectory was to diff its input
    # token count against the log on disk -- which is how the 10k cap survived
    # as long as it did. ~1.8 chars/token is the worst-case (JSON-dense) ratio;
    # prose runs nearer 3.9, so this over-estimates rather than surprising us.
    _n_steps = len(traj.get("steps") or [])
    _omitted = "...[" in traj_ctx and "steps omitted" in traj_ctx
    print(f"[judge] evidence {len(traj_ctx):,} chars over {_n_steps} steps, "
          f"budget {_budget:,} (derived; transport cap {_TRANSPORT_MAX_CHARS:,})"
          + ("  ** BUDGET BOUND: steps omitted **" if _omitted else ""))

    print(f"Grading {len(criteria)} criteria with {a.model}")
    results = asyncio.run(_run_judge(criteria, traj_ctx, final_ctx, a.model))

    # A judge that answered but whose reply would not parse must not leave a
    # breakdown of silent falses behind. wraysbury run 5 billed 2,937 output
    # tokens, wrote no verdicts, and published reward 0 with the best Channel A
    # of its eight trials -- the only trace was the token bill. Fail loudly and
    # write nothing, so the reader records the trial as UNSCORED rather than as
    # an agent that satisfied nothing.
    if not results:
        marker = _out_path.parent / "rubric_judge_failed.txt"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(
            f"judge produced no parseable verdicts\n"
            f"model: {a.model}\n"
            f"criteria: {len(criteria)}\n"
            f"evidence chars: {len(traj_ctx)}\n"
            f"steps: {_n_steps}\n"
            f"budget bound: {_omitted}\n"
        )
        print(f"ERROR: judge returned no parseable verdicts for {len(criteria)} "
              f"criteria; rubric channel is UNSCORED. See {marker}", file=sys.stderr)
        raise SystemExit(1)

    doc = _compute_scores(criteria, results)
    doc["evidence"] = {
        "budget_chars": _budget,
        "rendered_chars": len(traj_ctx),
        "steps_total": _n_steps,
        "steps_rendered": traj_ctx.count("tool="),
        "budget_bound": _omitted,
        "transport_max_chars": _TRANSPORT_MAX_CHARS,
    }

    _out_path.parent.mkdir(parents=True, exist_ok=True)
    _out_path.write_text(json.dumps(doc, indent=2))
    print(f"score={doc['score']} rc={doc['rc']} rb={doc['rb']} passed={doc['rubric_passed']}")
    print(f"Written: {_out_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        import traceback

        traceback.print_exc()
        out = _out_path or Path("/logs/verifier/rubric_breakdown.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {"score": 0.0, "rc": 0.0, "rb": 0.0, "rubric_passed": False, "per_criterion": [], "error": str(exc)},
                indent=2,
            )
        )
        sys.exit(1)
