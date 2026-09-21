"""The OpenHands agent: what it writes, how harbor sees it, how run_task.sh wires it.

Three layers, each checked where it can fail on its own:

  1. STREAM -- tools/openhands_agent/runner.py translates SDK events into
     Claude Code's stream-json dialect, because every grader here reads that
     stream and nothing else. Driven with stand-in events (the translator
     matches on class NAME, so no SDK is needed) and read back with the
     reshaper's own parse_stream, the same function that scores real runs.
  2. AGENT -- tools/openhands_agent/agent.py is imported by harbor's own
     interpreter, since that is the only place it ever runs.
  3. DISPATCH -- scripts/run_task.sh against a stub harbor and a fake ccbridge:
     the right --agent, the overlays, the bridge URL and secret in harbor's
     environment, and none of the claude-code-only flags.

Nothing here spends a model call or starts a task container.
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from conftest import docker_is_usable, mirror_harbor_package

REPO = Path(__file__).resolve().parents[2]
AGENT_DIR = REPO / "tools" / "openhands_agent"
RUNNER = AGENT_DIR / "runner.py"
RUN_TASK = REPO / "scripts" / "run_task.sh"
RESHAPER = REPO / "tools" / "delivery_utils" / "harbor_to_output.py"
PROXY_DIR = REPO / "tools" / "network" / "egress-proxy"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


runner = _load("openhands_runner_under_test", RUNNER)


# =============================================================================
# 1. STREAM
# =============================================================================

def _event(kind: str, **fields):
    """A stand-in SDK event: the translator dispatches on the class name."""
    return type(kind, (), {})() if not fields else _with(type(kind, (), {})(), fields)


def _with(obj, fields):
    for k, v in fields.items():
        setattr(obj, k, v)
    return obj


def _text(t: str):
    return SimpleNamespace(text=t)


def _call(call_id: str, name: str, args: dict, *, response_id: str, thought: str = "",
          action=None):
    return _event(
        "ActionEvent", llm_response_id=response_id, tool_call_id=call_id, tool_name=name,
        tool_call=SimpleNamespace(arguments=json.dumps(args)),
        thought=[_text(thought)] if thought else [], thinking_blocks=[],
        reasoning_content=None, action=action, timestamp="2026-09-22T00:00:00+00:00",
    )


def _obs(call_id: str, name: str, texts: list[str], *, is_error: bool = False):
    return _event("ObservationEvent", tool_call_id=call_id, tool_name=name,
                  observation=SimpleNamespace(content=[_text(t) for t in texts],
                                              is_error=is_error))


@pytest.fixture()
def recorded(tmp_path):
    """A short conversation: two parallel MCP calls, a shell call, a finish."""
    stream = runner.StreamWriter(tmp_path / "openhands.txt", model="anthropic/claude-opus-5",
                                 session_id="s1")
    rec = runner.Recorder(stream, model="anthropic/claude-opus-5")
    mcp_action = SimpleNamespace(data={"board_id": "b-1"})  # what reached the server
    for ev in [
        _event("SystemPromptEvent", system_prompt=_text("You are OpenHands."), timestamp=None),
        _event("MessageEvent", source="user", llm_message=SimpleNamespace(
            content=[_text("Reconcile the ledger.")], thinking_blocks=[], reasoning_content=None),
            timestamp=None, llm_response_id=None),
        _call("c1", "mcp__LightMonday__get_board", {"board_id": "b-1", "security_risk": "LOW",
                                                     "summary": "read the board"},
              response_id="r1", thought="Reading both apps.", action=mcp_action),
        _call("c2", "mcp__LightJira__create_issue",
              {"summary": "RC-004 declined", "security_risk": "LOW"},
              response_id="r1", action=SimpleNamespace(data={"summary": "RC-004 declined"})),
        _obs("c1", "mcp__LightMonday__get_board",
             ["[Tool 'get_board' executed.]", '{"status": "ok", "items": 24}']),
        _obs("c2", "mcp__LightJira__create_issue", ['{"status": "ok", "key": "BGBW-9"}']),
        _call("c3", "terminal", {"command": "ls /workspace/data", "security_risk": "LOW",
                                 "summary": "list"}, response_id="r2", action=None),
        _obs("c3", "terminal", ["a.pdf\nb.png"]),
        _call("c4", "finish", {"message": "Done: 19 settled, 5 declined."}, response_id="r3"),
        _obs("c4", "finish", ["Done: 19 settled, 5 declined."]),
    ]:
        rec(ev)
    stream.result(subtype="success", is_error=False, result=rec.final_text, num_turns=rec.actions,
                  usage=runner.claude_usage(1000, 50, 600, 300, 10), total_cost_usd=0.01)
    stream.close()
    return rec, tmp_path / "openhands.txt"


def _lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_mcp_tool_use_keeps_claude_code_names(recorded):
    _, path = recorded
    uses = [b for ev in _lines(path) if ev["type"] == "assistant"
            for b in ev["message"]["content"] if b["type"] == "tool_use"]
    assert [u["name"] for u in uses] == [
        "mcp__LightMonday__get_board", "mcp__LightJira__create_issue", "terminal", "finish"]


def test_sdk_meta_fields_are_split_off_but_a_real_summary_stays(recorded):
    rec, path = recorded
    uses = {b["id"]: b["input"] for ev in _lines(path) if ev["type"] == "assistant"
            for b in ev["message"]["content"] if b["type"] == "tool_use"}
    assert uses["c1"] == {"board_id": "b-1"}
    # Jira's create_issue declares its own `summary`; it is the call, not bookkeeping.
    assert uses["c2"] == {"summary": "RC-004 declined"}
    assert uses["c3"] == {"command": "ls /workspace/data"}
    calls = [c for s in rec.steps for c in s.get("tool_calls", [])]
    assert calls[0]["extra"] == {"security_risk": "LOW", "summary": "read the board"}


def test_mcp_result_is_the_servers_raw_payload(recorded):
    _, path = recorded
    results = {b["tool_use_id"]: b["content"][0]["text"] for ev in _lines(path)
               if ev["type"] == "user" and isinstance(ev["message"]["content"], list)
               for b in ev["message"]["content"]}
    assert json.loads(results["c1"]) == {"status": "ok", "items": 24}
    assert json.loads(results["c2"])["key"] == "BGBW-9"


def test_thought_is_emitted_once_per_llm_response(recorded):
    _, path = recorded
    texts = [b["text"] for ev in _lines(path) if ev["type"] == "assistant"
             for b in ev["message"]["content"] if b["type"] == "text"]
    assert texts.count("Reading both apps.") == 1


def test_reshaper_scores_the_stream_like_a_claude_code_one(recorded):
    """parse_stream is what turns a trial into trace, tool counts and a failure class."""
    rec, path = recorded
    reshaper = _load("harbor_to_output_under_test", RESHAPER)
    parsed = reshaper.parse_stream(path)
    assert parsed["instruction"] == "Reconcile the ledger."
    assert [t["tool"] for t in parsed["trace"]] == [
        "get_board", "create_issue", "terminal", "finish"]
    assert parsed["valid"] == 2          # the two MCP calls
    assert parsed["invalid"] == 0        # terminal/finish are built-ins, not mistakes
    assert parsed["final_answer"] == "Done: 19 settled, 5 declined."
    assert parsed["termination_reason"] == "success"
    # Anthropic semantics: input excludes the cache, which the reshaper adds back.
    assert parsed["usage"]["input_tokens"] == 100
    assert parsed["usage"]["cache_read_tokens"] == 600


def test_a_run_that_lost_the_model_classifies_as_infrastructure(tmp_path):
    reshaper = _load("harbor_to_output_under_test", RESHAPER)
    stream = runner.StreamWriter(tmp_path / "openhands.txt", model="m", session_id="s")
    stream.result(subtype="error_during_execution", is_error=True,
                  result="API Error: ConversationRunError: Connection refused")
    stream.close()
    parsed = reshaper.parse_stream(tmp_path / "openhands.txt")
    assert parsed["transport_error"]
    cls, _ = reshaper.classify_failure(False, [], [], parsed, None)
    assert cls == "infrastructure"


def test_max_iterations_is_recorded_as_claude_codes_max_turns(tmp_path):
    stream = runner.StreamWriter(tmp_path / "s.txt", model="m", session_id="s")
    rec = runner.Recorder(stream, model="m")
    rec(_event("ConversationErrorEvent", code="MaxIterationsReached", detail="limit 3"))
    stream.close()
    assert rec.conversation_error == ("MaxIterationsReached", "limit 3")
    # Human-readable only: both parsers skip system lines.
    assert _lines(tmp_path / "s.txt")[0]["subtype"] == "error"


def test_claude_usage_never_goes_negative():
    assert runner.claude_usage(10, 1, 20, 5, 0)["input_tokens"] == 0


def test_mcp_config_from_harbors_server_list():
    raw = json.dumps([
        {"name": "LightXero", "transport": "streamable-http", "url": "http://light-servers:9139/mcp"},
        {"name": "local", "transport": "stdio", "command": "srv", "args": ["-q"]},
    ])
    assert runner.mcp_config_from_env(raw) == {
        "LightXero": {"url": "http://light-servers:9139/mcp", "transport": "streamable-http"},
        "local": {"command": "srv", "args": ["-q"]},
    }
    assert runner.mcp_config_from_env(None) == {}


_THINKING_PROBE = r"""
import json, sys
sys.path.insert(0, "/src")
import openhands.sdk  # sets litellm.modify_params = True, which arms the drop
import litellm
from runner import pin_adaptive_thinking
from litellm.llms.anthropic.chat.transformation import AnthropicConfig
assert litellm.modify_params is True

# The shape that made LiteLLM drop thinking: the last assistant turn called a
# tool and carried no thinking block.
messages = [
    {"role": "user", "content": "go"},
    {"role": "assistant", "content": "", "tool_calls": [
        {"id": "t1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
    {"role": "tool", "tool_call_id": "t1", "content": "ok"},
]
tools = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {}}}}]

def body(thinking):
    params = {"max_tokens": 1000, "thinking": thinking, "tools": list(tools)}
    return AnthropicConfig().transform_request("claude-opus-5", list(messages), params, {}, {})

out = {"stock_adaptive": "thinking" in body({"type": "adaptive"})}
pin_adaptive_thinking(sys.argv[1])
out["pinned_adaptive"] = body({"type": "adaptive", "display": "summarized"}).get("thinking")
out["pinned_manual_kept_guard"] = "thinking" not in body({"type": "enabled", "budget_tokens": 500})
print(json.dumps(out))
"""


def _runtime_image_ready() -> bool:
    if not docker_is_usable():
        return False
    return subprocess.run(["docker", "image", "inspect", "openhands-runtime:latest"],
                          capture_output=True).returncode == 0


@pytest.mark.skipif(not _runtime_image_ready(), reason="needs docker and openhands-runtime:latest")
@pytest.mark.parametrize("display", ["omitted", "summarized"])
def test_adaptive_thinking_survives_a_tool_turn_with_the_chosen_display(display):
    """Against the LiteLLM the runtime image actually pins: stock LiteLLM drops
    adaptive thinking after a tool turn with no thinking block (which is what
    left every later block empty); the pin keeps it, with the display asked for,
    and leaves LiteLLM's guard for manual thinking alone."""
    proc = subprocess.run(
        ["docker", "run", "--rm", "--network", "none", "-v", f"{AGENT_DIR}:/src:ro",
         "-e", "OPENHANDS_SUPPRESS_BANNER=1", "-e", "LITELLM_LOCAL_MODEL_COST_MAP=True",
         "-e", "LITELLM_LOCAL_ANTHROPIC_BETA_HEADERS=True", "openhands-runtime:latest",
         "/opt/openhands-runtime/venv/bin/python", "-c", _THINKING_PROBE, display],
        capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, proc.stderr[-2000:]
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out["stock_adaptive"] is False, "LiteLLM no longer drops it; the pin may be unneeded"
    assert out["pinned_adaptive"] == {"type": "adaptive", "display": display}
    assert out["pinned_manual_kept_guard"] is True


_REFUSAL_PROBE = r"""
import json, sys, types
sys.path.insert(0, "/src")
import runner
from litellm.llms.anthropic.chat.transformation import AnthropicConfig
from litellm import ModelResponse

def fake_original(self, *args, **kwargs):
    r = ModelResponse(id=kwargs.get("rid", "msg_x"))
    r.choices[0].finish_reason = kwargs.get("finish", "stop")
    return r

AnthropicConfig.transform_response = fake_original
seen = []
runner.count_refusals(seen.append)
cfg = AnthropicConfig()
normal = cfg.transform_response(rid="msg_ok", finish="tool_calls")
refused = cfg.transform_response(rid="msg_no", finish="content_filter")
print(json.dumps({"seen": seen, "counted": runner.REFUSALS,
                  "unchanged": [normal.id, refused.choices[0].finish_reason]}))
"""


@pytest.mark.skipif(not _runtime_image_ready(), reason="needs docker and openhands-runtime:latest")
def test_refusals_are_counted_and_passed_through_untouched():
    """A refusal (LiteLLM: finish_reason content_filter) is recorded so the run
    is reported as a refusal rather than as an agent that got stuck, and the
    response itself reaches the SDK exactly as it arrived."""
    proc = subprocess.run(
        ["docker", "run", "--rm", "--network", "none", "-v", f"{AGENT_DIR}:/src:ro",
         "-e", "OPENHANDS_SUPPRESS_BANNER=1", "-e", "LITELLM_LOCAL_MODEL_COST_MAP=True",
         "-e", "LITELLM_LOCAL_ANTHROPIC_BETA_HEADERS=True", "openhands-runtime:latest",
         "/opt/openhands-runtime/venv/bin/python", "-c", _REFUSAL_PROBE],
        capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, proc.stderr[-2000:]
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out == {"seen": ["msg_no"], "counted": ["msg_no"],
                   "unchanged": ["msg_ok", "content_filter"]}


# A fake Anthropic endpoint that answers the statuses in `plan` first, the way a
# bridge answers when the host has lost its network, and then `reply`.
_FLAKY_ANTHROPIC = r"""
import json, sys, threading
from http.server import BaseHTTPRequestHandler, HTTPServer
sys.path.insert(0, "/src")
import runner

plan = []
hits = []
reply = {"content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn"}

class H(BaseHTTPRequestHandler):
    def do_POST(self):
        self.rfile.read(int(self.headers.get("content-length") or 0))
        status = plan.pop(0) if plan else 200
        hits.append(status)
        if status == 200:
            body = {"id": f"msg_{len(hits)}", "type": "message", "role": "assistant",
                    "model": "claude-opus-5", "stop_sequence": None,
                    "usage": {"input_tokens": 1, "output_tokens": 1}, **reply}
        else:
            body = {"type": "error", "error": {"type": "api_error", "message":
                    "upstream network error: [Errno 8] nodename nor servname provided, or not known"}}
        raw = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *a):
        pass

srv = HTTPServer(("127.0.0.1", 0), H)
threading.Thread(target=srv.serve_forever, daemon=True).start()
base = f"http://127.0.0.1:{srv.server_port}"
"""

_BAD_GATEWAY_PROBE = _FLAKY_ANTHROPIC + r"""
from openhands.sdk import LLM, Message, TextContent
import openhands.sdk.llm.llm as sdk_llm
from litellm.exceptions import BadGatewayError

retries = []
msgs = [Message(role="user", content=[TextContent(text="hi")])]

def attempt(num_retries, statuses):
    hits.clear(); retries.clear(); plan[:] = statuses
    llm = LLM(model="anthropic/claude-opus-5", api_key="x", base_url=base, usage_id="probe",
              num_retries=num_retries, retry_min_wait=0, retry_max_wait=0, retry_multiplier=0,
              retry_listener=lambda a, t, e: retries.append([a, t, getattr(e, "status_code", None)]))
    try:
        llm.completion(msgs)
        outcome = "ok"
    except Exception as exc:  # noqa: BLE001
        outcome = type(exc).__name__
    return {"outcome": outcome, "hits": list(hits), "retries": list(retries)}

out = {"stock": attempt(5, [502, 502])}
out["installed"] = [runner.retry_bad_gateway(), runner.retry_bad_gateway()]
out["listed"] = sum(k is BadGatewayError for k in sdk_llm.LLM_RETRY_EXCEPTIONS)
out["recovers"] = attempt(5, [502, 502])
out["gives_up"] = attempt(3, [502] * 5)
print(json.dumps(out))
"""


def _in_runtime(probe: str) -> dict:
    proc = subprocess.run(
        ["docker", "run", "--rm", "--network", "none", "-v", f"{AGENT_DIR}:/src:ro",
         "-e", "OPENHANDS_SUPPRESS_BANNER=1", "-e", "LITELLM_LOCAL_MODEL_COST_MAP=True",
         "-e", "LITELLM_LOCAL_ANTHROPIC_BETA_HEADERS=True", "openhands-runtime:latest",
         "/opt/openhands-runtime/venv/bin/python", "-c", probe],
        capture_output=True, text=True, timeout=240)
    assert proc.returncode == 0, proc.stderr[-2000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.mark.skipif(not _runtime_image_ready(), reason="needs docker and openhands-runtime:latest")
def test_a_bridge_502_is_retried_with_backoff_not_fatal():
    """A bridge answers 502 when the host briefly loses its network. The pinned
    SDK does not retry 502 (ed8fbb42 on glm-5.3 died on one); with the patch it
    backs off and retries like a 503, and still gives up after num_retries."""
    out = _in_runtime(_BAD_GATEWAY_PROBE)
    assert out["stock"] == {"outcome": "BadGatewayError", "hits": [502], "retries": []}, (
        "the SDK retries 502 by itself now; retry_bad_gateway may be unneeded")
    assert out["installed"] == [True, True] and out["listed"] == 1
    assert out["recovers"] == {"outcome": "ok", "hits": [502, 502, 200],
                               "retries": [[1, 5, 502], [2, 5, 502]]}
    assert out["gives_up"] == {"outcome": "BadGatewayError", "hits": [502, 502, 502],
                               "retries": [[1, 3, 502], [2, 3, 502]]}


_RUN_THROUGH_A_502_PROBE = _FLAKY_ANTHROPIC + r"""
import os
from pathlib import Path

plan[:] = [502]
reply = {"stop_reason": "tool_use", "content": [
    {"type": "tool_use", "id": "toolu_1", "name": "finish", "input": {"message": "done"}}]}
Path("/tmp/ws").mkdir()
Path("/tmp/instruction.md").write_text("Say done.")
os.environ.update(LLM_MODEL="anthropic/claude-opus-5", LLM_API_KEY="x", LLM_BASE_URL=base,
                  MAX_ITERATIONS="3", MAX_CONTINUATIONS="0")
rc = runner.main(["--instruction-file", "/tmp/instruction.md", "--logs-dir", "/tmp/logs",
                  "--workspace", "/tmp/ws"])
lines = [json.loads(l) for l in Path("/tmp/logs/openhands.txt").read_text().splitlines()]
print(json.dumps({
    "rc": rc, "hits": hits,
    "retries": [{k: l.get(k) for k in ("attempt", "max_attempts", "error_status")}
                for l in lines if l.get("subtype") == "api_retry"],
    "result": [l.get("subtype") for l in lines if l.get("type") == "result"]}))
"""


@pytest.mark.skipif(not _runtime_image_ready(), reason="needs docker and openhands-runtime:latest")
def test_a_run_rides_out_a_502_and_logs_the_retry():
    """The whole runner, at the SDK's real backoff (8 s before the first retry):
    the run finishes normally, and the stream says the model was unreachable."""
    out = _in_runtime(_RUN_THROUGH_A_502_PROBE)
    assert out == {"rc": 0, "hits": [502, 200],
                   "retries": [{"attempt": 1, "max_attempts": 5, "error_status": 502}],
                   "result": ["success"]}


# =============================================================================
# 2. AGENT -- under harbor's own interpreter
# =============================================================================

def _harbor_python() -> str | None:
    exe = shutil.which("harbor")
    if not exe:
        return None
    cand = Path(os.path.realpath(exe)).parent / "python"
    return str(cand) if cand.exists() else None


def _in_harbor(code: str, **env) -> subprocess.CompletedProcess:
    py = _harbor_python()
    if not py:
        pytest.skip("harbor is not installed")
    full_env = {**os.environ, "PYTHONPATH": str(REPO), **env}
    for key in ("OPENHANDS_LLM_BASE_URL", "OPENHANDS_LLM_API_KEY"):
        if key not in env:
            full_env.pop(key, None)
    return subprocess.run([py, "-c", code], capture_output=True, text=True, env=full_env,
                          cwd=str(REPO), timeout=120)


def test_harbor_resolves_the_import_path():
    proc = _in_harbor(
        "from harbor.agents.factory import AgentFactory\n"
        "from harbor.models.trial.config import AgentConfig\n"
        "c = AgentFactory.get_agent_class_from_config(AgentConfig(name='tools.openhands_agent.agent:OpenHandsAgent'))\n"
        "print(c.__name__, c.name())\n")
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.split() == ["OpenHandsAgent", "openhands"]


def test_preflight_refuses_a_run_with_no_bridge():
    proc = _in_harbor(
        "from tools.openhands_agent.agent import OpenHandsAgent\n"
        "try:\n"
        "    OpenHandsAgent.preflight({}, {})\n"
        "except ValueError as e:\n"
        "    print('REFUSED', e)\n")
    assert "REFUSED" in proc.stdout, proc.stderr[-2000:]
    assert "OPENHANDS_LLM_API_KEY" in proc.stdout


def test_thinking_display_only_takes_the_two_measured_values():
    proc = _in_harbor(
        "from tools.openhands_agent.agent import OpenHandsAgent\n"
        "print(OpenHandsAgent.parse_options({'thinking_display': 'summarized'}).thinking_display)\n"
        "try:\n"
        "    OpenHandsAgent.parse_options({'thinking_display': 'full'})\n"
        "except ValueError:\n"
        "    print('REFUSED')\n")
    assert proc.stdout.split() == ["summarized", "REFUSED"], proc.stderr[-2000:]


def test_unknown_agent_kwarg_is_refused():
    proc = _in_harbor(
        "from tools.openhands_agent.agent import OpenHandsAgent\n"
        "try:\n"
        "    OpenHandsAgent.parse_options({'thinking': 'adaptive'})\n"
        "except ValueError as e:\n"
        "    print('REFUSED', e)\n")
    assert "REFUSED" in proc.stdout, proc.stderr[-2000:]


def test_runner_env_carries_model_bridge_and_namespaced_servers(tmp_path):
    proc = _in_harbor(
        "import json, pathlib\n"
        "from harbor.models.task.config import MCPServerConfig\n"
        "from tools.openhands_agent.agent import OpenHandsAgent\n"
        f"a = OpenHandsAgent(logs_dir=pathlib.Path({str(tmp_path)!r}), model_name='claude-opus-5',\n"
        "    mcp_servers=[MCPServerConfig(name='LightXero', transport='streamable-http',\n"
        "                                 url='http://light-servers:9139/mcp')],\n"
        "    max_iterations=40)\n"
        "print(json.dumps(a._runner_env()))\n",
        OPENHANDS_LLM_BASE_URL="http://host.docker.internal:8765",
        OPENHANDS_LLM_API_KEY="ccb-test")
    assert proc.returncode == 0, proc.stderr[-2000:]
    env = json.loads(proc.stdout.strip().splitlines()[-1])
    assert env["LLM_MODEL"] == "anthropic/claude-opus-5"
    assert env["LLM_BASE_URL"] == "http://host.docker.internal:8765"
    assert env["LLM_API_KEY"] == "ccb-test"
    assert env["MAX_ITERATIONS"] == "40"
    assert env["LLM_THINKING_DISPLAY"] == "summarized"
    assert json.loads(env["MCP_SERVERS_JSON"]) == [
        {"name": "LightXero", "transport": "streamable-http", "url": "http://light-servers:9139/mcp"}]


@pytest.mark.parametrize("rc,output,expected", [
    (137, "rate_limit_error everywhere", "NonZeroAgentExitCodeError"),
    (1, '{"ccbridge": {"kind": "subscription_cap"}}', "ApiUsageLimitError"),
    (1, "ccbridge: missing/invalid bridge secret", "AgentAuthenticationError"),
    (1, "litellm.APIConnectionError: Connection refused", "NetworkConnectionError"),
])
def test_failures_are_classified_by_how_they_died(tmp_path, rc, output, expected):
    proc = _in_harbor(
        "import pathlib, types\n"
        "from tools.openhands_agent.agent import OpenHandsAgent\n"
        f"a = OpenHandsAgent(logs_dir=pathlib.Path({str(tmp_path)!r}), model_name='claude-opus-5')\n"
        f"r = types.SimpleNamespace(return_code={rc}, stdout={output!r}, stderr='')\n"
        "print(type(a._classify_exec_error('run', r)).__name__)\n")
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.strip().splitlines()[-1] == expected


def test_trajectory_validates_as_atif(recorded, tmp_path):
    rec, _ = recorded
    traj = runner.build_trajectory(
        rec, session_id="s1", sdk_version="1.49.2", tool_definitions=[],
        per_response={"r1": {"prompt_tokens": 900, "completion_tokens": 40, "cache_read_tokens": 500,
                             "cache_write_tokens": 300, "reasoning_tokens": 5}},
        totals={"prompt_tokens": 1000, "completion_tokens": 50, "cache_read_tokens": 600,
                "cache_write_tokens": 300, "reasoning_tokens": 10, "cost_usd": 0.01},
        extra={})
    path = tmp_path / "trajectory.json"
    path.write_text(json.dumps(traj))
    proc = _in_harbor(
        "import json, sys\n"
        "from harbor.models.trajectories.trajectory import Trajectory\n"
        f"t = Trajectory.model_validate(json.load(open({str(path)!r})))\n"
        "print(len(t.steps), t.final_metrics.total_prompt_tokens)\n")
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert proc.stdout.split() == ["5", "1000"]
    agent_steps = [s for s in traj["steps"] if s["source"] == "agent"]
    assert agent_steps[0]["metrics"]["cached_tokens"] == 500
    assert agent_steps[0]["observation"]["results"][0]["content"] == '{"status": "ok", "items": 24}'


# =============================================================================
# 3. DISPATCH -- run_task.sh against a stub harbor and a fake ccbridge
# =============================================================================

_HARBOR_STUB = """#!/usr/bin/env bash
printf '%s\\n' "$@" >> "$HARBOR_ARGS"
env >> "$HARBOR_ENV"
mkdir -p "$JOB_DIR"
touch "$JOB_DIR/result.json"
exit 0
"""

SECRET = "ccb-test-secret"


class _FakeBridge(BaseHTTPRequestHandler):
    """Answers the two questions run_task.sh asks the ccbridge."""

    def _authorised(self) -> bool:
        return self.headers.get("x-api-key") == SECRET

    def do_GET(self):  # noqa: N802
        body = {"ok": True, **({"token_prefix": "sk-ant-oat01-xxx..."} if self._authorised() else {})}
        self._send(200, body)

    def do_POST(self):  # noqa: N802
        self.rfile.read(int(self.headers.get("content-length", 0)))
        if not self._authorised():
            self._send(401, {"type": "error", "error": {"type": "authentication_error"}})
            return
        self._send(200, {"type": "message", "content": [{"type": "text", "text": "p"}]})

    def _send(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture()
def fake_bridge():
    server = HTTPServer(("127.0.0.1", 0), _FakeBridge)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server.server_address[1]
    server.shutdown()


def _dispatch(tmp_path: Path, port: int, **overrides) -> SimpleNamespace:
    if not mirror_harbor_package(tmp_path):
        pytest.skip("harbor is not installed; cannot mirror its package")
    task = tmp_path / "tasks" / "alpha"
    task.mkdir(parents=True)
    (task / "task.toml").write_text('name = "acme/alpha"\n')
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "harbor").write_text(_HARBOR_STUB)
    (bin_dir / "harbor").chmod(0o755)
    args_file, env_file = tmp_path / "argv.txt", tmp_path / "env.txt"
    env = dict(os.environ)
    for key in ("NETWORK_ISOLATION_OFF", "CC_MODE", "ANTHROPIC_BASE_URL", "AGENT",
                "OPENHANDS_LLM_BASE_URL", "OPENHANDS_LLM_API_KEY"):
        env.pop(key, None)
    env.update({
        "PATH": f"{bin_dir}:{env['PATH']}",
        "OUTPUT_DIR": str(tmp_path / "output"), "JOB": "alpha",
        "JOB_DIR": str(tmp_path / "output" / "alpha"),
        "HARBOR_ARGS": str(args_file), "HARBOR_ENV": str(env_file),
        "RUN_OFFSET": "0", "MODEL": "claude-opus-5", "N": "1",
        # The fake bridge stands in for a SHARED one; the per-run lifecycle
        # has its own test below, against a real bridge and a fake upstream.
        "CCBRIDGE_SHARED": "1",
        "CCBRIDGE_PORT": str(port), "CCBRIDGE_SECRET": SECRET,
        "AGENT_HEADROOM_ENABLED": "false", "GRADER_HEADROOM_ENABLED": "false",
        "SKIP_IMAGE_REFRESH": "1",
    })
    for key, value in overrides.items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = str(value)
    proc = subprocess.run([str(RUN_TASK), "--stage", "harbor", str(task)], capture_output=True,
                          text=True, env=env, cwd=str(REPO), timeout=240)
    argv = [a for a in args_file.read_text().split("\n") if a] if args_file.exists() else []
    henv: dict[str, str] = {}
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            k, sep, v = line.partition("=")
            if sep:
                henv.setdefault(k, v)
    return SimpleNamespace(returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr,
                           argv=argv, env=henv)


def _overlays(run) -> list[str]:
    return [run.argv[i + 1] for i, a in enumerate(run.argv) if a == "--extra-docker-compose"]


def test_openhands_is_the_default_and_goes_through_the_bridge(tmp_path, fake_bridge):
    run = _dispatch(tmp_path, fake_bridge, NETWORK_ISOLATION_OFF="1")
    assert run.returncode == 0, run.stderr[-3000:]
    assert run.argv[run.argv.index("--agent") + 1] == "tools.openhands_agent.agent:OpenHandsAgent"
    assert run.argv[run.argv.index("--model") + 1] == "claude-opus-5"
    assert str(AGENT_DIR / "overlay.yaml") in _overlays(run)
    # Open network: squid is not in the path, so its ccbridge config is not either.
    assert not any("overlay-ccbridge" in o for o in _overlays(run))
    assert run.env["OPENHANDS_LLM_BASE_URL"] == f"http://host.docker.internal:{fake_bridge}"
    assert run.env["OPENHANDS_LLM_API_KEY"] == SECRET
    assert run.env["PYTHONPATH"].split(":")[0] == str(REPO)
    # The claude-code layers are claude-code kwargs; harbor would reject them here.
    assert not any(a.startswith(("thinking=", "disallowed_tools=", "config="))
                   for a in run.argv)
    assert SECRET not in run.stdout + run.stderr, "the bridge secret was printed"


@pytest.mark.skipif(not docker_is_usable(), reason="needs docker (egress-proxy image)")
def test_isolation_adds_the_ccbridge_squid_config(tmp_path, fake_bridge):
    run = _dispatch(tmp_path, fake_bridge)
    assert run.returncode == 0, run.stderr[-3000:]
    overlays = _overlays(run)
    assert overlays.index(str(PROXY_DIR / "overlay.yaml")) < overlays.index(
        str(PROXY_DIR / "overlay-ccbridge.yaml"))
    assert f"squid also allows the ccbridge at host.docker.internal:{fake_bridge}" in run.stderr
    # One config per invocation (concurrent runs have different bridge ports),
    # removed when the invocation exits.
    conf = Path(run.env["EGRESS_SQUID_CONF"])
    assert conf.name.startswith("squid-") and not conf.exists()


def test_agent_kwargs_come_from_the_openhands_env(tmp_path, fake_bridge):
    run = _dispatch(tmp_path, fake_bridge, NETWORK_ISOLATION_OFF="1",
                    OPENHANDS_MAX_ITERATIONS="42", OPENHANDS_REASONING_EFFORT="medium",
                    OPENHANDS_THINKING_DISPLAY="summarized", OPENHANDS_NUM_RETRIES="8")
    assert run.returncode == 0, run.stderr[-3000:]
    aks = [run.argv[i + 1] for i, a in enumerate(run.argv) if a == "--ak"]
    assert "max_iterations=42" in aks and "reasoning_effort=medium" in aks
    assert "thinking_display=summarized" in aks and "num_retries=8" in aks


def test_a_bridge_holding_another_secret_is_refused(tmp_path, fake_bridge):
    run = _dispatch(tmp_path, fake_bridge, NETWORK_ISOLATION_OFF="1", CCBRIDGE_SECRET="wrong")
    assert run.returncode != 0
    assert "does not accept this harness's secret" in run.stderr
    assert not run.argv, "harbor ran against a bridge that would refuse every turn"


def _free_port() -> int:
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.skipif(shutil.which("uv") is None, reason="the ccbridge runs from its uv project")
def test_a_machine_with_no_usable_login_fails_fast_with_the_reason(tmp_path):
    """No bridge running and no loadable Claude login: the bridge's own --check
    says why, in seconds, before anything is launched. It used to start a bridge
    that died at once and then poll the dead port for the full timeout."""
    import time
    port = _free_port()
    started = time.monotonic()
    run = _dispatch(tmp_path, port, NETWORK_ISOLATION_OFF="1",
                    CLAUDE_CODE_CREDENTIALS="not-json", CCBRIDGE_START_TIMEOUT_SEC="120")
    assert run.returncode == 4, run.stderr[-3000:]
    assert "cannot load a Claude login" in run.stderr
    assert "not valid JSON" in run.stderr
    assert time.monotonic() - started < 90
    assert not run.argv, "harbor ran with no model behind the agent"


class _FakeAnthropic(BaseHTTPRequestHandler):
    """api.anthropic.com as far as the bridge can tell: answers /v1/messages."""

    seen: list = []

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))) or b"{}")
        type(self).seen.append({"path": self.path, "auth": self.headers.get("authorization"),
                                "model": body.get("model")})
        data = json.dumps({"id": "msg_1", "type": "message", "role": "assistant",
                           "model": body.get("model"), "stop_reason": "end_turn",
                           "content": [{"type": "text", "text": "p"}],
                           "usage": {"input_tokens": 1, "output_tokens": 1}}).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):  # noqa: N802 -- /healthz, when this stands in for zbridge
        data = b'{"ok": true}'
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture()
def fake_anthropic():
    _FakeAnthropic.seen = []
    server = HTTPServer(("127.0.0.1", 0), _FakeAnthropic)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server.server_address[1]
    server.shutdown()


def _listening(port: int) -> bool:
    import socket
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


@pytest.mark.skipif(shutil.which("uv") is None, reason="the ccbridge runs from its uv project")
def test_each_run_starts_and_stops_its_own_bridge(tmp_path, fake_anthropic):
    """The default lifecycle, end to end with the real bridge: a free port and a
    fresh secret for this run only, the live check routed through the bridge
    with the machine's token, and nothing left running afterwards."""
    run = _dispatch(tmp_path, 0, NETWORK_ISOLATION_OFF="1", CCBRIDGE_SHARED="0",
                    CCBRIDGE_PORT=None, CCBRIDGE_SECRET=None,
                    CLAUDE_CODE_OAUTH_TOKEN="sk-ant-oat01-per-run-test",
                    CCBRIDGE_UPSTREAM=f"http://127.0.0.1:{fake_anthropic}")
    assert run.returncode == 0, run.stderr[-3000:]
    base = run.env["OPENHANDS_LLM_BASE_URL"]
    port = int(base.rsplit(":", 1)[1])
    assert base == f"http://host.docker.internal:{port}" and port != 8765
    secret = run.env["OPENHANDS_LLM_API_KEY"]
    assert secret.startswith("ccb-run-")
    shared = REPO / "tools" / "bridges" / "ccbridge" / ".bridge_secret"
    assert not shared.exists() or shared.read_text().strip() != secret
    assert any(r["path"] == "/v1/messages" and r["auth"] == "Bearer sk-ant-oat01-per-run-test"
               for r in _FakeAnthropic.seen), _FakeAnthropic.seen
    assert "ccbridge for this run stopped" in run.stderr
    assert not _listening(port), "the run's bridge outlived the run"
    assert secret not in run.stdout + run.stderr, "the per-run secret was printed"


def test_glm_runs_the_same_agent_through_zbridge(tmp_path, fake_anthropic):
    """CC_MODE=zbridge: the OpenHands agent on GLM. zbridge speaks the same
    Anthropic protocol as the ccbridge, so only where the agent points changes;
    no ccbridge is started and the model defaults to glm-5.3."""
    # zbridge runs without auth (ensure_zbridge), so its stand-in answers anyone.
    run = _dispatch(tmp_path, 1, NETWORK_ISOLATION_OFF="1", CC_MODE="zbridge",
                    ZB_PORT=str(fake_anthropic), ZB_ZAI_API_KEY="zai-test", MODEL=None,
                    CCBRIDGE_SHARED=None)
    assert run.returncode == 0, run.stderr[-3000:]
    assert run.argv[run.argv.index("--agent") + 1] == "tools.openhands_agent.agent:OpenHandsAgent"
    assert run.argv[run.argv.index("--model") + 1] == "glm-5.3"
    assert run.env["OPENHANDS_LLM_BASE_URL"] == f"http://host.docker.internal:{fake_anthropic}"
    assert "ccbridge" not in run.stdout.replace("zbridge", "")


@pytest.mark.skipif(not docker_is_usable(), reason="needs docker (egress-proxy image)")
def test_glm_under_isolation_opens_zbridge_not_the_ccbridge(tmp_path, fake_anthropic):
    run = _dispatch(tmp_path, 1, CC_MODE="zbridge", ZB_PORT=str(fake_anthropic),
                    ZB_ZAI_API_KEY="zai-test", MODEL=None, CCBRIDGE_SHARED=None)
    assert run.returncode == 0, run.stderr[-3000:]
    overlays = _overlays(run)
    assert str(PROXY_DIR / "overlay-zbridge.yaml") in overlays
    assert str(PROXY_DIR / "overlay-ccbridge.yaml") not in overlays
    assert f"acl zbridge_port port {fake_anthropic}" in Path(run.env["EGRESS_SQUID_CONF"]).read_text()


def test_a_job_with_claude_code_runs_warns_before_mixing_agents(tmp_path, fake_bridge):
    job = tmp_path / "output" / "alpha"
    job.mkdir(parents=True)
    (job / "config.json").write_text(json.dumps(
        {"agents": [{"name": "claude-code", "model_name": "claude-opus-5"}]}))
    run = _dispatch(tmp_path, fake_bridge, NETWORK_ISOLATION_OFF="1")
    assert run.returncode == 0, run.stderr[-3000:]
    assert "holds runs from agent 'claude-code'" in run.stderr


def test_claude_code_is_still_dispatched_as_before(tmp_path, fake_bridge):
    run = _dispatch(tmp_path, fake_bridge, NETWORK_ISOLATION_OFF="1", AGENT="claude-code")
    assert run.returncode == 0, run.stderr[-3000:]
    assert run.argv[run.argv.index("--agent") + 1] == "claude-code"
    assert not any("openhands_agent" in o for o in _overlays(run))
    assert "OPENHANDS_LLM_API_KEY" not in run.env


# =============================================================================
# 4. The overlay's promises, read from the file
# =============================================================================

def test_overlay_mounts_the_runtime_read_only_and_withholds_the_token():
    doc = yaml.safe_load((AGENT_DIR / "overlay.yaml").read_text())
    main = doc["services"]["main"]
    assert "openhands_runtime:/opt/openhands-runtime:ro" in main["volumes"]
    assert main["environment"]["CLAUDE_CODE_OAUTH_TOKEN"] == ""
    assert main["depends_on"]["openhands-runtime"]["condition"] == "service_started"
    sidecar = doc["services"]["openhands-runtime"]
    assert sidecar["network_mode"] == "none"
    assert "build" not in sidecar
