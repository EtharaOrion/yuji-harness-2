#!/usr/bin/env python3
"""End-to-end smoke test for the MCP-Atlas -> Harbor pipeline.

Runs without Docker and without model credentials, so it is safe in CI and on
a laptop. It covers the path that used to be untested end to end:

  1. generate a Harbor bundle from a real dataset row
  2. check the bundle's structure (allowlist, compose `main`, staged bridge)
  3. validate task.toml against Harbor's own task model, when the harbor CLI
     is installed
  4. start the REST->MCP bridge for real against a stub sandbox and drive a
     full MCP handshake through it -- initialize, tools/list, tools/call --
     proving the agent would actually receive the task's tools
  5. run the oracle and confirm the verifier can extract a response from it

Usage:
    python3 scripts/smoke_test.py            # all checks
    python3 scripts/smoke_test.py --quick    # skip the live bridge check
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ADAPTER_DIR = REPO / "adapters" / "mcp_atlas"

PASSED: list[str] = []
FAILED: list[tuple[str, str]] = []
SKIPPED: list[tuple[str, str]] = []


class Skip(Exception):
    """Raised by a check that cannot run in this environment."""


def check(name: str):
    def decorator(fn):
        def wrapper(*args, **kwargs):
            try:
                fn(*args, **kwargs)
            except Skip as exc:
                SKIPPED.append((name, str(exc)))
                print(f"  SKIP  {name} -- {exc}")
            except Exception as exc:  # noqa: BLE001 - smoke test reports, not raises
                FAILED.append((name, f"{type(exc).__name__}: {exc}"))
                print(f"  FAIL  {name}")
                print(f"        {type(exc).__name__}: {exc}")
            else:
                PASSED.append(name)
                print(f"  ok    {name}")

        return wrapper

    return decorator


def load_adapter():
    sys.path.insert(0, str(ADAPTER_DIR))
    spec = importlib.util.spec_from_file_location("mcp_atlas_adapter", ADAPTER_DIR / "adapter.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


# --------------------------------------------------------------------------
# stub sandbox: the REST API the real image serves on :1984
# --------------------------------------------------------------------------

# A tool the sandbox serves that no task ever grants. The bridge must hide it,
# or a task is silently solvable with tools it was never scoped to.
UNGRANTED_TOOL = "smoke_ungranted_tool"

# Populated from the task under test in _bridge_live so the check works for
# any dataset row, not just one whose allowlist happens to match a hardcoded
# list. (An earlier hardcoded stub declared `exa_web_search_exa` un-granted --
# but the sample row does grant it, so the check failed on the test's own
# assumption rather than on the bridge.)
STUB_TOOLS: list[dict] = []


class StubSandbox(BaseHTTPRequestHandler):
    calls: list[dict] = []

    def log_message(self, *args):  # silence per-request logging
        pass

    def _json(self, payload, status=200):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._json({"status": "health_and_client_connection_ok"})
        else:
            self._json({"detail": "not found"}, 404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        if self.path == "/list-tools":
            self._json(STUB_TOOLS)
        elif self.path == "/call-tool":
            StubSandbox.calls.append(payload)
            self._json([{"type": "text", "text": f"called {payload.get('tool_name')}"}])
        else:
            self._json({"detail": "not found"}, 404)


def start_stub_sandbox() -> tuple[HTTPServer, str]:
    server = HTTPServer(("127.0.0.1", 0), StubSandbox)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_port}"


# --------------------------------------------------------------------------
# minimal MCP stdio client
# --------------------------------------------------------------------------


class MCPStdioClient:
    """Just enough JSON-RPC to drive an MCP stdio server."""

    def __init__(self, proc: subprocess.Popen):
        self.proc = proc
        self._id = 0

    def _send(self, message: dict) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(json.dumps(message) + "\n")
        self.proc.stdin.flush()

    def _read_response(self, expect_id: int, timeout: float = 60.0) -> dict:
        assert self.proc.stdout is not None
        deadline = time.time() + timeout
        while time.time() < deadline:
            line = self.proc.stdout.readline()
            if not line:
                stderr = self.proc.stderr.read() if self.proc.stderr else ""
                raise RuntimeError(f"bridge exited early. stderr:\n{stderr}")
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue  # not protocol traffic
            if message.get("id") == expect_id:
                if "error" in message:
                    raise RuntimeError(f"MCP error: {message['error']}")
                return message.get("result", {})
        raise TimeoutError(f"no response to request id={expect_id} within {timeout}s")

    def request(self, method: str, params: dict | None = None) -> dict:
        self._id += 1
        self._send({"jsonrpc": "2.0", "id": self._id, "method": method, "params": params or {}})
        return self._read_response(self._id)

    def notify(self, method: str, params: dict | None = None) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params or {}})


# --------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------


def sample_record(adapter):
    """A real dataset row when one is available, else a representative stub."""
    csv_path = REPO / "output" / "task_688a441c.csv"
    if csv_path.is_file():
        csv.field_size_limit(sys.maxsize)
        loader = adapter.MCPAtlasLoader(source=csv_path)
        for record in loader:
            return record
    return adapter.MCPAtlasRecord(
        task_id="smoke-task",
        prompt="Find a paper and translate its title.",
        gtfa_claims=["The response names the paper", "The response includes a translation"],
        enabled_tools=["arxiv_search_papers", "whois_whois_domain", "notion_API-get-self"],
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="skip the live bridge check")
    args = parser.parse_args()

    adapter = load_adapter()
    workdir = Path(tempfile.mkdtemp(prefix="mcp-atlas-smoke-"))
    bundle_root = workdir / "bundles"

    print(f"\nmcp-atlas smoke test\nworkdir: {workdir}\n")

    print("[1/7] bundle generation")

    record = sample_record(adapter)
    generator = adapter.MCPAtlasToHarbor(out_root=bundle_root)
    bundle = generator.generate_task(record)

    @check("adapter generates a bundle")
    def _generated():
        assert bundle.is_dir(), "no bundle directory"

    _generated()

    @check("bundle contains every required file")
    def _files():
        required = [
            "task.toml",
            "instruction.md",
            "solution/solve.sh",
            "tests/test.sh",
            "tests/agent_judge.py",
            "environment/Dockerfile",
            "environment/docker-compose.yaml",
            "environment/enabled_tools.txt",
            f"environment/{adapter.BRIDGE_FILENAME}",
        ]
        missing = [rel for rel in required if not (bundle / rel).is_file()]
        assert not missing, f"missing: {missing}"

    _files()

    @check("task's tool allowlist is written to the bundle")
    def _allowlist():
        written = (bundle / "environment" / "enabled_tools.txt").read_text().split()
        assert written == record.enabled_tools, (
            f"allowlist mismatch: {len(written)} written vs {len(record.enabled_tools)} in the task"
        )

    _allowlist()

    @check("compose defines the `main` service Harbor runs the agent in")
    def _compose():
        text = (bundle / "environment" / "docker-compose.yaml").read_text()
        try:
            import yaml

            services = yaml.safe_load(text)["services"]
        except ModuleNotFoundError:
            services = {"main": {}} if "\n  main:" in text else {}
        assert "main" in services, "compose has no `main` service -- Harbor would start no agent"
        assert "mcp-server" in services, "compose has no sandbox sidecar"

    _compose()

    print("\n[2/7] Harbor schema validation")

    @check("task.toml validates against Harbor's task model")
    def _harbor_model():
        harbor_python = shutil.which("harbor")
        if not harbor_python:
            raise Skip("harbor CLI not installed")
        script = (
            "import sys, tomllib, json;"
            "from pathlib import Path;"
            "from harbor.models.task.config import TaskConfig;"
            "cfg = TaskConfig.model_validate(tomllib.loads(Path(sys.argv[1]).read_text()));"
            "print(json.dumps({"
            "'schema_version': cfg.schema_version,"
            "'agent_timeout_sec': cfg.agent.timeout_sec,"
            "'mcp_servers': [s.name for s in cfg.environment.mcp_servers],"
            "'name': cfg.task.name,"
            "'description': bool(cfg.task.description),"
            "}))"
        )
        interpreter = str(Path(os.path.realpath(harbor_python)).parent / "python")
        result = subprocess.run(
            [interpreter, "-c", script, str(bundle / "task.toml")],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, f"Harbor rejected task.toml:\n{result.stderr.strip()}"
        info = json.loads(result.stdout)
        assert info["schema_version"] == "1.3", f"stale schema: {info['schema_version']}"
        assert info["agent_timeout_sec"] == adapter.DEFAULT_AGENT_TIMEOUT, (
            f"agent timeout dropped in migration: {info['agent_timeout_sec']}"
        )
        assert info["mcp_servers"], "no MCP servers -- the agent would run with zero tools"
        assert info["description"], "empty description blocks `harbor publish`"

    _harbor_model()

    @check("Harbor round-trips the bundle without rewriting it")
    def _roundtrip():
        if not shutil.which("harbor"):
            raise Skip("harbor CLI not installed")
        copy_root = workdir / "roundtrip"
        shutil.copytree(bundle_root, copy_root)
        before = (copy_root / bundle.name / "task.toml").read_text()
        result = subprocess.run(
            ["harbor", "task", "update", bundle.name, "--org", adapter.DEFAULT_ORG],
            cwd=copy_root, capture_output=True, text=True,
        )
        assert result.returncode == 0, f"harbor task update failed:\n{result.stderr.strip()}"
        after = (copy_root / bundle.name / "task.toml").read_text()
        assert before == after, (
            "Harbor rewrote task.toml -- the adapter is not emitting native schema. Diff its output."
        )

    _roundtrip()

    print("\n[3/7] REST->MCP bridge")

    @check("bridge module compiles")
    def _bridge_compiles():
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", str(ADAPTER_DIR / adapter.BRIDGE_FILENAME)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr.strip()

    _bridge_compiles()

    @check("staged bridge matches the repo module")
    def _bridge_staged():
        staged = (bundle / "environment" / adapter.BRIDGE_FILENAME).read_text()
        source = (ADAPTER_DIR / adapter.BRIDGE_FILENAME).read_text()
        assert staged == source, "bundle ships a stale copy of the bridge"

    _bridge_staged()

    @check("bridge serves the task's tools over MCP and proxies a tool call")
    def _bridge_live():
        if args.quick:
            raise Skip("--quick")
        if not shutil.which("uv"):
            raise Skip("uv not installed (needed to resolve the bridge's inline deps)")

        granted = list(record.enabled_tools[:3])
        assert granted, "sample task grants no tools; cannot exercise the allowlist"
        STUB_TOOLS[:] = [
            {"name": name, "description": name, "inputSchema": {"type": "object"}}
            for name in granted + [UNGRANTED_TOOL]
        ]

        server, url = start_stub_sandbox()
        StubSandbox.calls = []
        try:
            env = {
                **os.environ,
                "MCP_SERVER_URL": url,
                "ENABLED_TOOLS_FILE": str(bundle / "environment" / "enabled_tools.txt"),
            }
            proc = subprocess.Popen(
                ["uv", "run", "--no-project", str(ADAPTER_DIR / adapter.BRIDGE_FILENAME)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, bufsize=1, env=env,
            )
            try:
                client = MCPStdioClient(proc)
                init = client.request(
                    "initialize",
                    {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "clientInfo": {"name": "smoke", "version": "1"},
                    },
                )
                assert "serverInfo" in init, f"no serverInfo in initialize result: {init}"
                client.notify("notifications/initialized")

                listed = client.request("tools/list")
                names = {t["name"] for t in listed.get("tools", [])}
                assert names, "bridge exposed no tools"

                assert names == set(granted), (
                    f"allowlist not enforced. exposed={sorted(names)} expected={sorted(granted)}"
                )
                assert UNGRANTED_TOOL not in names, "un-granted tool leaked to the agent"

                # The bridge must also refuse a direct call to a tool it hid,
                # not just omit it from tools/list.
                refused = client.request(
                    "tools/call", {"name": UNGRANTED_TOOL, "arguments": {}}
                )
                assert refused.get("isError"), "bridge proxied a call to an un-granted tool"

                target = granted[0]
                called = client.request("tools/call", {"name": target, "arguments": {"q": "x"}})
                text = " ".join(
                    block.get("text", "") for block in called.get("content", [])
                )
                assert target in text, f"tool call did not reach the sandbox: {called}"
                assert StubSandbox.calls and StubSandbox.calls[0]["tool_name"] == target, (
                    f"sandbox never saw the call: {StubSandbox.calls}"
                )
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
        finally:
            server.shutdown()

    _bridge_live()

    print("\n[4/7] task data + artifacts")

    @check("a data-carrying task mounts its files where the tools can read them")
    def _data_task():
        manifest = REPO / "examples" / "tasks" / "vendor-audit" / "task.jsonl"
        if not manifest.is_file():
            raise Skip("examples/tasks/vendor-audit not present")
        (data_record,) = list(adapter.MCPAtlasLoader(source=manifest))
        assert data_record.data_dir is not None, "DATA_DIR was not picked up from the manifest"

        out = adapter.MCPAtlasToHarbor(out_root=workdir / "data_bundles").generate_task(data_record)

        staged = out / "environment" / "task_data"
        assert staged.is_dir() and any(staged.iterdir()), "task data was not copied into the bundle"

        import yaml

        services = yaml.safe_load((out / "environment" / "docker-compose.yaml").read_text())["services"]
        volumes = services["mcp-server"].get("volumes") or []
        assert any(adapter.TASK_DATA_PATH in v for v in volumes), (
            f"data not mounted on the sidecar, where the MCP servers run: {volumes}"
        )
        assert all(v.endswith(":ro") for v in volumes), "task data must be read-only"
        assert not services["main"].get("volumes"), (
            "data mounted on the agent container is unreachable by the tools"
        )
        # Both paths must sit under /data: the filesystem server is rooted
        # there and desktop-commander's allowlist is ["/data"].
        assert adapter.TASK_DATA_PATH.startswith("/data/")
        assert adapter.TASK_OUTPUT_PATH.startswith("/data/")

    _data_task()

    print("\n[5/7] oracle")

    @check("oracle writes a trajectory the verifier can read")
    def _oracle():
        logs = workdir / "logs" / "agent"
        logs.mkdir(parents=True, exist_ok=True)
        script = (bundle / "solution" / "solve.sh").read_text().replace("/logs/agent", str(logs))
        result = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
        assert result.returncode == 0, f"solve.sh failed:\n{result.stderr.strip()}"

        files = list(logs.glob("*.txt"))
        assert files, "oracle produced no trajectory -- the judge would raise and score 0"

        # Mirrors agent_judge.py::extract_agent_response.
        response = ""
        for line in files[0].read_text().splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "result" and event.get("result"):
                response = event["result"]
        assert response, "verifier would extract an empty agent response"
        if record.gtfa_claims:
            missing = [c for c in record.gtfa_claims if c not in response]
            assert not missing, f"{len(missing)} ground-truth claim(s) absent from the oracle answer"

    _oracle()

    print("\n[6/7] weighted grading")

    manifests = sorted((REPO / "examples" / "tasks").glob("*/task.jsonl"))

    def _grading_check(manifest):
        @check(f"{manifest.parent.name}: oracle scores 1.0 through both grading channels")
        def _weighted_grading():
            (rec,) = list(adapter.MCPAtlasLoader(source=manifest))
            if not rec.tests_dir:
                raise Skip("task ships no grading files")
            out = adapter.MCPAtlasToHarbor(
                out_root=workdir / "weighted" / manifest.parent.name
            ).generate_task(rec)

            for rel in ("tests/test_outputs.py", "tests/test_weights.json", "tests/rubric.json",
                        "tests/traj_asserts.py", "tests/weighted_judge.py",
                        "tests/rubric_weighted.py", "tests/weighted_judge_entry.py"):
                assert (out / rel).is_file(), f"weighted bundle missing {rel}"
            assert "weighted_judge_entry.py" in (out / "tests" / "test.sh").read_text(), (
                "test.sh does not run the weighted judge"
            )

            # Replay the oracle, then grade it exactly as weighted_judge_entry.py
            # does in-container.
            logs = workdir / "wlogs" / "agent"
            logs.mkdir(parents=True, exist_ok=True)
            script = (out / "solution" / "solve.sh").read_text().replace("/logs/agent", str(logs))
            subprocess.run(["bash", "-c", script], check=True, capture_output=True)

            messages = []
            for line in (logs / "oracle.txt").read_text().splitlines():
                event = json.loads(line)
                if event.get("type") == "message" and event.get("message"):
                    messages.append(event["message"])
            assert any(m.get("tool_calls") for m in messages), (
                "oracle made no tool calls, so it can never score on Channel A"
            )

            sys.path.insert(0, str(out / "tests"))
            import weighted_judge as wj
            import rubric_weighted as rw

            weights = wj.load_weights(out / "tests" / "test_weights.json")
            # Guard tests are *supposed* to fail on a clean run (a guard passing
            # means the bad thing happened), so pytest's own report is noise here
            # -- the signal is the weighted value below.
            with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
                traj_results = wj._run_traj_pytest(messages, out / "tests" / "test_outputs.py")
            channel_a = wj.score_traj_tests(traj_results, weights.traj_tests.tests)
            assert channel_a == 1.0, f"Channel A scored {channel_a} on the reference trajectory"

            criteria = rw.parse_rubric(json.loads((out / "tests" / "rubric.json").read_text()))
            assert criteria, "rubric.json parsed to no criteria"
            scored = rw.score_verdicts(criteria, [{"id": c.id, "score": 1.0} for c in criteria])
            assert scored["value"] == 1.0, f"Channel B scored {scored['value']} on a perfect judge"

            result = wj.judge_weighted(
                test_weights_file=out / "tests" / "test_weights.json",
                traj_results=traj_results,
                rubric_value=scored["value"],
                rubric_weight=weights.rubric.weight,
                rubric_rows=scored["rows"],
            )
            assert set(result["ledger"]) == {"traj_tests", "rubric"}, (
                f"both channels must be active: {result['ledger']}"
            )
            assert result["reward"] == 1.0, f"oracle reward is {result['reward']}, expected 1.0"

        _weighted_grading()

    if not manifests:
        SKIPPED.append(("example tasks", "examples/tasks/ is empty"))
        print("  SKIP  example tasks -- examples/tasks/ is empty")
    for _manifest in manifests:
        _grading_check(_manifest)

    print("\n[7/7] unit suites")

    @check("adapter + scoring + mcp_eval unit tests")
    def _unit():
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "adapters",
             "services/scoring/tests", "services/mcp_eval/tests", "-q"],
            cwd=REPO, capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stdout.strip()[-2000:]

    _unit()

    print("\n" + "=" * 62)
    print(f"passed {len(PASSED)}   failed {len(FAILED)}   skipped {len(SKIPPED)}")
    for name, reason in SKIPPED:
        print(f"  skipped: {name} -- {reason}")
    for name, reason in FAILED:
        print(f"  FAILED : {name} -- {reason}")
    print("=" * 62)

    if FAILED:
        print(f"\nartifacts kept for inspection: {workdir}")
        return 1
    shutil.rmtree(workdir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
