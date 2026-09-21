"""judge_client.py: what tests/test.sh runs in `main` once the trajectory exists.

`main` grades nothing now. The client posts the trajectory to the judge
container, which runs the bundle's evaluate.sh and writes the reports itself.
A fake judge stands in for that container, so these check the client's side of
the contract -- what it writes, when it exits 0, and that it never routes the
call through main's egress proxy -- without docker or quota.
"""
from __future__ import annotations

import json
import os
import stat
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import judge_client  # noqa: E402

EVALUATED = {"ok": True, "reason": None, "returncode": 0,
             "reward": {"reward": 0.42, "completion_rate": 1.0, "misbehave_rate": 0.0},
             "rubric_criteria": 12,
             "written": ["ctrf.json", "reward.json", "rubric_breakdown.json"],
             "log_tail": "[5/5 reward] {'reward': 0.42}", "graded_in": "judge-container",
             "model": "gpt-5.6-sol", "codex_version": "codex-cli 0.154.0"}


class FakeJudge:
    def __init__(self):
        self.health = []          # statuses to answer /healthz with, then 200
        self.evaluate = (200, EVALUATED)
        self.requests = []
        self.on_post = None       # called as the judge starts grading


@pytest.fixture
def judge():
    fake = FakeJudge()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, doc):
            body = json.dumps(doc).encode()
            self.send_response(code)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            code = fake.health.pop(0) if fake.health else 200
            self._send(code, {"status": "ok" if code == 200 else "unavailable",
                              "reason": None if code == 200 else "warming up"})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            fake.requests.append({"path": self.path, "token": self.headers.get("x-judge-token"),
                                  "body": body})
            if fake.on_post:
                fake.on_post()
            self._send(*fake.evaluate)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    fake.url = f"http://127.0.0.1:{srv.server_address[1]}"
    yield fake
    srv.shutdown()
    srv.server_close()


@pytest.fixture
def run(tmp_path, monkeypatch, judge):
    monkeypatch.setattr(judge_client.time, "sleep", lambda s: None)
    monkeypatch.setenv("JUDGE_URL", judge.url)
    monkeypatch.setenv("JUDGE_TOKEN", "tok")
    monkeypatch.setenv("JUDGE_WAIT_SEC", "5")
    # Never the real /workspace of whatever box runs this suite.
    monkeypatch.setenv("JUDGE_WORKSPACE", str(tmp_path / "workspace"))
    (tmp_path / "traj.json").write_text(json.dumps({"steps": [], "final_message": "done"}))
    logs = tmp_path / "verifier"

    def go():
        rc = judge_client.main(["--trajectory", str(tmp_path / "traj.json"),
                                "--logs-dir", str(logs)])
        return rc, logs
    return go


def _marker(logs):
    return json.loads((logs / judge_client.MARKER_NAME).read_text())


def test_the_trajectory_is_posted_and_the_marker_records_the_outcome(run, judge):
    rc, logs = run()
    assert rc == 0
    sent = judge.requests[0]
    assert sent["path"] == "/evaluate"
    assert sent["token"] == "tok"
    assert sent["body"]["trajectory"]["final_message"] == "done"
    marker = _marker(logs)
    assert marker["ok"] is True and marker["graded_in"] == "judge-container"
    assert marker["rubric_criteria"] == 12
    assert "reward.json" in marker["written"]


def test_the_client_writes_no_reports_of_its_own(run, judge):
    """The judge writes them into the same mounted dir; relaying them through
    main would put the numbers back in the container the agent had root in."""
    rc, logs = run()
    assert rc == 0
    assert sorted(p.name for p in logs.iterdir()) == [judge_client.MARKER_NAME]


def test_a_failed_evaluation_says_why_and_fails(run, judge):
    judge.evaluate = (200, {**EVALUATED, "ok": False, "reason": "evaluation exited 3",
                            "reward": None})
    rc, logs = run()
    assert rc == 1
    marker = _marker(logs)
    assert marker["ok"] is False and marker["graded_in"] is None
    assert marker["reason"] == "evaluation exited 3"


def test_no_token_means_no_call(run, judge, monkeypatch):
    monkeypatch.delenv("JUDGE_TOKEN")
    rc, logs = run()
    assert rc == 1 and not judge.requests
    assert "JUDGE_TOKEN" in _marker(logs)["reason"]


def test_a_refusal_surfaces_the_judges_reason(run, judge):
    judge.evaluate = (401, {"error": "bad or missing x-judge-token"})
    rc, logs = run()
    assert rc == 1
    assert "401" in _marker(logs)["reason"] and "x-judge-token" in _marker(logs)["reason"]


def test_it_waits_for_the_judge_to_become_healthy(run, judge):
    judge.health = [503, 503]
    rc, _ = run()
    assert rc == 0 and len(judge.requests) == 1


def test_an_unreachable_judge_is_reported_not_hung(run, monkeypatch):
    monkeypatch.setenv("JUDGE_URL", "http://127.0.0.1:1")
    monkeypatch.setenv("JUDGE_WAIT_SEC", "0")
    rc, logs = run()
    assert rc == 1
    assert "unreachable" in _marker(logs)["reason"]


def test_an_unreadable_trajectory_fails_before_the_call(run, judge, tmp_path):
    rc = judge_client.main(["--trajectory", str(tmp_path / "missing.json"),
                            "--logs-dir", str(tmp_path / "verifier")])
    assert rc == 1 and not judge.requests


def _mode(path):
    return stat.S_IMODE(os.lstat(path).st_mode)


def test_the_agents_files_are_readable_by_the_judge_when_it_grades(run, judge, tmp_path):
    """OpenHands' file_editor creates every file 0600; the judge reads the
    workspace as uid 1000, so an unopened report grades as an empty one."""
    out = tmp_path / "workspace" / "out"
    out.mkdir(parents=True)
    report = out / "report.md"
    report.write_text("# report\n")
    report.chmod(0o600)
    out.chmod(0o700)
    seen = {}
    judge.on_post = lambda: seen.update(report=_mode(report), out=_mode(out))
    rc, _ = run()
    assert rc == 0
    assert seen == {"report": 0o644, "out": 0o755}


def test_bits_are_only_added(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "run.sh").write_text("")
    (ws / "run.sh").chmod(0o700)
    (ws / "open.txt").write_text("")
    (ws / "open.txt").chmod(0o666)
    _, failed = judge_client.open_workspace_to_judge(ws)
    assert _mode(ws / "run.sh") == 0o744
    assert _mode(ws / "open.txt") == 0o666
    assert failed == []


def test_the_data_mount_and_symlinks_are_left_alone(tmp_path):
    """data/ is the bundle's read-only input, and a chmod through an agent's
    symlink would land on whatever it points at in main."""
    ws, outside = tmp_path / "ws", tmp_path / "secret"
    (ws / "data").mkdir(parents=True)
    (ws / "data" / "input.pdf").write_text("")
    (ws / "data" / "input.pdf").chmod(0o600)
    outside.write_text("")
    outside.chmod(0o600)
    (ws / "link").symlink_to(outside)
    judge_client.open_workspace_to_judge(ws)
    assert _mode(ws / "data" / "input.pdf") == 0o600
    assert _mode(outside) == 0o600


def test_no_workspace_is_not_an_error(run, judge):
    rc, _ = run()
    assert rc == 0 and len(judge.requests) == 1


def test_the_call_never_goes_through_mains_egress_proxy(run, monkeypatch):
    """Under isolation main's HTTP(S)_PROXY is squid, which would 403 the judge."""
    for var in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"):
        monkeypatch.setenv(var, "http://127.0.0.1:9")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    rc, _ = run()
    assert rc == 0
