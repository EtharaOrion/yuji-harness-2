"""judge_client.py: what tests/test.sh runs in `main` once the trajectory exists.

`main` grades nothing now. The client posts the trajectory to the judge
container, which runs the bundle's evaluate.sh and writes the reports itself.
A fake judge stands in for that container, so these check the client's side of
the contract -- what it writes, when it exits 0, and that it never routes the
call through main's egress proxy -- without docker or quota.
"""
from __future__ import annotations

import json
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


def test_the_call_never_goes_through_mains_egress_proxy(run, monkeypatch):
    """Under isolation main's HTTP(S)_PROXY is squid, which would 403 the judge."""
    for var in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"):
        monkeypatch.setenv(var, "http://127.0.0.1:9")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    rc, _ = run()
    assert rc == 0
