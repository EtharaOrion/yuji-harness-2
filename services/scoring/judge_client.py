#!/usr/bin/env python3
"""Hand this run to the judge container, which grades every channel.

tests/test.sh runs in `main` -- the container the agent worked in -- so nothing
is graded there any more. It builds the trajectory with the bundle's own parser
and calls this, which posts the trajectory to the judge (tools/judge). The judge
runs /harness/scoring/tests/evaluate.sh: the state dump, Channel A, the rubric and
the ledger, all with the codex login and the answer files that `main` never gets.

Reports are NOT relayed back through here. The judge writes them straight into
/logs/verifier, which harbor mounts into both containers, so the files land
where harbor collects them no matter what main does afterwards.

This writes one file of its own:
  judge_container.json   whether the evaluation ran in the judge container, and
                         if not, why. The judge-container check reads it, and
                         so does scripts/run_task.sh.

Exit 0 only when the judge reports a finished evaluation with a reward written.

Env: JUDGE_TOKEN (required), JUDGE_URL (http://judge:8770),
     JUDGE_WAIT_SEC (120), JUDGE_REQUEST_TIMEOUT_SEC (1200 -- inside the
     bundle's [verifier] timeout_sec of 1800).
Stdlib only: `main` has no requests/httpx.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

MARKER_NAME = "judge_container.json"
GRADED_IN = "judge-container"

# Never through squid. The judge is a sibling on the compose bridge; under
# network isolation main's HTTP(S)_PROXY points at egress-proxy, which would
# deny it. An explicit empty ProxyHandler keeps that true even if NO_PROXY drifts.
_opener = build_opener(ProxyHandler({}))


def _error_body(exc: HTTPError) -> str:
    try:
        doc = json.loads(exc.read() or b"{}")
        return str(doc.get("error") or doc.get("reason") or doc)
    except (ValueError, OSError):
        return str(exc)


def wait_until_healthy(url: str, wait_sec: float) -> str | None:
    """None once /healthz answers 200, else the last reason it did not."""
    deadline = time.monotonic() + wait_sec
    last = "judge never answered"
    while True:
        try:
            with _opener.open(f"{url}/healthz", timeout=5) as resp:
                if resp.status == 200:
                    return None
                last = f"healthz returned {resp.status}"
        except HTTPError as exc:
            last = f"healthz {exc.code}: {_error_body(exc)}"
        except (URLError, OSError) as exc:
            last = f"judge unreachable at {url}: {exc}"
        if time.monotonic() >= deadline:
            return last
        time.sleep(2)


def request_evaluation(url: str, token: str, trajectory: dict, timeout: float) -> dict:
    body = json.dumps({"trajectory": trajectory}).encode()
    req = Request(f"{url}/evaluate", data=body, method="POST",
                  headers={"Content-Type": "application/json", "x-judge-token": token})
    try:
        with _opener.open(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except HTTPError as exc:
        return {"ok": False, "reason": f"judge answered {exc.code}: {_error_body(exc)}"}
    except (URLError, OSError, ValueError) as exc:
        return {"ok": False, "reason": f"evaluate request failed: {exc}"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--trajectory", required=True,
                    help="the trajectory tests/test.sh built from the agent log")
    ap.add_argument("--logs-dir", default="/logs/verifier",
                    help="where to leave judge_container.json (default: %(default)s)")
    a = ap.parse_args(argv)

    url = os.environ.get("JUDGE_URL", "http://judge:8770").rstrip("/")
    token = os.environ.get("JUDGE_TOKEN", "")
    logs = Path(a.logs_dir)
    logs.mkdir(parents=True, exist_ok=True)
    marker = logs / MARKER_NAME

    def finish(doc: dict) -> int:
        ok = bool(doc.get("ok"))
        marker.write_text(json.dumps({
            "ok": ok,
            "graded_in": doc.get("graded_in") if ok else None,
            "reason": doc.get("reason"),
            "url": url,
            "model": doc.get("model"),
            "codex_version": doc.get("codex_version"),
            "returncode": doc.get("returncode"),
            "rubric_criteria": doc.get("rubric_criteria"),
            "written": doc.get("written"),
        }, indent=2))
        if doc.get("log_tail"):
            print(doc["log_tail"].rstrip())
        if ok:
            print(f"[judge-client] graded in {doc.get('graded_in')} by {doc.get('model')}: "
                  f"{len(doc.get('written') or [])} report(s) in {logs}")
            return 0
        print(f"[judge-client] the judge container did not finish the evaluation: "
              f"{doc.get('reason')}", file=sys.stderr)
        return 1

    if not token:
        return finish({"ok": False, "reason": "JUDGE_TOKEN is not set in the verifier "
                                             "environment (task.toml [verifier.env])"})
    try:
        trajectory = json.loads(Path(a.trajectory).read_text())
    except (OSError, ValueError) as exc:
        return finish({"ok": False, "reason": f"cannot read {a.trajectory}: {exc}"})

    why = wait_until_healthy(url, float(os.environ.get("JUDGE_WAIT_SEC", "120")))
    if why:
        return finish({"ok": False, "reason": why})
    return finish(request_evaluation(url, token, trajectory,
                                     float(os.environ.get("JUDGE_REQUEST_TIMEOUT_SEC", "1200"))))


if __name__ == "__main__":
    sys.exit(main())
