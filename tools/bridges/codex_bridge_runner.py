#!/usr/bin/env python3
"""Launch the codex-login Anthropic judge bridge for one run.

Same CLI contract as kiro_bridge_runner.py so callers (scripts/run_task.sh,
scripts/regrade_output.sh) can swap one for the other. Each Messages request
becomes one `codex exec` run against the ChatGPT subscription the user logged
in with (`codex login`); the login itself never leaves the host.

Backed by the sibling `codex_bridge` package. The sys.path shim lets this
script import that package straight from the checkout, so the harness does
not depend on a pip-install for the runner's interpreter.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


def _find_free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


def _call_logger(path: str):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()

    def record(rec) -> None:
        line = json.dumps({
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "label": rec.label,
            "requested": rec.requested,
            "served": rec.served,
            "ok": rec.ok,
            "error": rec.error,
            "usage": rec.usage,
        })
        with lock, out.open("a") as fh:
            fh.write(line + "\n")

    return record


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the codex-login Anthropic judge bridge.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=0, help="bind port (0 = pick a free one)")
    ap.add_argument("--config-out", required=True, help="write port + keys here as JSON")
    ap.add_argument("--calls-out", default="", help="append one JSON line per served call")
    args = ap.parse_args()

    try:
        from codex_bridge import AnthropicServer, CodexConfig, credential_problem, shared_bridge
    except ImportError as exc:
        print(f"[codex_bridge_runner] import failed: {exc}", file=sys.stderr)
        return 1

    config_obj = CodexConfig.from_env(os.environ)
    problem = credential_problem(config_obj)
    if problem:
        print(f"[codex_bridge_runner] codex not usable: {problem}", file=sys.stderr)
        return 1

    judge_model = config_obj.judge_model
    port = args.port or _find_free_port(args.host)

    server = AnthropicServer(shared_bridge(config_obj), host=args.host, port=port,
                             on_call=_call_logger(args.calls_out) if args.calls_out else None)
    agent_key = server.add_key("agent")
    judge_key = server.add_key("judge", judge_model)
    server.start()

    config = {
        "host": args.host,
        "port": port,
        "agent_key": agent_key,
        "judge_key": judge_key,
        "agent_model": "(request-named)",
        "judge_model": judge_model,
    }
    out = Path(args.config_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(config, indent=2) + "\n")
    tmp.replace(out)
    print(f"[codex_bridge_runner] serving on {args.host}:{port}", file=sys.stderr)
    print(f"[codex_bridge_runner] judge model = {judge_model}", file=sys.stderr)
    print(f"[codex_bridge_runner] concurrency = {config_obj.concurrency}", file=sys.stderr)

    stopping = False

    def _stop(signum, _frame):
        nonlocal stopping
        stopping = True
        print(f"[codex_bridge_runner] signal {signum} -- stopping", file=sys.stderr)

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    try:
        while not stopping:
            time.sleep(1.0)
    finally:
        try:
            server.stop()
        except Exception as exc:  # noqa: BLE001
            print(f"[codex_bridge_runner] stop() raised: {exc}", file=sys.stderr)
        try:
            out.unlink(missing_ok=True)
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
