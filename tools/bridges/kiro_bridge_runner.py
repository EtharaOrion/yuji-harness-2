#!/usr/bin/env python3
"""Launch the Kiro Anthropic bridge for one run, mint agent + judge keys.

Called from scripts/run_task.sh when CC_MODE=kiro. Mirrors the role that
start_zbridge.sh plays for GLM: start a local Anthropic-compatible endpoint on
a free port, mint credentials for the agent (serves whatever model the request
names) and for the verifier judge (pinned to KIRO_JUDGE_MODEL, so a task's
rubric_judge_cli.py gets a fixed model regardless of what judge.toml asks for).

The kiro-cli login never leaves the host. The container only ever holds the
minted keys, which are valid for the lifetime of this process.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import sys
import time
from pathlib import Path


def _find_free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the Kiro Anthropic bridge for one task.")
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind address (default 127.0.0.1; use the docker bridge gateway when the agent runs under a sidecar)")
    ap.add_argument("--port", type=int, default=0, help="bind port (0 = pick a free one)")
    ap.add_argument("--config-out", required=True,
                    help="write the chosen port + keys here as JSON; shell reads it then signals ready")
    args = ap.parse_args()

    try:
        from kiro_bridge import AnthropicServer, credential_problem, shared_bridge
    except ImportError as exc:
        print(f"[kiro_bridge_runner] import failed: {exc}", file=sys.stderr)
        return 1

    problem = credential_problem()
    if problem:
        print(f"[kiro_bridge_runner] kiro-cli not usable: {problem}", file=sys.stderr)
        return 1

    judge_model = os.environ.get("KIRO_JUDGE_MODEL", "").strip() or None
    agent_model = os.environ.get("KIRO_MODEL", "").strip() or None
    port = args.port or _find_free_port(args.host)

    server = AnthropicServer(shared_bridge(), host=args.host, port=port)
    agent_key = server.add_key("agent")
    judge_key = server.add_key("judge", judge_model)
    server.start()

    config = {
        "host": args.host,
        "port": port,
        "agent_key": agent_key,
        "judge_key": judge_key,
        "agent_model": agent_model or "(request-named)",
        "judge_model": judge_model or "(request-named)",
    }
    out = Path(args.config_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(config, indent=2) + "\n")
    tmp.replace(out)
    print(f"[kiro_bridge_runner] serving on {args.host}:{port}", file=sys.stderr)
    print(f"[kiro_bridge_runner] agent model = {config['agent_model']}", file=sys.stderr)
    print(f"[kiro_bridge_runner] judge model = {config['judge_model']}", file=sys.stderr)

    stopping = False

    def _stop(signum, _frame):
        nonlocal stopping
        stopping = True
        print(f"[kiro_bridge_runner] signal {signum} -- stopping", file=sys.stderr)

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    try:
        while not stopping:
            time.sleep(1.0)
    finally:
        try:
            server.stop()
        except Exception as exc:  # noqa: BLE001
            print(f"[kiro_bridge_runner] stop() raised: {exc}", file=sys.stderr)
        try:
            out.unlink(missing_ok=True)
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
