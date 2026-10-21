#!/usr/bin/env python3
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
    ap = argparse.ArgumentParser(
        description="Run the Bedrock Anthropic-wire judge proxy for one task."
    )
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--config-out", required=True)
    args = ap.parse_args()

    try:
        from bedrock_bridge import BedrockJudgeServer
        from bedrock_bridge.server import BedrockBridgeError
    except ImportError as exc:
        print(f"[bedrock_bridge_runner] import failed: {exc}", file=sys.stderr)
        return 1

    judge_model = os.environ.get("JUDGE_MODEL", "").strip() or os.environ.get("MODEL", "").strip()
    if not judge_model:
        print("[bedrock_bridge_runner] JUDGE_MODEL (or MODEL) must be set", file=sys.stderr)
        return 1
    region = (os.environ.get("BEDROCK_AWS_REGION") or os.environ.get("AWS_REGION")
              or os.environ.get("AWS_DEFAULT_REGION") or "").strip()
    if not region:
        print("[bedrock_bridge_runner] BEDROCK_AWS_REGION / AWS_REGION must be set", file=sys.stderr)
        return 1

    try:
        server = BedrockJudgeServer(pinned_model=judge_model, host=args.host,
                                    port=args.port or _find_free_port(args.host),
                                    aws_region=region).start()
    except BedrockBridgeError as exc:
        print(f"[bedrock_bridge_runner] start failed: {exc}", file=sys.stderr)
        return 1

    config = {
        "host": args.host,
        "port": server.port,
        "api_key": server.api_key,
        "judge_model": judge_model,
        "aws_region": region,
    }
    out = Path(args.config_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(config, indent=2) + "\n")
    tmp.replace(out)
    print(f"[bedrock_bridge_runner] serving on {args.host}:{server.port}", file=sys.stderr)
    print(f"[bedrock_bridge_runner] judge model = {judge_model} (region {region})",
          file=sys.stderr)

    stopping = False

    def _stop(signum, _frame):
        nonlocal stopping
        stopping = True
        print(f"[bedrock_bridge_runner] signal {signum} -- stopping", file=sys.stderr)

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    try:
        while not stopping:
            time.sleep(1.0)
    finally:
        try:
            server.stop()
        except Exception as exc:  # noqa: BLE001
            print(f"[bedrock_bridge_runner] stop() raised: {exc}", file=sys.stderr)
        try:
            out.unlink(missing_ok=True)
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
