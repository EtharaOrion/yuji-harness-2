from __future__ import annotations

import argparse
import json
import socket
import sys
import time

from codex_bridge.bridge import credential_problem, doctor, shared_bridge
from codex_bridge.config import CodexConfig


def _find_free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


def _serve(args: argparse.Namespace) -> int:
    from codex_bridge.anthropic_server import AnthropicServer

    problem = credential_problem()
    if problem:
        print(f"codex-bridge: {problem}", file=sys.stderr)
        return 1
    config = CodexConfig.from_env()
    port = args.port or _find_free_port(args.host)
    server = AnthropicServer(shared_bridge(config), host=args.host, port=port)
    agent_key = server.add_key("agent")
    judge_model = args.pin_model or config.judge_model
    judge_key = server.add_key("judge", judge_model)
    server.start()
    print(json.dumps({
        "host": args.host,
        "port": port,
        "agent_key": agent_key,
        "judge_key": judge_key,
        "judge_model": judge_model,
    }, indent=2))
    sys.stdout.flush()
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("", file=sys.stderr)
    finally:
        server.stop()
    return 0


def _doctor(args: argparse.Namespace) -> int:
    report = doctor()
    if args.json:
        print(json.dumps({
            "ok": report.ok,
            "cli_present": report.cli_present,
            "judge_model": report.judge_model,
            "cli_path": report.cli_path,
            "error": report.error,
        }))
    else:
        print(f"ok: {report.ok}")
        print(f"cli_present: {report.cli_present}")
        if report.cli_path:
            print(f"cli_path: {report.cli_path}")
        if report.judge_model:
            print(f"judge_model: {report.judge_model}")
        if report.error:
            print(f"error: {report.error}", file=sys.stderr)
    return 0 if report.ok else 1


def _models(args: argparse.Namespace) -> int:
    bridge = shared_bridge()
    for m in bridge.models():
        print(m["id"])
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="codex-bridge",
        description="Anthropic-shaped judge endpoint on a codex login.",
    )
    sub = ap.add_subparsers(dest="command", required=True)

    s = sub.add_parser("serve", help="Start the HTTP endpoint (prints the keys as JSON).")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=0, help="0 = pick a free port")
    s.add_argument("--pin-model", default="",
                   help="judge model; overrides CODEX_JUDGE_MODEL")
    s.set_defaults(func=_serve)

    d = sub.add_parser("doctor", help="Check that `codex login status` passes.")
    d.add_argument("--json", action="store_true")
    d.set_defaults(func=_doctor)

    m = sub.add_parser("models", help="Print the judge model id.")
    m.set_defaults(func=_models)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
