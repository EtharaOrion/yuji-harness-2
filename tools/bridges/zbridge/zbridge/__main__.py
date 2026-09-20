"""CLI entry: python -m zbridge [--host --port --check]."""
from __future__ import annotations

import argparse
import logging
import sys

import uvicorn

from .bridge import Config, build_app
from .credentials import CredentialsError, EnvKeyProvider


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m zbridge")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8766)
    p.add_argument("--log-level", default="info")
    p.add_argument("--check", action="store_true", help="Print config summary and exit.")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = Config()
    provider = EnvKeyProvider()

    print(f"[zbridge] upstream = {cfg.upstream_url}")
    print(f"[zbridge] default_model = {cfg.default_model}")
    print(f"[zbridge] buffer_and_retry = {cfg.buffer_and_retry}")
    print(f"[zbridge] bridge_secret_set = {bool(cfg.bridge_secret)}")

    try:
        key = provider.get()
        print(f"[zbridge] api_key = {key[:6]}...{key[-4:]} ({len(key)} chars)")
    except CredentialsError as e:
        print(f"[zbridge] ERROR: {e}", file=sys.stderr)
        if args.check:
            return 2

    if args.check:
        print("[zbridge] check OK")
        return 0

    print(f"[zbridge] listening on http://{args.host}:{args.port}")
    print("[zbridge] point clients at:")
    print(f"[zbridge]     export ANTHROPIC_API_BASE=http://{args.host}:{args.port}")
    print("[zbridge]     export ANTHROPIC_API_KEY=$ZB_BRIDGE_SECRET")

    uvicorn.run(build_app(provider), host=args.host, port=args.port, log_level=args.log_level)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
