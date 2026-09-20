#!/usr/bin/env python3
"""
claude_account.py — resolve the Claude account behind a run.

Reads the Claude Code OAuth credentials from the local credential store, calls
the profile endpoint, and returns the identity fields the finance pipeline
needs. tools/finance/finance_reporter.py calls get_claude_account_info() once per run.

    from tools.finance.claude_account import get_claude_account_info
    info = get_claude_account_info()
    info["subscription_id"]        # == organization_uuid

Credential sources, in priority order:
  1. CLAUDE_CODE_OAUTH_TOKEN env var  (run_task.sh exports this)
  2. CLAUDE_CODE_CREDENTIALS env var  (inline JSON, for CI)
  3. ~/.claude/.credentials.json      (Linux, and macOS CLI fallback)
  4. macOS Keychain service "Claude Code-credentials"

Environment:
  CLAUDE_PROFILE_URL     override the endpoint
  CLAUDE_PROFILE_JSON    inline profile JSON; skips the network (tests)
  CLAUDE_ACCOUNT_CACHE_TTL   cache lifetime in seconds (default 86400)

Every failure is soft: get_claude_account_info() returns a dict whose values are
None plus an "error" key, so a profile outage can never block usage reporting.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PROFILE_URL = os.environ.get("CLAUDE_PROFILE_URL",
                             "https://api.anthropic.com/api/oauth/profile")
KEYCHAIN_SERVICE = "Claude Code-credentials"
CREDENTIALS_FILE = Path.home() / ".claude" / ".credentials.json"
CACHE_PATH = Path.home() / ".cache" / "mcp-atlas" / "claude_account.json"
DEFAULT_TTL = 86400
TIMEOUT_SEC = 15

# Client identity the Claude CLI presents upstream; the profile endpoint is
# part of the OAuth surface and expects these.
ANTHROPIC_VERSION = "2023-06-01"
OAUTH_BETA = "oauth-2025-04-20"
CLI_USER_AGENT = "claude-cli/1.0.60 (external, cli)"

UUID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
                     r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")

EMPTY = {
    "account_uuid": None,
    "name": None,
    "email": None,
    "organization_uuid": None,
    "subscription_id": None,
    "subscription_status": None,
    "rate_limit_tier": None,
}


# ------------------------------------------------------------------ credentials
def _token_from_payload(raw: str) -> str | None:
    try:
        return (json.loads(raw).get("claudeAiOauth") or {}).get("accessToken") or None
    except Exception:
        return None


def _read_keychain() -> str | None:
    if sys.platform != "darwin":
        return None
    try:
        out = subprocess.run(
            ["security", "find-generic-password", "-s", KEYCHAIN_SERVICE, "-w"],
            capture_output=True, text=True, timeout=10)
        return _token_from_payload(out.stdout) if out.returncode == 0 else None
    except Exception:
        return None


def resolve_token() -> str | None:
    """Locate an OAuth access token. Never logs or returns it anywhere else."""
    token = (os.environ.get("CLAUDE_CODE_OAUTH_TOKEN") or "").strip()
    if token:
        return token

    inline = os.environ.get("CLAUDE_CODE_CREDENTIALS")
    if inline and (tok := _token_from_payload(inline)):
        return tok

    if CREDENTIALS_FILE.is_file():
        try:
            if tok := _token_from_payload(CREDENTIALS_FILE.read_text()):
                return tok
        except OSError:
            pass

    return _read_keychain()


# ---------------------------------------------------------------------- profile
def _cache_read(ttl: int) -> dict | None:
    try:
        if time.time() - CACHE_PATH.stat().st_mtime > ttl:
            return None
        return json.loads(CACHE_PATH.read_text())
    except Exception:
        return None


def _cache_write(profile: dict) -> None:
    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(json.dumps(profile))
        CACHE_PATH.chmod(0o600)
    except Exception:
        pass


def fetch_profile(refresh: bool = False) -> tuple[dict | None, str]:
    """Return (profile_json, note). profile_json is None when unavailable."""
    inline = os.environ.get("CLAUDE_PROFILE_JSON")
    if inline:
        try:
            return json.loads(inline), "from CLAUDE_PROFILE_JSON"
        except Exception as exc:
            return None, f"CLAUDE_PROFILE_JSON is not valid JSON: {exc}"

    ttl = int(os.environ.get("CLAUDE_ACCOUNT_CACHE_TTL") or DEFAULT_TTL)
    if not refresh and (cached := _cache_read(ttl)) is not None:
        return cached, "from cache"

    token = resolve_token()
    if not token:
        return None, "no Claude credentials found"

    req = urllib.request.Request(PROFILE_URL, method="GET", headers={
        "Authorization": f"Bearer {token}",
        "anthropic-version": ANTHROPIC_VERSION,
        "anthropic-beta": OAUTH_BETA,
        "user-agent": CLI_USER_AGENT,
        "Accept": "application/json",
    })
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_SEC) as resp:
            profile = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return None, f"HTTP {exc.code} from profile endpoint"
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"

    if not isinstance(profile, dict):
        return None, f"unexpected response type: {type(profile).__name__}"
    _cache_write(profile)
    return profile, "fetched"


# --------------------------------------------------------------------- extract
def _walk(obj, path=()):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk(v, path + (str(k),))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk(v, path + (str(i),))
    else:
        yield path, obj


def _find_uuid(profile: dict, *needles: str) -> str | None:
    """Locate a UUID by searching key paths, so a shape change can't break us."""
    hits = []
    for path, val in _walk(profile):
        if isinstance(val, str) and UUID_RE.match(val):
            joined = "_".join(path).lower()
            if any(n in joined for n in needles):
                hits.append((len(path), val))
    return min(hits)[1] if hits else None


def _first(profile: dict, section: str, *keys):
    node = profile.get(section)
    if isinstance(node, dict):
        for k in keys:
            if node.get(k):
                return node[k]
    return None


_cached_info: dict | None = None


def get_claude_account_info(refresh: bool = False) -> dict:
    """Account identity for the finance payload. Cached per process.

    Always returns the full key set; unavailable fields are None and an "error"
    key explains why, so callers never need to guard against a missing key.
    """
    global _cached_info
    if _cached_info is not None and not refresh:
        return _cached_info

    profile, note = fetch_profile(refresh=refresh)
    if profile is None:
        _cached_info = {**EMPTY, "error": note, "source": "unavailable"}
        return _cached_info

    org_uuid = _find_uuid(profile, "organization", "org")
    info = {
        "account_uuid": _find_uuid(profile, "account", "user"),
        "name": _first(profile, "account", "full_name", "display_name"),
        "email": _first(profile, "account", "email", "email_address"),
        "organization_uuid": org_uuid,
        # The profile exposes no subscription ID string; the organization is the
        # entity Anthropic bills (billing_type "stripe_subscription").
        "subscription_id": org_uuid,
        "subscription_status": _first(profile, "organization", "subscription_status"),
        "rate_limit_tier": _first(profile, "organization", "rate_limit_tier"),
        "source": note,
    }
    _cached_info = info
    return info


def _redact(obj):
    """Structure with credentials stripped and PII masked; UUIDs preserved."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            low = str(k).lower()
            if any(h in low for h in ("token", "secret", "key", "password")):
                out[k] = "<redacted>"
            elif isinstance(v, str) and any(h in low for h in ("email", "name", "phone")) \
                    and not UUID_RE.match(v):
                out[k] = f"<{low}: {len(v)} chars>"
            else:
                out[k] = _redact(v)
        return out
    if isinstance(obj, list):
        return [_redact(v) for v in obj]
    return obj


def main() -> int:
    args = set(sys.argv[1:])
    info = get_claude_account_info(refresh="--refresh" in args)
    if "--raw" in args:
        profile, _ = fetch_profile()
        print(json.dumps(_redact(profile or {}), indent=2))
        return 0
    safe = dict(info)
    for k in ("name", "email"):
        if safe.get(k):
            safe[k] = f"<{k}: {len(safe[k])} chars>"
    print(json.dumps(safe, indent=2))
    return 0 if info.get("account_uuid") or info.get("organization_uuid") else 1


if __name__ == "__main__":
    sys.exit(main())
