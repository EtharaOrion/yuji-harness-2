"""Credentials: does the harness know a login WORKS, or only that a file exists?

Every check here used to answer the second question. A token string that had
expired, a codex login that had lapsed, a z.ai key that was rejected -- all three
printed "OK" at second zero and failed twenty minutes later, inside a paid trial,
looking like a bad agent rather than a bad credential.

The rule these pin: a refusal needs a POSITIVE fact. Expired is a fact. "Not
logged in" is a fact. "I could not run the check" is not, and must never block a
run that would have worked.
"""
from __future__ import annotations

import os
import re
import subprocess
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
RUN_TASK = REPO / "scripts" / "run_task.sh"


def _fn(name: str) -> str:
    """One function lifted out of run_task.sh, so a unit test can drive it."""
    src = RUN_TASK.read_text()
    m = re.search(r"^%s\(\) \{.*?^\}$" % re.escape(name), src, re.S | re.M)
    assert m, f"{name}() not found in run_task.sh"
    return m.group(0)


def _bash(body: str, **env) -> subprocess.CompletedProcess:
    e = dict(os.environ)
    e.update({k: str(v) for k, v in env.items()})
    return subprocess.run(["bash", "-c", body], capture_output=True, text=True,
                          env=e, timeout=90)


PRELUDE = 'AUTH_MIN_REMAINING_SEC="${AUTH_MIN_REMAINING_SEC:-900}"\nCLAUDE_TOKEN_EXPIRES_MS=""\n'


# --------------------------------------------------------------- claude token

def _expiry_probe(expires_ms) -> subprocess.CompletedProcess:
    return _bash(
        PRELUDE + _fn("check_claude_token_expiry")
        + f'\nCLAUDE_TOKEN_EXPIRES_MS="{expires_ms}"\n'
        'if check_claude_token_expiry; then echo ALLOW; else echo REFUSE; fi')


def test_an_expired_token_is_refused():
    """harbor forwards the token into a container that cannot refresh it, so an
    expired accessToken is every turn failing auth -- not a warning."""
    r = _expiry_probe(1_000_000_000_000)          # 2001
    assert "REFUSE" in r.stdout, r.stdout + r.stderr
    assert "EXPIRED" in r.stderr


def test_a_token_expiring_mid_run_warns_but_never_blocks():
    r = _expiry_probe(int((time.time() + 300) * 1000))
    assert "ALLOW" in r.stdout, r.stdout + r.stderr
    assert "expires in" in r.stderr


def test_a_healthy_token_says_nothing():
    r = _expiry_probe(int((time.time() + 86400) * 1000))
    assert "ALLOW" in r.stdout
    assert r.stderr.strip() == ""


@pytest.mark.parametrize("value", ["", "banana", "null", "-1x"])
def test_an_unreadable_expiry_is_no_opinion(value):
    """The field is absent on a hand-exported token and on older CLI stores.
    Treating "I cannot tell" as "expired" would refuse working credentials."""
    r = _expiry_probe(value)
    assert "ALLOW" in r.stdout, r.stdout + r.stderr


# ---------------------------------------------------------------- codex login

def _codex_probe(**env) -> subprocess.CompletedProcess:
    return _bash(_fn("check_codex_login")
                 + '\nif check_codex_login; then echo ALLOW; else echo REFUSE; fi', **env)


@pytest.mark.skipif(not __import__("shutil").which("codex"), reason="codex not installed")
def test_a_logged_out_codex_is_refused(tmp_path):
    """`codex login status` is the only thing that knows. auth.json being
    non-empty was what the harness checked, and an expired login passes that."""
    r = _codex_probe(CODEX_HOME=str(tmp_path))
    assert "REFUSE" in r.stdout, r.stdout + r.stderr
    assert "not logged in" in r.stderr.lower()


def test_a_codex_that_cannot_be_asked_does_not_block(tmp_path):
    """A CLI that is missing, slow or shouting about something else is unknown,
    not logged out. Refusing here would block every machine with an odd PATH."""
    r = _bash(_fn("check_codex_login")
              + '\nif check_codex_login; then echo ALLOW; else echo REFUSE; fi',
              PATH="/usr/bin:/bin")
    assert "ALLOW" in r.stdout, r.stdout + r.stderr
    assert "could not confirm" in r.stderr


# ------------------------------------------------------------------- zbridge

def _zbridge_probe(port: int) -> subprocess.CompletedProcess:
    return _bash(_fn("zbridge_live_check")
                 + '\nif zbridge_live_check; then echo ALLOW; else echo REFUSE; fi',
                 ZB_PORT=port)


def test_a_rejected_glm_key_is_refused(fake_upstream):
    """zbridge's own /healthz answers {"ok": true} whatever key it holds, so the
    only way to know is to send something through it."""
    r = _zbridge_probe(fake_upstream(401))
    assert "REFUSE" in r.stdout, r.stdout + r.stderr
    assert "z.ai rejected" in r.stderr


def test_a_working_glm_key_is_allowed(fake_upstream):
    r = _zbridge_probe(fake_upstream(200))
    assert "ALLOW" in r.stdout, r.stdout + r.stderr


@pytest.mark.parametrize("status", [500, 503, 429])
def test_an_upstream_wobble_warns_but_never_blocks(fake_upstream, status):
    """A 500 or a rate limit is not a statement about the credential. Blocking
    on one would fail runs that a retry would have completed."""
    r = _zbridge_probe(fake_upstream(status))
    assert "ALLOW" in r.stdout, r.stdout + r.stderr
    assert "could not verify" in r.stderr


def test_a_bridge_that_is_not_there_warns_but_never_blocks():
    r = _zbridge_probe(1)     # nothing listens on :1
    assert "ALLOW" in r.stdout, r.stdout + r.stderr
    assert "could not verify" in r.stderr


@pytest.fixture
def fake_upstream():
    """A throwaway HTTP server standing in for zbridge, answering one status."""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    servers = []

    def start(status: int) -> int:
        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                self.send_response(status)
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"{}")

            def log_message(self, *a):
                pass

        srv = HTTPServer(("127.0.0.1", 0), H)
        servers.append(srv)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        return srv.server_address[1]

    yield start
    for s in servers:
        s.shutdown()


# ------------------------------------------------- the key the script deleted

def test_an_exported_api_key_is_not_wiped_then_demanded(tmp_path):
    """run_task.sh clears ANTHROPIC_API_KEY at line 53 to stop a Claude Code
    session's own proxy reaching the container, then check_credentials told you
    to set ANTHROPIC_API_KEY. A tasker with only an API key could loop on that
    forever. resolve_auth hands the caller's value back instead."""
    body = (PRELUDE + _fn("resolve_auth")
            + '\nresolve_auth\necho "KEY=${ANTHROPIC_API_KEY:-<EMPTY>}"')
    env = {k: v for k, v in os.environ.items()
           if k not in ("CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_API_KEY")}
    env.update({"HOME": str(tmp_path),          # no ~/.claude/.credentials.json
                "PATH": "/usr/bin:/bin",        # no `security`, so no keychain
                "CALLER_ANTHROPIC_API_KEY": "sk-ant-caller"})
    r = subprocess.run(["bash", "-c", body], capture_output=True, text=True,
                       env=env, timeout=60)
    assert "KEY=sk-ant-caller" in r.stdout, r.stdout + r.stderr


def test_the_script_still_saves_the_key_before_clearing_it():
    """The save has to happen above the unset, or the restore above is dead."""
    src = RUN_TASK.read_text()
    save = src.index("CALLER_ANTHROPIC_API_KEY=")
    clear = src.index("unset ANTHROPIC_BASE_URL ANTHROPIC_API_KEY")
    assert save < clear


def test_no_credentials_at_all_warns_rather_than_crashing(tmp_path):
    body = (PRELUDE + _fn("resolve_auth") + '\nresolve_auth\necho "RC=$?"')
    env = {k: v for k, v in os.environ.items()
           if k not in ("CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_API_KEY")}
    env.update({"HOME": str(tmp_path), "PATH": "/usr/bin:/bin",
                "CALLER_ANTHROPIC_API_KEY": ""})
    r = subprocess.run(["bash", "-c", body], capture_output=True, text=True,
                       env=env, timeout=60)
    assert "RC=0" in r.stdout, r.stdout + r.stderr
    assert "no CLAUDE_CODE_OAUTH_TOKEN" in r.stderr


# ------------------------------------------------- what `codex login status` misses
#
# Measured on codex-cli in the judge image and on the host:
#
#   no auth.json          rc=1  "Not logged in"
#   {"tokens": null}      rc=0  "Logged in using ChatGPT"     <- wrong
#   {}                    rc=0  "Logged in using ChatGPT"     <- wrong
#   malformed id_token    rc=1  "Error checking login status: invalid ID token"
#
# So the CLI only really detects a MISSING file. codex_auth_is_empty reads the
# one fact it misses, from the file, with no network call.

def _empty_probe(text: str, tmp_path) -> bool:
    f = tmp_path / "auth.json"
    f.write_text(text)
    r = _bash(_fn("codex_auth_is_empty")
              + f'\nif codex_auth_is_empty "{f}"; then echo EMPTY; else echo HAS; fi')
    return "EMPTY" in r.stdout


@pytest.mark.parametrize("text", ['{}', '{"tokens": null}',
                                  '{"auth_mode": "chatgpt", "OPENAI_API_KEY": null, "tokens": null}'])
def test_a_credential_file_with_no_credential_is_caught(text, tmp_path):
    assert _empty_probe(text, tmp_path)


@pytest.mark.parametrize("text", [
    '{"tokens": {"access_token": "x"}}',
    '{"OPENAI_API_KEY": "sk-x", "tokens": null}',
])
def test_a_real_credential_is_not_called_empty(text, tmp_path):
    assert not _empty_probe(text, tmp_path)


@pytest.mark.parametrize("text", ['not json at all', '[]', '{"tokens": "a string"}'])
def test_a_shape_it_does_not_recognise_says_nothing(text, tmp_path):
    """Silence, not refusal. Guessing at an unfamiliar credential format would
    block runs whose login is perfectly good."""
    assert not _empty_probe(text, tmp_path)


def test_the_real_host_credential_passes():
    """The check has to agree with the machine it runs on, or it is useless."""
    import json
    auth = Path(os.environ.get("CODEX_AUTH_FILE",
                               Path.home() / ".codex" / "auth.json"))
    if not auth.is_file():
        pytest.skip("no codex login on this machine")
    r = _bash(_fn("codex_auth_is_empty")
              + f'\nif codex_auth_is_empty "{auth}"; then echo EMPTY; else echo HAS; fi')
    assert "HAS" in r.stdout, r.stdout + r.stderr
