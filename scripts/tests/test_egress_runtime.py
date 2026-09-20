"""What the proxy DOES, not what its config says.

Every other test in this directory parses a file. test_egress_allowlist.py says
so in its own docstring -- "All of it is config parsing. Nothing here builds an
image or starts a container." That is the right trade for CI on every commit,
but it leaves a class of regression uncovered: squid takes the FIRST matching
http_access rule, so the allowlist's meaning depends on the order of three lines
that a parser sees as a set.

The specific hole this closes is recorded as prose in squid.conf. Without
`http_access deny CONNECT !SSL_ports`, `CONNECT api.anthropic.com:22` skips the
443 rule on port, falls through to the bare host rule, matches, and is tunnelled
-- observed once by hand as TCP_TUNNEL/503 rather than TCP_DENIED/403, then
fixed, then never tested. Reorder those lines today and every existing test
still passes.

This file starts the real proxy and reads its real access log. It also exercises
the capture path added for per-run proof of denial: the container is given an
/egress-out mount, and the log is read back from the HOST side of it, which is
the same path tools/network/egress-proxy/overlay.yaml uses under harbor.

Requires docker and a built `egress-proxy:latest` (`make build-egress-proxy`).
Skipped, never failed, when either is missing -- an unbuilt image is a laptop
without the tag, not a broken allowlist.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

# Overridable so the suite can be pointed at a deliberately-broken build to
# check that these assertions actually fail when the policy is wrong. A test
# that has never been seen to fail is not evidence.
IMAGE = os.environ.get("EGRESS_PROXY_IMAGE", "egress-proxy:latest")

# The allowed host, and a host that must never be. Kept literal rather than
# parsed out of squid.conf: a test that derives its expectation from the file
# under test cannot catch that file being wrong.
ALLOWED = "api.anthropic.com"
DENIED = "example.com"


def _docker_ok() -> bool:
    if shutil.which("docker") is None:
        return False
    return subprocess.run(["docker", "image", "inspect", IMAGE],
                          capture_output=True).returncode == 0


pytestmark = pytest.mark.skipif(
    not _docker_ok(),
    reason=f"docker or {IMAGE} unavailable (`make build-egress-proxy`)",
)


def _sh(*args: str, timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout)


@pytest.fixture(scope="module")
def proxy():
    """A running proxy plus the host side of its /egress-out mount.

    /tmp and not pytest's tmp_path: Docker Desktop shares /tmp by default and
    does not share the /private/var/folders path tmp_path hands out, so a bind
    mount from there comes up empty on macOS and the capture assertions fail for
    a reason that has nothing to do with the proxy.
    """
    out = Path(tempfile.mkdtemp(dir="/tmp", prefix="egress-out-"))
    run = _sh("docker", "run", "-d", "--rm",
              "-p", "0:3128",
              "-v", f"{out}:/egress-out",
              IMAGE)
    if run.returncode != 0:
        pytest.skip(f"could not start {IMAGE}: {run.stderr.strip()}")
    cid = run.stdout.strip()

    try:
        port = _sh("docker", "port", cid, "3128").stdout.strip()
        port = port.splitlines()[0].rsplit(":", 1)[1]

        # squid is listening when it answers, not when docker says "started".
        # Poll rather than sleep: a fixed sleep is either slow or flaky.
        deadline = time.time() + 30
        while time.time() < deadline:
            probe = _sh("curl", "-sS", "-o", "/dev/null", "-w", "%{http_code}",
                        "--max-time", "3", "-x", f"http://127.0.0.1:{port}",
                        f"http://{DENIED}/", timeout=10)
            if probe.stdout.strip() == "403":
                break
            time.sleep(0.5)
        else:
            pytest.fail(f"proxy never answered on :{port}\n"
                        f"{_sh('docker', 'logs', cid).stderr}")

        yield {"cid": cid, "port": port, "out": out}
    finally:
        _sh("docker", "rm", "-f", cid)
        shutil.rmtree(out, ignore_errors=True)


def _request(proxy, url: str) -> None:
    """Drive one request through the proxy. The outcome is irrelevant -- the
    assertion is on what squid LOGGED, which is recorded either way."""
    _sh("curl", "-sS", "-o", "/dev/null", "--max-time", "10",
        "-x", f"http://127.0.0.1:{proxy['port']}", url, timeout=20)


def _log(proxy) -> str:
    """The access log, read from the host side of the /egress-out bind mount.

    tail -F | tee is not instantaneous, so poll briefly. Falls back to nothing
    rather than hanging; callers assert on content and will say what is missing.
    """
    path = proxy["out"] / "egress-access.log"
    deadline = time.time() + 10
    while time.time() < deadline:
        if path.is_file() and path.read_text().strip():
            time.sleep(0.3)          # let the last line land
            return path.read_text()
        time.sleep(0.3)
    return path.read_text() if path.is_file() else ""


def _entries(log: str, host: str) -> list[list[str]]:
    return [f for f in (l.split() for l in log.splitlines())
            if len(f) >= 7 and host in f[6]]


# --------------------------------------------------------------- the capture


def test_access_log_reaches_the_host_mount(proxy):
    """The per-run capture path itself.

    Without this the audit gets an empty file and reports "no denials", which is
    indistinguishable from a clean run -- the exact confusion the capture was
    added to remove.
    """
    _request(proxy, f"http://{DENIED}/")
    log = _log(proxy)
    assert log.strip(), (
        f"{proxy['out']}/egress-access.log is empty or absent -- the tee in "
        f"entrypoint.sh is not reaching the /egress-out mount"
    )


# ------------------------------------------------------- the allowlist itself


def test_allowed_host_is_not_denied(proxy):
    """api.anthropic.com must get through on 443.

    Asserted as "not denied" rather than "succeeded": whether the TCP_TUNNEL
    completes depends on the test host having internet, which is not what this
    file is about.
    """
    _request(proxy, f"https://{ALLOWED}/")
    entries = _entries(_log(proxy), ALLOWED)
    assert entries, f"no access.log entry for {ALLOWED}"
    assert not any("DENIED" in f[3] for f in entries), (
        f"the allowed host was denied: {[f[3] for f in entries]}"
    )


def test_unlisted_host_is_denied_over_connect(proxy):
    _request(proxy, f"https://{DENIED}/")
    entries = _entries(_log(proxy), DENIED)
    connects = [f for f in entries if f[5] == "CONNECT"]
    assert connects, f"no CONNECT entry for {DENIED}"
    assert all("DENIED" in f[3] for f in connects), (
        f"an unlisted host was tunnelled: {[f[3] for f in connects]}"
    )


def test_unlisted_host_is_denied_over_plain_http(proxy):
    _request(proxy, f"http://{DENIED}/")
    entries = _entries(_log(proxy), DENIED)
    gets = [f for f in entries if f[5] == "GET"]
    assert gets, f"no GET entry for {DENIED}"
    assert all("DENIED" in f[3] for f in gets), (
        f"an unlisted host was fetched: {[f[3] for f in gets]}"
    )


# ------------------------------------------------------- the ordering property


def test_allowed_host_is_denied_on_a_non_ssl_port(proxy):
    """The regression squid.conf documents in prose and nothing tested.

    `http_access allow CONNECT SSL_ports allowed_hosts` looks like it pins
    CONNECT to 443, but squid takes the first matching rule and the later
    `http_access allow allowed_hosts` names no method and no port. Without
    `http_access deny CONNECT !SSL_ports` ahead of both, this request skips the
    443 rule, matches the bare host rule, and is tunnelled -- one host wide, but
    not what the allowlist claims to do.
    """
    _request(proxy, f"https://{ALLOWED}:22/")
    entries = [f for f in _entries(_log(proxy), ALLOWED)
               if f[5] == "CONNECT" and f[6].endswith(":22")]
    denied = [f for f in entries if "DENIED" in f[3]]

    # Assert on the presence of a DENIAL, not on the absence of a tunnel.
    # Verified against a build with the guard removed: squid ACCEPTS the CONNECT
    # and logs nothing at all until the transaction ends, so the regression
    # shows up as an empty log rather than as a TCP_TUNNEL line. Asserting
    # `all(... DENIED ...)` over an empty list would pass vacuously; asserting
    # the denial exists catches both shapes.
    assert denied, (
        f"CONNECT {ALLOWED}:22 was not denied -- logged {[f[3] for f in entries] or 'nothing'}. "
        f"An empty result here means squid accepted the tunnel and is still "
        f"holding it open. Check that `http_access deny CONNECT !SSL_ports` "
        f"still precedes the allow rules in tools/network/egress-proxy/squid.conf."
    )


# ------------------------------------------- the log the auditor has to parse


def test_auditor_reads_this_log(proxy):
    """End of the chain: squid's real output through the real parser.

    scan_access_log's field offsets are pinned to squid's native format. A
    format change would otherwise surface as an audit that silently finds
    nothing, months later, on a run that mattered.
    """
    import sys
    sys.path.insert(0, str(REPO / "tools" / "network"))
    import detect_internet_use as diu

    diu.FINDINGS.clear()
    _request(proxy, f"https://{DENIED}/")
    log_path = proxy["out"] / "egress-access.log"
    _log(proxy)

    attempts = diu.scan_access_log(log_path)
    assert attempts, "the auditor parsed nothing out of a non-empty access log"
    assert any(r["host"] == DENIED and r["denied"] for r in attempts), (
        f"the auditor did not see the denial of {DENIED}: "
        f"{json.dumps(attempts[:5], indent=2)}"
    )
    assert any(f["kind"] == "proxy-denied" for f in diu.FINDINGS), (
        "a denied host produced no blocking finding"
    )
    diu.FINDINGS.clear()
