"""The proxy's allowlist and the wiring that forces traffic through it.

test_network_isolation.py asserts the *topology* — that main sits on an
internal network and the proxy is the only way out. This file asserts what
happens once traffic arrives at that proxy, which is the part that decides
whether the block is real:

  1. the allowlist is exactly one host, and `deny all` is the last word on it;
  2. main is pointed at the proxy and waits for it to be healthy;
  3. the two files that independently decide "internal vs internet" —
     overlay.yaml's NO_PROXY and detect_internet_use.py's INTERNAL_HOSTS —
     still agree, since squid.conf's comment makes that a standing pact.

All of it is config parsing. Nothing here builds an image or starts a
container, so it runs in CI on every commit rather than only before a release.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
PROXY_DIR = REPO / "tools" / "network" / "egress-proxy"
SQUID_CONF = PROXY_DIR / "squid.conf"
OVERLAY = PROXY_DIR / "overlay.yaml"
DOCKERFILE = PROXY_DIR / "Dockerfile"
ENTRYPOINT = PROXY_DIR / "entrypoint.sh"
DETECTOR = REPO / "tools" / "network" / "detect_internet_use.py"
# INTERNAL_HOSTS moved here when the rules became shared between the audit
# and the in-container PreToolUse hook; detect_internet_use.py imports them.
RULES = REPO / "tools" / "network" / "egress_rules.py"

# The whole point of the sidecar. Widening this set is a deliberate act and
# should have to edit a test that says so out loud.
EXPECTED_ALLOWLIST = {"api.anthropic.com"}


def _directives(text: str) -> list[str]:
    """squid.conf lines with comments and blanks stripped."""
    out = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            out.append(line)
    return out


@pytest.fixture(scope="module")
def squid_lines() -> list[str]:
    assert SQUID_CONF.is_file(), f"squid.conf missing at {SQUID_CONF}"
    return _directives(SQUID_CONF.read_text())


@pytest.fixture(scope="module")
def overlay() -> dict:
    assert OVERLAY.is_file(), f"overlay.yaml missing at {OVERLAY}"
    return yaml.safe_load(OVERLAY.read_text())


@pytest.fixture(scope="module")
def internal_hosts() -> set[str]:
    """INTERNAL_HOSTS read from source, not imported.

    egress_rules.py is also a CLI (it runs as the PreToolUse hook); parsing the
    literal keeps this test from depending on whether importing it has side
    effects, and from needing the sys.path juggling that import would want.
    """
    tree = ast.parse(RULES.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "INTERNAL_HOSTS":
            return set(ast.literal_eval(node.value))
    pytest.fail(f"INTERNAL_HOSTS not found in {RULES}")


# --------------------------------------------------------------- allowlist


def test_allowlist_is_exactly_the_documented_host(squid_lines):
    """One host, named explicitly. A diff here is a policy change."""
    allowed: set[str] = set()
    for line in squid_lines:
        m = re.match(r"acl\s+\S+\s+dstdomain\s+(.+)$", line)
        if m:
            allowed.update(m.group(1).split())
    assert allowed == EXPECTED_ALLOWLIST, (
        f"allowlist drifted: {sorted(allowed)} != {sorted(EXPECTED_ALLOWLIST)}. "
        "Widening it silently turns a closed-world task into an open one."
    )


def test_allowlist_entries_are_exact_hosts_not_subdomain_wildcards(squid_lines):
    """`.anthropic.com` matches every subdomain; `api.anthropic.com` does not.

    squid's dstdomain treats a leading dot as "this domain and all children",
    so the dot is the difference between one endpoint and an entire estate.
    """
    for line in squid_lines:
        m = re.match(r"acl\s+\S+\s+dstdomain\s+(.+)$", line)
        if not m:
            continue
        for entry in m.group(1).split():
            assert not entry.startswith("."), (
                f"dstdomain '{entry}' is a subdomain wildcard; name the exact host instead"
            )


def test_default_is_deny_and_it_is_the_last_word(squid_lines):
    """squid takes the first matching rule, so a later allow would never fire —
    but an allow inserted ABOVE deny all would. Pin the ordering."""
    access = [l for l in squid_lines if l.startswith("http_access")]
    assert access, "no http_access rules at all — squid would fall back to its built-in policy"
    assert access[-1] == "http_access deny all", (
        f"last http_access rule is '{access[-1]}', not 'http_access deny all'"
    )


def test_nothing_allows_all(squid_lines):
    """The failure mode the config comment warns about, asserted."""
    for line in squid_lines:
        assert not re.match(r"http_access\s+allow\s+all\b", line), (
            "http_access allow all defeats the entire sidecar"
        )


def test_every_allow_rule_is_qualified_by_the_allowlist(squid_lines):
    """Including CONNECT. A bare `http_access allow CONNECT` would tunnel
    anywhere on 443, which is every host that matters."""
    for line in squid_lines:
        if re.match(r"http_access\s+allow\b", line):
            assert "allowed_hosts" in line, (
                f"allow rule not qualified by the allowlist: '{line}'"
            )


def test_glm_config_is_opus_config_plus_zbridge_only(squid_lines):
    """GLM runs load squid-zbridge.conf. Without its zbridge lines it must be
    squid.conf exactly, so the two cannot drift apart."""
    zbridge_only = [
        "acl zbridge_host dstdomain host.docker.internal",
        "acl zbridge_port port 8766",
        "http_access allow zbridge_host zbridge_port",
    ]
    glm = _directives((PROXY_DIR / "squid-zbridge.conf").read_text())
    assert [l for l in glm if l not in zbridge_only] == squid_lines
    assert all(l in glm for l in zbridge_only), glm
    assert glm.index(zbridge_only[-1]) < glm.index("http_access deny all")


# ------------------------------------------------- operational invariants
# Each of these is called load-bearing in squid.conf's own comments; a proxy
# that dies at startup is indistinguishable from a network outage inside main.


def test_access_log_is_a_file_not_dev_stdout(squid_lines):
    log = [l for l in squid_lines if l.startswith("access_log")]
    assert log, "no access_log — a denial would be invisible to the operator"
    assert "/dev/stdout" not in log[0], (
        "squid drops to the 'proxy' user and cannot open /dev/stdout; it exits FATAL"
    )


def test_pid_filename_is_configured(squid_lines):
    """The healthcheck is `squid -k check`, which needs this to find the
    running instance. Without it the proxy never reports healthy and main's
    depends_on blocks forever."""
    assert any(l.startswith("pid_filename") for l in squid_lines)


def test_caching_is_off(squid_lines):
    """A replayed response is not a reproduction of the run."""
    assert "cache deny all" in squid_lines


def test_client_address_is_not_leaked_upstream(squid_lines):
    assert "forwarded_for delete" in squid_lines


def test_proxy_listens_on_the_port_the_overlay_points_at(squid_lines, overlay):
    ports = [l for l in squid_lines if l.startswith("http_port")]
    assert ports, "no http_port"
    port = ports[0].split()[1]
    env = overlay["services"]["main"]["environment"]
    assert env["HTTPS_PROXY"].endswith(f":{port}"), (
        f"overlay points main at {env['HTTPS_PROXY']} but squid listens on {port}"
    )


# ------------------------------------------------------------ overlay wiring


def test_all_four_proxy_vars_are_set(overlay):
    """Tools split on case: curl reads lowercase, most SDKs read uppercase.
    Setting only one pair leaves a hole for whichever half is missed."""
    env = overlay["services"]["main"]["environment"]
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        assert env.get(var) == "http://egress-proxy:3128", f"{var} not pointed at the proxy"


def test_main_waits_for_the_proxy_to_be_healthy(overlay):
    """Otherwise the first model call races squid's startup and fails with
    connection refused, which reads like an outage rather than a config bug."""
    dep = overlay["services"]["main"]["depends_on"]["egress-proxy"]
    assert dep["condition"] == "service_healthy"


def test_proxy_is_prebuilt_never_built_per_run(overlay):
    """`build:` here would rebuild the image inside every task project."""
    proxy = overlay["services"]["egress-proxy"]
    assert "image" in proxy, "egress-proxy has no image:"
    assert "build" not in proxy, "egress-proxy must not build per run"


def test_proxy_is_the_only_service_spanning_both_networks(overlay):
    assert overlay["networks"]["default"]["internal"] is True
    assert "egress" in overlay["networks"]
    assert set(overlay["services"]["egress-proxy"]["networks"]) == {"default", "egress"}
    assert "networks" not in overlay["services"]["main"], (
        "main must inherit the internal default, not name a network of its own"
    )


# ------------------------------------------- the cross-file consistency pact
# squid.conf: "Change one, look at the other."


def test_no_proxy_is_set_in_both_cases_identically(overlay):
    env = overlay["services"]["main"]["environment"]
    assert env["NO_PROXY"] == env["no_proxy"], "NO_PROXY and no_proxy disagree"


def test_sidecar_traffic_bypasses_the_proxy(overlay):
    """Routing bridge-local traffic through squid would fail the allowlist and
    take every MCP tool down with it."""
    no_proxy = {h.strip() for h in overlay["services"]["main"]["environment"]["NO_PROXY"].split(",")}
    for host in ("light-servers", "main", "localhost", "127.0.0.1", "::1"):
        assert host in no_proxy, f"{host} would be routed through squid and denied"


def test_the_proxy_itself_is_not_proxied(overlay):
    no_proxy = {h.strip() for h in overlay["services"]["main"]["environment"]["NO_PROXY"].split(",")}
    assert "egress-proxy" in no_proxy, "the proxy would be asked to proxy itself"


def test_bridge_local_hosts_bypass_the_proxy(overlay, internal_hosts):
    """The hosts that actually live on the compose bridge must agree in both
    files: the auditor must not flag them, and NO_PROXY must not route them
    through squid (which would deny them and take every MCP tool down)."""
    no_proxy = {h.strip() for h in overlay["services"]["main"]["environment"]["NO_PROXY"].split(",")}
    bridge_local = {"light-servers", "main", "localhost", "127.0.0.1", "::1"}
    assert bridge_local <= internal_hosts, (
        f"auditor would flag bridge-local traffic: {sorted(bridge_local - internal_hosts)}"
    )
    assert bridge_local <= no_proxy, (
        f"proxy would deny bridge-local traffic: {sorted(bridge_local - no_proxy)}"
    )


def test_hosts_the_auditor_calls_internal_are_reachable(overlay, internal_hosts):
    """The standing pact, now actually held.

    This carried an xfail(strict=True) for as long as detect_internet_use.py
    counted host.docker.internal and 0.0.0.0 as internal while NO_PROXY did not
    list them -- a call to either passed the audit and took a 403 from squid.
    The two were reconciled by dropping them from INTERNAL_HOSTS (see the
    comment on that set for why that direction and not the other), so the marker
    is gone and this is a plain assertion again.
    """
    no_proxy = {h.strip() for h in overlay["services"]["main"]["environment"]["NO_PROXY"].split(",")}
    missing = internal_hosts - no_proxy
    assert not missing, (
        f"detect_internet_use.py treats {sorted(missing)} as internal, but NO_PROXY "
        f"does not list them, so the proxy will deny them."
    )


# ---------------------------------------------------------------- packaging


def test_malformed_acl_fails_the_build_not_the_run(squid_lines):
    """A squid that exits at container start costs an agent phase to diagnose."""
    df = DOCKERFILE.read_text()
    assert "squid -k parse" in df, "Dockerfile does not validate squid.conf at build time"


def test_entrypoint_surfaces_denials_in_docker_logs():
    """The access log is a file, so without this tail an operator debugging a
    blocked run sees an empty `docker logs`."""
    sh = ENTRYPOINT.read_text()
    assert "tail -F /var/log/squid/access.log" in sh
    assert "chown" in sh and "proxy:proxy" in sh, "squid cannot write its own log dir otherwise"
    assert sh.rstrip().endswith('exec /usr/local/bin/entrypoint.sh "$@"'), (
        "must exec the upstream entrypoint so squid is PID 1 and receives signals"
    )
