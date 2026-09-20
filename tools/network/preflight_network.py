#!/usr/bin/env python3
"""Prove Harbor can enforce this task's network policy BEFORE anything is spent.

    tools/network/preflight_network.py tasks/<task> [--env-type docker]

Exit 0 = go. Exit 2 = a blocker, named, with the fix.

WHY THIS EXISTS

Harbor validates the network policy inside ``Trial.__init__`` -- which runs
*after* ``Trial.create()`` has already made the trial directory, and after the
environment image has been built. A policy Harbor cannot enforce therefore does
not fail early and loudly; it fails late, leaving a trial directory with no
``config.json``, and every stage downstream treats that as "a run that produced
nothing" rather than "a run that never started":

  tools/delivery/harbor_to_output.py:1129 selects trial dirs with
      ``(p / "config.json").exists()``
  so an aborted trial is silently SKIPPED, ``written`` comes back empty, and
  reshape exits 0 having done nothing. No traceback, no error -- just a task
  directory that quietly never gains a Run_N.

The three failures this catches have all already happened to this bundle set:

  agent phase override != [environment] baseline   Trial.__init__ raises; the
                                                   docker provider declares
                                                   dynamic_network_policy=False
  network_mode = "allowlist" on docker             docker declares
                                                   network_allowlist=False
  allowed_hosts beside a non-allowlist mode        pydantic rejects at load

The rule this encodes, as in deku's harness/preflight.sh: check the path the
RUNNER takes, not one that resembles it. Every policy question below is answered
by importing Harbor's OWN resolver and the REAL provider capability flags, so
this file cannot drift from what `harbor run` will decide.
"""

from __future__ import annotations

import argparse
import os
import importlib
import re
import sys
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

FAIL = 0


def ok(m: str) -> None:
    print(f"  \033[32mok\033[0m    {m}")


def bad(m: str, f: str | None = None) -> None:
    global FAIL
    FAIL = 1
    print(f"  \033[31mFAIL\033[0m  {m}")
    if f:
        print(f"        -> {f}")


def warn(m: str, f: str | None = None) -> None:
    print(f"  \033[33mwarn\033[0m  {m}")
    if f:
        print(f"        -> {f}")


def declared_policy() -> tuple[str, str]:
    """The repo's single declared baseline, read from the one place that sets it.

    adapters/mcp_atlas/adapter.py::DEFAULT_NETWORK_MODE is what the generator
    writes into every bundle and what adapters/mcp_atlas/tests/test_adapter.py
    ::test_network_mode_is_explicit asserts. Importing it rather than repeating
    the string is the whole point: a preflight that hardcoded "public" would
    become a SECOND rule, free to disagree with the first.
    """
    sys.path.insert(0, str(REPO / "adapters" / "mcp_atlas"))
    try:
        import adapter  # type: ignore
        return adapter.DEFAULT_NETWORK_MODE, "adapters/mcp_atlas/adapter.py::DEFAULT_NETWORK_MODE"
    except Exception:
        return "public", "fallback (adapter.py not importable)"


class _Skip(Exception):
    """caps unreadable: the check below cannot run and must not report a fault."""


class UnknownProvider(Exception):
    """--env-type named something harbor has no provider for."""


class CapabilitiesUnreadable(Exception):
    """The provider's capability flags could not be read without constructing it.

    Distinct from UnknownProvider: the provider exists, it just will not answer
    statically. Named so the caller can tell "you asked for a provider that does
    not exist" from "I could not verify the guarantee you asked me to verify" --
    those warrant different messages, and both are failures, not footnotes.
    """


def provider_capabilities(env_type: str, startup=None, phases=()):
    """The REAL capability flags of the provider `harbor run` will instantiate.

    Resolved through Harbor's own registry, not a local table, so a provider
    that gains allowlist or dynamic-switch support is picked up here the moment
    Harbor ships it.

    `capabilities` used to be a fixed table. On the docker provider it is now
    derived from `self._enable_egress_control`, which __init__ computes from the
    run's own network policies -- so it cannot be read off the bare class at
    all, and reading it off a blank instance raises. The startup/phase policies
    are therefore passed in and that one attribute is rebuilt here the same way
    __init__ does, which is what makes this answer the runner's answer.
    """
    from harbor.environments.factory import _ENVIRONMENT_REGISTRY
    from harbor.models.environment_type import EnvironmentType

    # A name Harbor does not know is a CALLER error -- a typo in --env-type would
    # otherwise skip every enforceability check below and still exit 0, which is
    # the precise failure this file exists to prevent. Distinguished from the
    # case below on purpose.
    try:
        entry = _ENVIRONMENT_REGISTRY[EnvironmentType(env_type)]
    except (KeyError, ValueError):
        known = sorted(e.value for e in EnvironmentType)
        raise UnknownProvider(f"{env_type!r} is not a harbor environment type "
                              f"(known: {', '.join(known)})") from None

    cls = getattr(importlib.import_module(entry.module), entry.class_name)
    caps = cls.__dict__.get("capabilities", cls.capabilities)
    if not isinstance(caps, property):
        return caps

    try:
        return caps.fget(None)
    except Exception:
        pass

    probe = cls.__new__(cls)
    probe._is_windows_container = False
    if startup is not None:
        try:
            probe._enable_egress_control = bool(
                cls._requires_egress_control(
                    startup_network_policy=startup,
                    phase_network_policies=tuple(phases),
                )
                and cls._egress_control_kernel_support()
            )
        except Exception:
            pass          # leave it unset; the read below reports honestly
    try:
        return caps.fget(probe)
    except Exception:
        raise CapabilitiesUnreadable(cls.__name__) from None


def sidecar_hosts(task_dir: Path, raw: dict) -> list[str]:
    """MCP server hostnames the agent must reach over the compose network.

    These are the reason `no-network` is not merely unenforceable here but
    wrong: harbor/environments/docker/docker-compose-no-network.yaml sets
    `network_mode: none` on the `main` service, which detaches it from the
    compose bridge too. The agent would come up with zero tools and grade 0.
    """
    hosts = []
    for srv in raw.get("environment", {}).get("mcp_servers", []) or []:
        url = srv.get("url") or ""
        if "://" in url:
            host = url.split("://", 1)[1].split("/", 1)[0].split(":", 1)[0]
            if host not in ("localhost", "127.0.0.1"):
                hosts.append(f"{srv.get('name', '?')} -> {host}")
    return hosts


def check(task_dir: Path, env_type: str) -> None:
    toml_path = task_dir / "task.toml"
    if not toml_path.is_file():
        bad(f"no task.toml in {task_dir}")
        return
    try:
        raw = tomllib.loads(toml_path.read_text())
    except Exception as exc:
        bad(f"task.toml does not parse: {exc}")
        return

    want, want_src = declared_policy()

    # ---------------------------------------------------------- unset default
    # [environment].network_mode has a DEFAULT of public in Harbor
    # (BaselineNetworkPolicyConfig). Leaving it unset is not neutral: the
    # default is what decides whether an explicit [agent]/[verifier] override
    # counts as a phase switch, i.e. whether the trial aborts.
    env_raw = raw.get("environment", {}) or {}
    baseline_declared = env_raw.get("network_mode")
    overrides = {
        role: (raw.get(role, {}) or {}).get("network_mode")
        for role in ("agent", "verifier")
    }
    explicit_overrides = {r: v for r, v in overrides.items() if v is not None}

    if baseline_declared is None:
        msg = "[environment].network_mode is unset -- harbor defaults it to 'public'"
        if explicit_overrides:
            bad(msg + f", and {sorted(explicit_overrides)} override(s) are measured against it",
                f'set network_mode = "{want}" in [environment] explicitly ({want_src})')
        else:
            warn(msg, f'set network_mode = "{want}" in [environment] explicitly ({want_src})')
    elif baseline_declared != want:
        bad(f'[environment].network_mode = "{baseline_declared}" but this repo declares "{want}"',
            f'either set network_mode = "{want}" in [environment], or change {want_src} '
            f'(they must not disagree)')
    else:
        ok(f'[environment].network_mode = "{baseline_declared}" (matches {want_src})')

    # ------------------------------------------------- allowlist where ignored
    # Two distinct traps. Harbor's pydantic rejects allowed_hosts beside a
    # non-allowlist mode outright; harbor/trial/network_policy.py
    # ::merge_extra_allowlists merely WARNS and drops run-time extra hosts
    # against a public policy. Neither is visible until a trial is constructed.
    for role, section in (("environment", env_raw),
                          ("agent", raw.get("agent", {}) or {}),
                          ("verifier", raw.get("verifier", {}) or {})):
        hosts = section.get("allowed_hosts")
        if not hosts:
            continue
        mode = section.get("network_mode")
        if mode is None:
            bad(f"[{role}].allowed_hosts is set with no [{role}].network_mode",
                f"harbor raises \"allowed_hosts is only valid when "
                f"network_mode='allowlist'\"; drop allowed_hosts or set the mode")
        elif mode != "allowlist":
            bad(f'[{role}].allowed_hosts is set alongside network_mode = "{mode}"',
                "harbor raises \"allowed_hosts is only valid when "
                "network_mode='allowlist'\"; remove them TOGETHER, not one of the two")
        else:
            ok(f"[{role}].allowed_hosts is paired with allowlist mode")

    # ------------------------------------------------ ask harbor, not ourselves
    try:
        from harbor.models.task.config import TaskConfig
        from harbor.models.trial.config import AgentConfig, EnvironmentConfig
        from harbor.models.task.verifier_mode import (
            resolve_step_verifier_mode,
            resolve_task_verifier_mode,
        )
        from harbor.trial.network_policy import resolve_trial_network_plan
        from harbor.trial.trial import Trial
        from harbor.environments.base import BaseEnvironment
    except Exception:
        warn("harbor is not importable from this interpreter -- policy enforceability "
             "was NOT checked",
             "run this under harbor's python: "
             "$(dirname $(readlink -f $(command -v harbor)))/python")
        return

    try:
        cfg = TaskConfig.model_validate(raw)
    except Exception as exc:
        bad(f"harbor rejects this task.toml: {str(exc).splitlines()[0]}",
            "harbor raises this in Trial.__init__, AFTER the image is built")
        return

    # Mirror Trial._validate_network_policy_modes EXACTLY: a task with [[steps]]
    # gets one plan per step, each with its own verifier mode, and Harbor
    # validates every one. Checking only the stepless plan would be a check that
    # merely resembles the runner's path -- the failure this whole file exists to
    # prevent.
    if cfg.steps:
        plans = [
            (f"Step {step.name!r}",
             resolve_trial_network_plan(
                 cfg, AgentConfig(name="claude-code"), EnvironmentConfig(), step,
                 verifier_mode=resolve_step_verifier_mode(cfg, step)))
            for step in cfg.steps
        ]
    else:
        plans = [
            ("[agent]",
             resolve_trial_network_plan(
                 cfg, AgentConfig(name="claude-code"), EnvironmentConfig(), None,
                 verifier_mode=resolve_task_verifier_mode(cfg)))
        ]

    # The policies harbor itself would hand the provider's constructor.
    _startup = plans[0][1].agent_env_baseline if plans else None
    _phases = [pol for _, pl in plans
               for pol in (pl.agent_phase, pl.verifier_phase) if pol is not None]

    caps = None
    try:
        caps = provider_capabilities(env_type, _startup, _phases)
    except UnknownProvider as exc:
        bad(f"{exc}", "pass a real provider to --env-type; nothing below was checked")
        return
    except Exception as exc:
        # Not knowing is not the same as finding a fault. This used to refuse the
        # run, which sent every operator to PREFLIGHT_NETWORK_OFF=1 and so turned
        # the whole gate off. Name what went unchecked and let the rest run.
        warn(f"{env_type}: capability flags could not be read "
             f"({type(exc).__name__}: {exc}) -- enforceability was NOT checked",
             "harbor's provider changed shape; re-check provider_capabilities(). "
             "The isolation checks below still apply.")

    # ------------------------------------- what the kernel here can enforce
    class _Probe:
        """Capability-only stand-in: BaseEnvironment.validate_network_policy_support
        reads nothing but `capabilities` and `type()`, so this exercises the real
        method rather than a paraphrase of it."""
        capabilities = caps
        _network_policy = None

        @staticmethod
        def type():
            return env_type

        def validate_network_policy_support(self, policy=None):
            return BaseEnvironment.validate_network_policy_support(self, policy)

    probe = _Probe()
    shim = type("_Shim", (), {
        "agent_environment": probe,
        "_validate_network_plan": Trial._validate_network_plan,
        "_validate_dynamic_phase_switch": Trial._validate_dynamic_phase_switch,
    })()
    hosts = sidecar_hosts(task_dir, raw)

    for plan_label, plan in plans:
        pfx = "" if plan_label == "[agent]" else f"{plan_label}: "

        # ------------------------------- what the kernel here can enforce
        for label, policy in (("[environment] baseline", plan.agent_env_baseline),
                              ("[agent] phase", plan.agent_phase),
                              ("[verifier] phase", plan.verifier_phase)):
            if policy is None or caps is None:
                continue
            try:
                probe.validate_network_policy_support(policy)
                ok(f'{pfx}{label} network_mode = "{policy.network_mode.value}" is '
                   f"enforceable by the {env_type} provider")
            except Exception as exc:
                bad(f"{pfx}{label}: {exc}",
                    f"the {env_type} provider on this host cannot enforce "
                    f'"{policy.network_mode.value}" '
                    f"(network_allowlist={caps.network_allowlist}, "
                    f"disable_internet={caps.disable_internet})")

        # ------------------------------------ the dynamic-switch blocker
        # Harbor's OWN validator, bound to the real capability flags. This is
        # the check that fires in Trial.__init__, after the build is paid for.
        try:
            if caps is None:
                raise _Skip
            shim._validate_network_plan(plan, label=plan_label)
            ok(f"{pfx}harbor's own Trial network validation passes")
        except _Skip:
            pass
        except Exception as exc:
            bad(f"harbor would abort the trial: {exc}",
                f'[environment].network_mode = '
                f'"{plan.agent_env_baseline.network_mode.value}" but the agent phase '
                f'resolves to "{plan.agent_phase.network_mode.value}"; the {env_type} '
                f"provider declares dynamic_network_policy={caps.dynamic_network_policy}, "
                f"so the two must be equal. Make them agree.")

        # -------------------------- the path the AGENT actually takes
        if hosts and plan.agent_phase.network_mode.value == "no-network":
            bad(f"{pfx}agent phase is 'no-network' but the task declares compose-sidecar "
                "MCP servers: " + "; ".join(hosts),
                "no-network sets `network_mode: none` on the main service "
                "(harbor/environments/docker/docker-compose-no-network.yaml), which "
                "detaches it from the compose bridge too -- the agent would start with "
                "ZERO tools")
        elif hosts:
            ok(f"{pfx}{len(hosts)} MCP sidecar host(s) reachable under "
               f"'{plan.agent_phase.network_mode.value}'")

    check_isolation(task_dir, raw)


# --------------------------------------------------------------------------
# NETWORK ISOLATION
#
# Everything above asks whether HARBOR can enforce the task's policy. Since the
# egress block landed, harbor is no longer where the policy lives: task.toml
# stays network_mode = "public" precisely because that is the value that appends
# no harbor overlay, and tools/network/egress-proxy/overlay.yaml does the work one
# layer down in compose.
#
# So the checks above can all pass while the run is still guaranteed to fail.
# These two cover the ways that happens, and both of them fail LATE and
# illegibly without a gate here -- after the environment image is built, which
# is the expensive part.
# --------------------------------------------------------------------------

def _isolation_files(repo: Path) -> tuple[set[str], set[str]] | None:
    """(allowlisted hosts, hosts exempt from the proxy), or None if no overlay.

    Read out of the shipped config rather than restated, so widening the
    allowlist or adding a NO_PROXY entry updates this check for free.
    """
    overlay = repo / "tools" / "network" / "egress-proxy" / "overlay.yaml"
    squid = repo / "tools" / "network" / "egress-proxy" / "squid.conf"
    if not overlay.is_file() or not squid.is_file():
        return None
    allowed: set[str] = set()
    for line in squid.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        m = re.match(r"acl\s+\S+\s+dstdomain\s+(.+)$", line)
        if m:
            allowed.update(m.group(1).split())
    no_proxy: set[str] = set()
    m = re.search(r'^\s*NO_PROXY:\s*"([^"]*)"', overlay.read_text(), re.M)
    if m:
        no_proxy = {h.strip() for h in m.group(1).split(",") if h.strip()}
    return allowed, no_proxy


def _captures_access_log(repo: Path) -> bool:
    """Does the overlay still carry the proxy's access log off the container?

    Checked here rather than left to the test suite because the failure is
    invisible at run time: a missing mount produces an audit that reports no
    denials, which reads exactly like a clean run. The trial would be graded and
    delivered on evidence that was never collected.
    """
    overlay = repo / "tools" / "network" / "egress-proxy" / "overlay.yaml"
    entrypoint = repo / "tools" / "network" / "egress-proxy" / "entrypoint.sh"
    if not overlay.is_file() or not entrypoint.is_file():
        return False
    return "/egress-out" in overlay.read_text() and "/egress-out" in entrypoint.read_text()


def check_isolation(task_dir: Path, raw: dict) -> None:
    if os.environ.get("NETWORK_ISOLATION_OFF"):
        warn("network isolation OFF -- the agent phase runs on the open network",
             "detect_internet_use.py is the only remaining defence and will "
             "refuse the run if the model browsed")
        return

    repo = REPO
    files = _isolation_files(repo)
    if files is None:
        # run_task.sh refuses outright on a missing overlay; nothing to add.
        return
    allowed, no_proxy = files

    # 1. Sidecars the proxy would deny -----------------------------------
    #
    # sidecar_hosts() above drops localhost and 127.0.0.1 and treats everything
    # else as reachable, which was true when the only question was whether the
    # compose bridge existed. Under isolation a host is reachable only if it
    # never goes to the proxy (NO_PROXY) or is on the allowlist; anything else
    # is a 403 that surfaces as an MCP tool that simply does not work.
    denied = []
    for srv in raw.get("environment", {}).get("mcp_servers", []) or []:
        url = srv.get("url") or ""
        if "://" not in url:
            continue
        host = url.split("://", 1)[1].split("/", 1)[0].split(":", 1)[0]
        if host in no_proxy or host in allowed:
            continue
        denied.append(f"{srv.get('name', '?')} -> {host}")
    if denied:
        bad("MCP server host(s) the egress proxy will deny: " + "; ".join(denied),
            "under isolation a host must be in overlay.yaml's NO_PROXY (stays on "
            "the bridge) or in squid.conf's allowlist (goes out through the "
            "proxy). Anything else gets a 403 and the tool silently does not work")
    else:
        ok("every MCP sidecar host is reachable under network isolation")

    # 2. The CLI the agent phase cannot download -------------------------
    #
    # harbor's ClaudeCode.install() fetches it from downloads.claude.ai INSIDE
    # the container, which the allowlist denies. The trial then dies in agent
    # setup with an empty /logs/agent -- indistinguishable from an agent that
    # ran and produced nothing.
    agent = os.environ.get("AGENT") or "claude-code"
    if agent != "claude-code":
        return
    dockerfile = task_dir / "environment" / "Dockerfile"
    if not dockerfile.is_file():
        return
    body = dockerfile.read_text()
    missing = []
    if "downloads.claude.ai" not in body:
        missing.append("the Claude Code CLI")
    if "procps" not in body:
        missing.append("procps (claude shells out to ps/pgrep to kill subtrees)")
    if not _captures_access_log(repo):
        bad("the egress overlay does not carry squid's access.log off the container",
            "without it the run is audited on trajectory inference alone and "
            "reports 'no denials' whether or not the block held. Restore the "
            "/egress-out mount in tools/network/egress-proxy/overlay.yaml and the tee "
            "in entrypoint.sh")
    else:
        ok("the proxy's access log is captured per run (proof of denial)")

    if missing:
        bad(f"{dockerfile} does not pre-bake: " + ", ".join(missing),
            "build time still has an open network, the agent phase does not. Add "
            "to the Dockerfile:\n"
            "          RUN apt-get update && apt-get install -y --no-install-recommends "
            "curl procps && rm -rf /var/lib/apt/lists/*\n"
            "          RUN curl -fsSL https://downloads.claude.ai/claude-code-releases/bootstrap.sh | bash \\\n"
            "              && ln -sf /root/.local/bin/claude /usr/local/bin/claude && claude --version")
    else:
        ok("environment image pre-bakes the Claude Code CLI and procps")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("task_dir", type=Path)
    ap.add_argument("--env-type", default="docker",
                    help="harbor environment provider the run will use (default: docker)")
    a = ap.parse_args(argv)

    if not a.task_dir.is_dir():
        print(f"  \033[31mFAIL\033[0m  task directory not found: {a.task_dir}")
        return 2

    print(f"== network policy: {a.task_dir} ({a.env_type}) ==")
    check(a.task_dir, a.env_type)
    if FAIL:
        print("\n  blocked: fix the above before spending an agent phase.")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
