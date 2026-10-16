#!/usr/bin/env python3
"""Prove Harbor can enforce this task's network policy BEFORE anything is spent.

    tools/network/preflight_network.py tasks/<task> [--env-type docker]
                                       [--allow-agent-host HOST ...]
                                       [--agent-mode | --egress-control]

Exit 0 = go. Exit 2 = a blocker, named, with the fix. --agent-mode prints only
the agent-phase network_mode Harbor will enforce (public, allowlist or
no-network) and exits 0; scripts/run_task.sh gates its agent-side layers on it.

WHY THIS EXISTS

Harbor validates the network policy inside ``Trial.__init__`` -- which runs
*after* ``Trial.create()`` has already made the trial directory, and after the
environment image has been built. A policy Harbor cannot enforce therefore does
not fail early and loudly; it fails late, leaving a trial directory with no
``config.json``, and every stage downstream treats that as "a run that produced
nothing" rather than "a run that never started":

  tools/delivery_utils/harbor_to_output.py:1129 selects trial dirs with
      ``(p / "config.json").exists()``
  so an aborted trial is silently SKIPPED, ``written`` comes back empty, and
  reshape exits 0 having done nothing. No traceback, no error -- just a task
  directory that quietly never gains a Run_N.

The failures this catches:

  a phase policy the provider cannot enforce       Harbor rejects the trial; on
                                                   docker, allowlist and phase
                                                   switching need the egress
                                                   sidecar's nftables support
  allowed_hosts beside a non-allowlist mode        pydantic rejects at load

The policy itself is the bundle's. Trinity's bundle standard has every task.toml
declare network_mode for [environment], [agent] and [verifier] (and
[verifier.environment] under environment_mode = "separate"); this file reports a
missing declaration and never substitutes a policy of its own.

The rule this encodes, as in deku's harness/preflight.sh: check the path the
RUNNER takes, not one that resembles it. Every policy question below is answered
by importing Harbor's OWN resolver and the REAL provider capability flags, so
this file cannot drift from what `harbor run` will decide.
"""

from __future__ import annotations

import argparse
import os
import importlib
import sys
import tomllib
from pathlib import Path


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


def provider_capabilities(env_type: str, startup=None, phases=(), uses_compose=False):
    """The REAL capability flags of the provider `harbor run` will instantiate.

    Resolved through Harbor's own registry, not a local table, so a provider
    that gains allowlist or dynamic-switch support is picked up here the moment
    Harbor ships it.

    `capabilities` used to be a fixed table. On the docker provider it is now
    derived from `self._enable_egress_control`, which __init__ computes from the
    run's own network policies -- so it cannot be read off the bare class at
    all, and reading it off a blank instance raises. The startup/phase policies
    are therefore passed in and that one attribute is rebuilt here the same way
    __init__ does, which is what makes this answer the runner's answer. 0.24.0's
    property also reads `_uses_compose`, which a bare instance cannot answer, so
    the probe class answers it from the task directory instead.
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

    probe_cls = type(cls.__name__, (cls,), {"_uses_compose": property(lambda self: uses_compose)})
    probe = probe_cls.__new__(probe_cls)
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


def resolve_plans(raw: dict, agent_hosts: list[str]):
    """Harbor's own network plans for this task: [(label, plan), ...].

    Mirrors Trial._validate_network_policy_modes: a task with [[steps]] gets one
    plan per step, each with its own verifier mode. `agent_hosts` are the run's
    --allow-agent-host values, merged by Harbor exactly as `harbor run` does.
    """
    from harbor.models.task.config import TaskConfig
    from harbor.models.trial.config import AgentConfig, EnvironmentConfig
    from harbor.models.task.verifier_mode import (
        resolve_step_verifier_mode,
        resolve_task_verifier_mode,
    )
    from harbor.trial.network_policy import resolve_trial_network_plan

    cfg = TaskConfig.model_validate(raw)
    agent = AgentConfig(name="claude-code", extra_allowed_hosts=list(agent_hosts))
    if cfg.steps:
        return cfg, [
            (f"Step {step.name!r}",
             resolve_trial_network_plan(
                 cfg, agent, EnvironmentConfig(), step,
                 verifier_mode=resolve_step_verifier_mode(cfg, step)))
            for step in cfg.steps
        ]
    return cfg, [
        ("[agent]",
         resolve_trial_network_plan(
             cfg, agent, EnvironmentConfig(), None,
             verifier_mode=resolve_task_verifier_mode(cfg)))
    ]


def check(task_dir: Path, env_type: str, agent_hosts: list[str]) -> None:
    toml_path = task_dir / "task.toml"
    if not toml_path.is_file():
        bad(f"no task.toml in {task_dir}")
        return
    try:
        raw = tomllib.loads(toml_path.read_text())
    except Exception as exc:
        bad(f"task.toml does not parse: {exc}")
        return

    # ------------------------------------------------------ declared per phase
    # Harbor defaults an undeclared [environment].network_mode to public and an
    # undeclared phase to its baseline. Running such a bundle is Harbor's call,
    # not this file's, so a gap is reported and never filled in here.
    env_raw = raw.get("environment", {}) or {}
    ver_raw = raw.get("verifier", {}) or {}
    declared = {
        "[environment]": env_raw.get("network_mode"),
        "[agent]": (raw.get("agent", {}) or {}).get("network_mode"),
        "[verifier]": ver_raw.get("network_mode"),
    }
    if ver_raw.get("environment_mode") == "separate" or ver_raw.get("environment"):
        declared["[verifier.environment]"] = (ver_raw.get("environment") or {}).get("network_mode")
    for table, mode in declared.items():
        if mode is None:
            warn(f"{table}.network_mode is not declared -- harbor falls back to its default",
                 "the bundle standard declares it in every phase; that is the bundle's fix")
        else:
            ok(f'{table}.network_mode = "{mode}" (declared by the bundle)')

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
        from harbor.trial.trial import Trial
        from harbor.environments.base import BaseEnvironment
    except Exception:
        bad("harbor is not importable from this interpreter -- policy enforceability "
            "cannot be checked",
            "run this under harbor's python: "
            "$(dirname $(readlink -f $(command -v harbor)))/python")
        return

    try:
        cfg, plans = resolve_plans(raw, agent_hosts)
    except Exception as exc:
        bad(f"harbor rejects this task.toml: {str(exc).splitlines()[0]}",
            "harbor raises this in Trial.__init__, AFTER the image is built")
        return

    # ------------------------------------------------ where the verifier runs
    # Harbor 0.24 resolves the verifier environment on its own: environment_mode
    # "separate" builds it from [verifier.environment].docker_image, else
    # tests/Dockerfile or tests/docker-compose.yaml, else a copy of
    # [environment]. Trial-level --extra-docker-compose and --mounts reach the
    # AGENT environment only, so nothing this harness adds touches the verifier.
    try:
        from harbor.models.task.paths import TaskPaths
        from harbor.models.task.verifier_mode import (
            resolve_task_verifier_mode,
            resolve_verifier_environment_definition,
        )
        mode = resolve_task_verifier_mode(cfg).value
        if mode == "separate":
            root = task_dir.resolve()
            definition = resolve_verifier_environment_definition(cfg, TaskPaths(root))
            source = (f"image {definition.config.docker_image}" if definition.config.docker_image
                      else f"{definition.directory.resolve().relative_to(root)}/")
            ok(f'[verifier] environment_mode = "separate" -- built from {source}')
        else:
            warn(f'[verifier] environment_mode resolves to "{mode}" -- the bundle\'s tests run '
                 "in the agent's container",
                 "the bundle standard declares environment_mode = \"separate\"; that is the "
                 "bundle's fix")
    except Exception as exc:
        bad(f"harbor cannot resolve this task's verifier environment: "
            f"{str(exc).splitlines()[0]}",
            "harbor resolves it when the trial starts and would fail there")

    # The policies harbor itself would hand the provider's constructor.
    _startup = plans[0][1].agent_env_baseline if plans else None
    _phases = [pol for _, pl in plans
               for pol in (pl.agent_phase, pl.verifier_phase) if pol is not None]

    caps = None
    try:
        caps = provider_capabilities(
            env_type, _startup, _phases,
            uses_compose=(task_dir / "environment" / "docker-compose.yaml").is_file())
    except UnknownProvider as exc:
        bad(f"{exc}", "pass a real provider to --env-type; nothing below was checked")
        return
    except Exception as exc:
        # The policy is the guarantee and Harbor enforces it, so not being able
        # to read whether it can is a blocker, not a footnote. Harbor would
        # refuse the same trial after the image build; refuse it before.
        bad(f"{env_type}: capability flags could not be read "
            f"({type(exc).__name__}: {exc}) -- enforceability was NOT checked",
            "harbor's provider changed shape; re-check provider_capabilities()")

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
        elif hosts and plan.agent_phase.network_mode.value == "public":
            ok(f"{pfx}{len(hosts)} MCP sidecar host(s) reachable under 'public'")

    check_isolation(task_dir, raw, plans)


# --------------------------------------------------------------------------
# WHAT THE AGENT ACTUALLY NEEDS TO REACH
#
# Everything above asks whether HARBOR can enforce the task's policy. These ask
# whether the policy, once enforced, still lets the run do its work -- both fail
# LATE and illegibly without a gate here, after the environment image is built.
# --------------------------------------------------------------------------

def check_isolation(task_dir: Path, raw: dict, plans) -> None:
    # 1. MCP sidecars under a closed agent phase ------------------------
    #
    # Compose sidecars need not be listed in the allowlist: harbor 0.24.0 moves
    # every service without its own networking into the egress sidecar's
    # namespace, where they reach each other on 127.0.0.1. Their service NAMES
    # resolve only because scripts/patch_harbor.py aliases each moved service to
    # 127.0.0.1 on the sidecar; unpatched, http://<service>:<port> fails (all 22
    # MCP servers of a light-servers bundle failed to connect). no-network is
    # handled above, where the sidecars are unreachable.
    for plan_label, plan in plans:
        pfx = "" if plan_label == "[agent]" else f"{plan_label}: "
        if plan.agent_phase.network_mode.value == "allowlist":
            hosts = [h for h in plan.agent_phase.allowed_hosts]
            ok(f"{pfx}agent phase allowlist: {', '.join(hosts) or '(empty)'}; compose "
               "sidecars reachable by service name (sidecar aliases, patch_harbor.py)")

    # 2. The CLI the agent setup cannot download -------------------------
    #
    # harbor's ClaudeCode.install() fetches it from downloads.claude.ai inside
    # the container. Setup runs under the [environment] baseline (the [agent]
    # policy applies to agent.run() only), so only a closed BASELINE denies it.
    #
    # claude-code only. The openhands agent (scripts/run_task.sh's default)
    # brings its SDK in from the openhands-runtime image through
    # tools/openhands_agent/overlay.yaml, so the bundle needs nothing baked.
    agent = os.environ.get("AGENT") or "openhands"
    if agent != "claude-code" or not plans:
        return
    if plans[0][1].agent_env_baseline.network_mode.value == "public":
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

    if missing:
        bad(f"{dockerfile} does not pre-bake: " + ", ".join(missing),
            "the [environment] baseline is closed, so agent setup cannot download "
            "them. Add to the Dockerfile:\n"
            "          RUN apt-get update && apt-get install -y --no-install-recommends "
            "curl procps && rm -rf /var/lib/apt/lists/*\n"
            "          RUN curl -fsSL https://downloads.claude.ai/claude-code-releases/bootstrap.sh | bash \\\n"
            "              && ln -sf /root/.local/bin/claude /usr/local/bin/claude && claude --version")
    else:
        ok("environment image pre-bakes the Claude Code CLI and procps")


def egress_control(task_dir: Path, agent_hosts: list[str]) -> int:
    """`--egress-control`: print on/off -- will harbor's docker sidecar govern main?

    Harbor turns it on when any policy of the agent environment is not public
    (DockerEnvironment._requires_egress_control): the [environment] baseline, or
    the agent or shared-verifier phase.
    """
    raw = tomllib.loads((task_dir / "task.toml").read_text())
    _, plans = resolve_plans(raw, agent_hosts)
    policies = []
    for _, plan in plans:
        policies.append(plan.agent_env_baseline)
        policies.append(plan.agent_phase)
        if plan.verifier_env_baseline is None:
            policies.append(plan.verifier_phase)
    print("on" if any(p.network_mode.value != "public" for p in policies) else "off")
    return 0


def agent_mode(task_dir: Path, agent_hosts: list[str]) -> int:
    """`--agent-mode`: print the agent-phase network_mode harbor will enforce."""
    raw = tomllib.loads((task_dir / "task.toml").read_text())
    _, plans = resolve_plans(raw, agent_hosts)
    modes = {plan.agent_phase.network_mode.value for _, plan in plans}
    # A stepped task can close some steps and not others; any closed step makes
    # the run closed for the layers that read this.
    for mode in ("no-network", "allowlist", "public"):
        if mode in modes:
            print(mode)
            return 0
    return 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("task_dir", type=Path)
    ap.add_argument("--env-type", default="docker",
                    help="harbor environment provider the run will use (default: docker)")
    ap.add_argument("--allow-agent-host", action="append", default=[], metavar="HOST",
                    help="a host this run passes to harbor as --allow-agent-host")
    ap.add_argument("--agent-mode", action="store_true",
                    help="print the agent-phase network_mode harbor will enforce, and exit")
    ap.add_argument("--egress-control", action="store_true",
                    help="print on/off: whether harbor's docker egress sidecar governs main")
    a = ap.parse_args(argv)

    if a.agent_mode:
        return agent_mode(a.task_dir, a.allow_agent_host)
    if a.egress_control:
        return egress_control(a.task_dir, a.allow_agent_host)

    if not a.task_dir.is_dir():
        print(f"  \033[31mFAIL\033[0m  task directory not found: {a.task_dir}")
        return 2

    print(f"== network policy: {a.task_dir} ({a.env_type}) ==")
    check(a.task_dir, a.env_type, a.allow_agent_host)
    if FAIL:
        print("\n  blocked: fix the above before spending an agent phase.")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
