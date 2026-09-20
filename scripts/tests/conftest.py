"""Shared fixtures for the run_task.sh stage tests.

The one thing in here exists because run_task.sh's harbor stage runs
scripts/patch_harbor.py before it dispatches, and that script locates the harbor
package RELATIVE TO whichever `harbor` is first on PATH -- it globs
<venv>/lib/python*/site-packages/harbor/... where <venv> is the binary's
grandparent.

Any test that shadows `harbor` with a stub therefore points patch_harbor.py at a
tmp dir containing no harbor install, and it raises before harbor is ever
invoked. The symptom is an empty argv file and an assertion about a flag that
was never passed, which reads like a dispatch bug in run_task.sh rather than a
missing fixture.

test_network_policy.py grew its own private copy of this for the same reason.
This is that helper, shared, so the next file to stub `harbor` gets it for free.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

import pytest


def mirror_harbor_package(tmp_path: Path) -> bool:
    """Mirror the files patch_harbor.py rewrites into a stub venv layout.

    COPIES, never links: the patcher rewrites them in place, and a link would
    edit the real harbor install out from under every other test on the machine.

    Returns False when harbor is not installed at all, so callers can skip
    rather than fail -- a laptop without harbor is not a broken run_task.sh.
    """
    real_harbor = shutil.which("harbor")
    if not real_harbor:
        return False

    venv_root = Path(real_harbor).resolve().parent.parent
    for cc in venv_root.glob(
        "lib/python*/site-packages/harbor/agents/installed/claude_code.py"
    ):
        rel = cc.relative_to(venv_root)
        pkg = cc.parent.parent.parent
        for src, dest_rel in (
            (pkg / "agents" / "installed" / "claude_code.py", rel),
            (pkg / "trial" / "trial.py",
             rel.parent.parent.parent / "trial" / "trial.py"),
            # patch_harbor.py's score-table suppression rewrites this one; without
            # it the patcher reports a MISS, exits non-zero, and every stubbed
            # harbor stage dies before dispatch.
            (pkg / "cli" / "jobs.py",
             rel.parent.parent.parent / "cli" / "jobs.py"),
        ):
            if src.exists():
                dest = tmp_path / dest_rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(src.read_text())
        return True
    return False


@pytest.fixture
def harbor_package_mirror(tmp_path):
    """Make `tmp_path` look enough like a harbor venv for patch_harbor.py."""
    if not mirror_harbor_package(tmp_path):
        pytest.skip("harbor is not installed; cannot mirror its package")
    return tmp_path


@lru_cache(maxsize=1)
def docker_is_usable() -> bool:
    """Whether a daemon actually answers -- not merely whether a client is installed.

    The `docker` client stays on PATH after Desktop or OrbStack stops, so
    which() alone calls a machine ready when it cannot build an image.
    """
    if shutil.which("docker") is None:
        return False
    try:
        return subprocess.run(
            ["docker", "info"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30,
        ).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


requires_docker = pytest.mark.skipif(
    not docker_is_usable(),
    reason="needs a running Docker daemon. run_task.sh reaches it two ways, "
           "both fatal: it waits 120s for one to appear and then exits 3 "
           "(run_task.sh:652-655), or it fails provisioning an image "
           "(ensure_image, run_task.sh:581-610). Either lands well before the "
           "behaviour under test. Marked per-test rather than per-module "
           "because most tests in these files stop at an earlier gate and pass "
           "without a daemon -- skipping those too would cost real coverage.",
)


def credentials_are_usable() -> bool:
    """Would run_task.sh's check_credentials let a preflight through?

    It refuses on a missing or logged-out codex and on a missing Claude token,
    which lands before anything a preflight test is asking about.
    """
    if shutil.which("codex") is None:
        return False
    try:
        out = subprocess.run(["codex", "login", "status"], capture_output=True,
                             text=True, timeout=30)
        if "not logged in" in f"{out.stdout} {out.stderr}".lower():
            return False
    except (OSError, subprocess.SubprocessError):
        return False
    if os.environ.get("CLAUDE_CODE_OAUTH_TOKEN") or os.environ.get("ANTHROPIC_API_KEY"):
        return True
    if (Path.home() / ".claude" / ".credentials.json").is_file():
        return True
    return subprocess.run(
        ["security", "find-generic-password", "-s", "Claude Code-credentials", "-w"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode == 0 if shutil.which("security") else False


requires_credentials = pytest.mark.skipif(
    not credentials_are_usable(),
    reason="needs a logged-in codex and a Claude credential: check_credentials "
           "exits 4 before the preflight stage runs.",
)


requires_harbor = pytest.mark.skipif(
    shutil.which("harbor") is None,
    reason="harbor is not installed. run_task.sh runs patch_harbor.py (:1510) "
           "before it dispatches any stage, and that raises outright when the "
           "harbor package cannot be located -- so the script exits 1 long "
           "before the behaviour under test. Same reasoning as "
           "mirror_harbor_package(): a machine without harbor is not a broken "
           "run_task.sh, and CI runners do not have it on PATH.",
)
