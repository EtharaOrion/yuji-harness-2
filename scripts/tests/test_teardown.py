"""Teardown of a run's containers and volumes, in run_task.sh and stop_harness.sh.

Harbor deletes its own compose project when `harbor run` RETURNS (--delete is
its default, and that teardown carries --volumes). Nothing did so when the
process did NOT return: run_task.sh had no trap at all, and the `|| { ... }`
around the harbor call swallowed the 130, so a Ctrl-C left every container of
the project up, with workspace_data attached to them, and the script walked on
into reshape, mask and finance.

`make stop-harness` could not clear the wreckage either. Its sweep was
stopped-only, so it collected whichever containers happened to sit in `created`
and left every RUNNING one alone; those held the volume open, so the
`dangling=true` volume filter matched nothing and it reported "no unused
harness volumes" while two projects' worth of them sat on the disk.

The docker-backed tests prove the behaviour end to end. The rest are structural,
and cheap enough to catch a revert.
"""
import os
import signal
import subprocess
import time
from pathlib import Path

import pytest

from conftest import requires_docker

REPO = Path(__file__).resolve().parent.parent.parent
RUN_TASK = REPO / "scripts" / "run_task.sh"
STOP_HARNESS = REPO / "scripts" / "stop_harness.sh"

PROJECT = "zz-pytest__teardown__env"
VOLUME = f"{PROJECT}_workspace_data"
CONTAINER = f"{PROJECT}-main-1"
BYSTANDER = "zz-pytest-bystander-1"


def _sh(*cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def _names():
    return _sh("docker", "ps", "-a", "--format", "{{.Names}}").stdout.split()


def _volumes():
    return _sh("docker", "volume", "ls", "-q").stdout.split()


@pytest.fixture
def fake_project():
    """A container+volume shaped like harbor's, plus an unrelated bystander."""
    def _make():
        _sh("docker", "volume", "create",
            "--label", f"com.docker.compose.project={PROJECT}", VOLUME)
        _sh("docker", "run", "-d", "--name", CONTAINER,
            "--label", f"com.docker.compose.project={PROJECT}",
            "-v", f"{VOLUME}:/data", "alpine:3", "sleep", "600")
        # No "__" in the project name, so no sweep may ever touch it.
        _sh("docker", "run", "-d", "--name", BYSTANDER,
            "--label", "com.docker.compose.project=zz-pytest-bystander",
            "alpine:3", "sleep", "600")
    yield _make
    for c in (CONTAINER, BYSTANDER):
        _sh("docker", "rm", "-f", c)
    _sh("docker", "volume", "rm", "-f", VOLUME)


# --------------------------------------------------------------- structural ---

def test_run_task_traps_int_and_term():
    src = RUN_TASK.read_text()
    assert "trap on_interrupt INT" in src
    assert "trap on_terminate TERM" in src


def test_run_task_snapshots_projects_before_harbor():
    """The delta is only this run's if it is taken immediately before harbor."""
    src = RUN_TASK.read_text()
    snap = src.index("snapshot_compose_projects\n")
    harbor = src.index("HARBOR_OUTPUT_OFF=1 command harbor")
    assert snap < harbor
    between = src[snap:harbor]
    assert "docker" not in between, "something starts containers before the snapshot"


def test_run_task_stops_on_an_interrupted_harbor():
    """130/143 must not fall through into reshape, mask and finance."""
    src = RUN_TASK.read_text()
    after = src[src.index("HARBOR_OUTPUT_OFF=1 command harbor"):]
    case = after[:after.index("# A trial DIRECTORY is not a trial")]
    assert "130|143)" in case
    assert "teardown_this_runs_projects" in case
    assert 'exit "$_hrc"' in case


def test_teardown_is_scoped_to_harbor_projects():
    """Never a bare prune: only compose projects carrying harbor's marker."""
    src = RUN_TASK.read_text()
    body = src[src.index("teardown_this_runs_projects() {"):]
    body = body[:body.index("\n}\n")]
    assert "grep '__'" in body
    assert "label=com.docker.compose.project=$p" in body
    assert "prune" not in body


def test_stop_harness_sweeps_running_when_nothing_is_live():
    src = STOP_HARNESS.read_text()
    assert '[ "$FORCE" = 1 ] || [ "${#LIVE[@]}" -eq 0 ]' in src


def test_stop_harness_volumes_no_longer_rely_on_dangling():
    """A volume attached to a surviving container is never dangling."""
    src = STOP_HARNESS.read_text()
    assert "--filter dangling=true" not in src
    assert "_harness_volumes" in src


def test_scripts_are_valid_bash():
    for script in (RUN_TASK, STOP_HARNESS):
        assert _sh("bash", "-n", str(script)).returncode == 0, script


# ------------------------------------------------------------ docker-backed ---

@requires_docker
def test_stop_harness_removes_running_orphans_and_their_volumes(fake_project):
    fake_project()
    assert CONTAINER in _names()
    assert VOLUME in _volumes()

    out = _sh("bash", str(STOP_HARNESS))
    assert CONTAINER not in _names(), out.stdout + out.stderr
    assert VOLUME not in _volumes(), out.stdout + out.stderr
    assert BYSTANDER in _names(), "swept a project that is not harbor's"


@requires_docker
def test_anonymous_volumes_are_swept_too():
    """egress-proxy's base declares VOLUME twice, so each proxy leaks two.

    64-hex name, no labels at all, so no compose-project filter will ever find
    them once their container is gone. Four per run with the judge's proxy; two
    interrupted runs left eight of them on the disk.
    """
    proj = "zz-pytest__anon__env"
    name = f"{proj}-egress-proxy-1"
    anon = []
    try:
        _sh("docker", "run", "-d", "--name", name,
            "--label", f"com.docker.compose.project={proj}",
            "-v", "/var/log/squid", "-v", "/var/spool/squid",
            "alpine:3", "sleep", "600")
        anon = _sh("docker", "inspect", name, "--format",
                   "{{range .Mounts}}{{if eq .Type \"volume\"}}{{println .Name}}{{end}}{{end}}"
                   ).stdout.split()
        assert len(anon) == 2, anon
        assert all(len(v) == 64 for v in anon), "expected anonymous volumes"
        assert all(v in _volumes() for v in anon)

        out = _sh("bash", str(STOP_HARNESS))
        left = [v for v in anon if v in _volumes()]
        assert not left, f"anonymous volumes survived: {left}\n{out.stdout}{out.stderr}"
    finally:
        _sh("docker", "rm", "-f", name)
        for v in anon:
            _sh("docker", "volume", "rm", "-f", v)


@requires_docker
def test_a_named_volume_nobody_owns_is_left_alone():
    """Only anonymous volumes are taken on shape. A named one is somebody's."""
    vol = "zz-pytest-precious-data"
    try:
        _sh("docker", "volume", "create", vol)
        _sh("bash", str(STOP_HARNESS))
        assert vol in _volumes(), "swept a named volume that is not harbor's"
    finally:
        _sh("docker", "volume", "rm", "-f", vol)


@requires_docker
def test_dry_run_reports_the_volume_it_would_remove(fake_project):
    """The old filter reported nothing here, which is how the leak stayed hidden."""
    fake_project()
    out = _sh("bash", str(STOP_HARNESS), "--dry-run")
    assert VOLUME in out.stdout
    assert CONTAINER in _names(), "--dry-run removed something"
    assert VOLUME in _volumes(), "--dry-run removed something"


@requires_docker
def test_sigint_tears_down_the_run(tmp_path):
    """A real Ctrl-C: SIGINT to the process group, as a terminal delivers it."""
    src = RUN_TASK.read_text()
    helpers = src[src.index("# ----------------------------------------------------------------"
                            " teardown ---"):src.index("list_trial_dirs() {")]
    script = tmp_path / "interrupt.sh"
    script.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n" + helpers + f'''
trap on_interrupt INT
trap on_terminate TERM
snapshot_compose_projects
docker volume create --label com.docker.compose.project={PROJECT} {VOLUME} >/dev/null
docker run -d --name {CONTAINER} --label com.docker.compose.project={PROJECT} \\
    -v {VOLUME}:/data alpine:3 sleep 600 >/dev/null
echo READY
sleep 600
echo REACHED_THE_END
''')
    log = tmp_path / "out.txt"
    with log.open("w") as fh:
        p = subprocess.Popen(["bash", str(script)], stdout=fh,
                             stderr=subprocess.STDOUT, start_new_session=True)
        try:
            for _ in range(60):
                if CONTAINER in _names():
                    break
                time.sleep(1)
            assert CONTAINER in _names(), log.read_text()

            os.killpg(os.getpgid(p.pid), signal.SIGINT)
            rc = p.wait(timeout=60)
        finally:
            if p.poll() is None:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            _sh("docker", "rm", "-f", CONTAINER)
            _sh("docker", "volume", "rm", "-f", VOLUME)

    text = log.read_text()
    assert rc == 130, f"exit {rc}, expected 130\n{text}"
    assert "REACHED_THE_END" not in text, "walked on past the interrupt"
    assert CONTAINER not in _names(), text
    assert VOLUME not in _volumes(), text
