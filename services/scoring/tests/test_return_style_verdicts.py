"""A test's verdict is its return value when it returns one.

Both styles exist across the task bundles and both have to grade honestly:

    assert style   def test_x(): assert cond      -> verdict is the pytest outcome
    return style   def test_x(): return cond      -> verdict is bool(cond)

Reading only the pytest outcome silently mis-scores every return-style test,
because a function that returns instead of asserting always passes in pytest --
its only failure signals are a raised exception and a failed assert. That is
not a missing score but a fabricated one: every goal records as met and every
guard as tripped, whatever the agent did. The bundles are overwhelmingly
return-style (88 of 90 tests in one, 59 of 61 in another) and they run pytest
with `-W ignore::pytest.PytestReturnNotNoneWarning`, so nothing warned either.

These tests run a real pytest session over a fixture carrying both styles and
assert on what each plugin recorded, because the bug lived in the hook wiring
rather than in any function that could be called directly.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

SCORING = Path(__file__).resolve().parents[1]

# One of each shape the graders must tell apart. The guards matter most: a
# guard is negative-weighted, so recording a clean one as "tripped" subtracts
# a penalty the agent never earned.
FIXTURE = '''\
def test_goal_met():      return True
def test_goal_missed():   return False
def test_guard_clean():   return False
def test_guard_tripped(): return True
def test_assert_ok():     assert True
def test_assert_bad():    assert False
def test_raises():        raise RuntimeError("boom")
'''

WEIGHTS = {
    "components": {
        "traj_tests": {
            "weight": 1, "graded": True,
            "tests": {
                "test_goal_met": 5, "test_goal_missed": 5,
                "test_guard_clean": -5, "test_guard_tripped": -5,
                "test_assert_ok": 5, "test_assert_bad": 5, "test_raises": 5,
            },
        }
    }
}

EXPECTED = {
    "test_goal_met": True,
    "test_goal_missed": False,
    "test_guard_clean": False,
    "test_guard_tripped": True,
    "test_assert_ok": True,
    "test_assert_bad": False,
    "test_raises": False,
}


@pytest.fixture(scope="module")
def bundle(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("bundle")
    (d / "test_outputs.py").write_text(FIXTURE)
    (d / "test_weights.json").write_text(json.dumps(WEIGHTS))
    return d


def _run(bundle: Path, plugin: str, env: dict[str, str]) -> None:
    full = {"PYTHONPATH": f"{bundle}{os_sep()}{SCORING}", **env}
    subprocess.run(
        [sys.executable, "-m", "pytest", str(bundle / "test_outputs.py"),
         "-p", plugin, "-q", "-p", "no:cacheprovider",
         "-W", "ignore::pytest.PytestReturnNotNoneWarning"],
        cwd=bundle, env={**_base_env(), **full}, capture_output=True, check=False,
    )


def os_sep() -> str:
    import os
    return os.pathsep


def _base_env() -> dict[str, str]:
    import os
    return dict(os.environ)


def _collector_plugin(dest: Path) -> str:
    """Write the plugin weighted_judge embeds, so the test covers the shipped
    source rather than a copy that can drift away from it."""
    src = (SCORING / "weighted_judge.py").read_text()
    body = re.search(r"_COLLECTOR_PLUGIN = '''\\\n(.*?)\n'''", src, re.S)
    assert body, "could not find _COLLECTOR_PLUGIN in weighted_judge.py"
    (dest / "collector_plugin.py").write_text(body.group(1))
    return "collector_plugin"


def test_collector_records_the_return_value(bundle, tmp_path):
    """weighted_judge's collector -- the grader that actually scores a run."""
    plugin = _collector_plugin(bundle)
    out = tmp_path / "results.json"
    _run(bundle, plugin, {"MCPATLAS_GRADER_RESULTS": str(out)})
    assert json.loads(out.read_text()) == EXPECTED


def test_ctrf_reports_the_return_value(bundle, tmp_path):
    """ctrf.json must not read 7/7 for a run the grader scored 3/7."""
    out = tmp_path / "ctrf.json"
    _run(bundle, "ctrf_pytest_plugin",
         {"CTRF_OUT": str(out), "CTRF_WEIGHTS": str(bundle / "test_weights.json")})
    doc = json.loads(out.read_text())
    got = {t["name"]: t["status"] for t in doc["results"]["tests"]}
    assert got == {n: ("passed" if v else "failed") for n, v in EXPECTED.items()}
    summary = doc["results"]["summary"]
    assert (summary["passed"], summary["failed"]) == (3, 4)


def test_a_raising_test_is_a_failure_not_a_verdict(bundle, tmp_path):
    """A test that raised never reached its return, so the pytest outcome is
    the only truth available -- it must not be read as a missing verdict and
    quietly credited."""
    plugin = _collector_plugin(bundle)
    out = tmp_path / "results.json"
    _run(bundle, plugin, {"MCPATLAS_GRADER_RESULTS": str(out)})
    assert json.loads(out.read_text())["test_raises"] is False
