"""Did the rubric get graded inside the judge container?

Report-only. A bundle's tests/test.sh runs this file as its last step, inside
`main`, in a pytest process of its own, and prints the result. It is
deliberately NOT in test_outputs.py and NOT in test_weights.json: a judge that
could not run is an infrastructure fault, not something the agent did, so it
must never move the agent's score or its failure class. When it fails,
scripts/run_task.sh grades the rubric on the host instead and the run is still
scored.

The harness owns this file, shared by every bundle, and the judge service mounts
it with the rest of services/scoring:

    python3 -m pytest /harness/scoring/tests/test_judge_container.py

It asserts on paths that exist only in a trial container, so the repo's own
suites skip it (see --ignore in the Makefile and .github/workflows).

Reads what services/scoring/judge_client.py leaves in /logs/verifier.
"""
import json
import os
from pathlib import Path

import pytest

LOGS = Path(os.environ.get("JUDGE_LOGS_DIR", "/logs/verifier"))
EXPECTED_MODEL = os.environ.get("JUDGE_MODEL", "gpt-5.6-sol")

# Skip itself outside a trial, rather than relying on every caller to pass
# --ignore. The Makefile and python-tests.yml do; scripts/smoke_test.py does
# not, so it has been reporting four failures on a clean checkout. A rule kept
# in three callers and enforced by none of them drifts the moment a fourth
# appears.
#
# The condition cannot hide a real fault: tests/test.sh does `mkdir -p
# /logs/verifier` before it runs this, harbor mounts that directory into both
# containers, and codexbridge refuses to grade at all when it is not writable.
# In a trial the directory is always there; in a checkout it never is.
if not LOGS.is_dir():
    pytest.skip(f"{LOGS} is absent: this file asserts on a trial container and "
                "is run there by tests/test.sh, not by the repo's suites",
                allow_module_level=True)


def _json(name):
    path = LOGS / name
    assert path.is_file(), f"{path} missing -- step 3 never reached the judge client"
    return json.loads(path.read_text())


def test_rubric_was_graded_in_the_judge_container():
    doc = _json("judge_container.json")
    assert doc.get("ok") is True, f"judge container did not grade: {doc.get('reason')}"
    assert doc.get("graded_in") == "judge-container", doc


def test_graded_by_the_pinned_model():
    doc = _json("judge_container.json")
    assert doc.get("model") == EXPECTED_MODEL, doc


def test_verdicts_were_written():
    rb = _json("rubric_breakdown.json")
    rows = rb.get("per_criterion") or rb.get("results") or []
    assert rows, "rubric_breakdown.json holds no verdicts"


def test_judge_usage_was_recorded():
    usage = _json("judge_tokens.json")
    lines = usage if isinstance(usage, list) else [usage]
    assert lines, "judge_tokens.json is empty"
    assert all(line.get("model_name") == EXPECTED_MODEL for line in lines), lines
