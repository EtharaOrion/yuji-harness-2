"""FAILURE_CLASSES must name exactly what classify_failure can return.

The histogram in pass@N.json is seeded from FAILURE_CLASSES, so the tuple is
the published shape of that document. Nothing at runtime compares it against
classify_failure -- a class added to the function and not to the tuple would
still be counted (the increment uses .get), it would just never appear at 0,
which is the absent-vs-zero ambiguity the seeding exists to remove. This test
is the only thing holding the two together.
"""
import ast
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
SOURCE = REPO / "tools" / "delivery" / "harbor_to_output.py"


def _returned_classes() -> set[str]:
    """The first element of every `return "...", ...` in classify_failure."""
    tree = ast.parse(SOURCE.read_text())
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "classify_failure")
    found = set()
    for node in ast.walk(fn):
        if not isinstance(node, ast.Return) or not isinstance(node.value, ast.Tuple):
            continue
        head = node.value.elts[0]
        assert isinstance(head, ast.Constant) and isinstance(head.value, str), (
            f"classify_failure returns a non-literal class at line {node.lineno}; "
            "the tuple can no longer be checked statically"
        )
        found.add(head.value)
    return found


def _declared_classes() -> tuple[str, ...]:
    tree = ast.parse(SOURCE.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "FAILURE_CLASSES" for t in node.targets
        ):
            return tuple(e.value for e in node.value.elts)
    raise AssertionError("FAILURE_CLASSES not found in harbor_to_output.py")


def test_declared_matches_returned():
    declared, returned = _declared_classes(), _returned_classes()
    assert set(declared) == returned, (
        f"missing from FAILURE_CLASSES: {sorted(returned - set(declared))}; "
        f"declared but unreachable: {sorted(set(declared) - returned)}"
    )


def test_no_duplicates():
    declared = _declared_classes()
    assert len(declared) == len(set(declared))


# --- a run that lost the model API ------------------------------------------
# Measured: a headroom proxy container was OOM-killed mid-session, docker
# stopped resolving its name, and Claude Code ended with "API Error: Can't reach
# the API server (EAI_AGAIN)". Harbor recorded no exception, the agent had made
# tool calls, so every rule below `passed` matched and the trial published a
# reward of 0 -- an infrastructure death scored as a bad answer.

import importlib.util
import json


def _module():
    spec = importlib.util.spec_from_file_location("harbor_to_output", SOURCE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_LOST_API = "API Error: Can't reach the API server — check your internet or DNS (EAI_AGAIN)"


def _stream(**over):
    base = {"valid": 2, "trace": [{"tool": "Bash"}], "transport_error": None}
    base.update(over)
    return base


def test_a_run_that_lost_the_api_is_infrastructure():
    cls, reason = _module().classify_failure(
        False, [{"name": "t1", "outcome": "missed"}], [{"number": 1, "outcome": "missed"}],
        _stream(transport_error=_LOST_API), None, rubric_expected=True)
    assert cls == "infrastructure", (cls, reason)
    assert "EAI_AGAIN" in reason


def test_a_finished_run_is_still_judged_on_its_answer():
    cls, _ = _module().classify_failure(
        False, [{"name": "t1", "outcome": "missed"}], [{"number": 1, "outcome": "missed"}],
        _stream(), None, rubric_expected=True)
    assert cls != "infrastructure", "a run that reached the end is not an outage"


def _write_stream(tmp_path, result_event):
    p = tmp_path / "claude-code.jsonl"
    p.write_text("\n".join(json.dumps(e) for e in [
        {"type": "assistant", "message": {"content": [
            {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "ls"}}]}},
        result_event,
    ]) + "\n")
    return p


def test_the_parser_names_a_lost_api(tmp_path):
    """Claude Code reports this with subtype "success", so termination_reason
    cannot carry it: the subtype wins and the run reads as a normal finish."""
    stream = _module().parse_stream(_write_stream(tmp_path, {
        "type": "result", "subtype": "success", "is_error": True, "result": _LOST_API}))
    assert stream["transport_error"], stream.get("termination_reason")


def test_the_parser_leaves_a_normal_finish_alone(tmp_path):
    stream = _module().parse_stream(_write_stream(tmp_path, {
        "type": "result", "subtype": "success", "is_error": False,
        "result": "Done. The API totals are in /workspace/out.csv."}))
    assert stream["transport_error"] is None


_LOST_API_400 = "API Error: 400 Tool reference 'tool_search_tool_regex' not found in available tools"


def test_an_api_error_that_is_not_a_transport_failure_is_still_infrastructure(tmp_path):
    """The two runs the first version of this rule missed.

    It listed transport wordings (EAI_AGAIN, ECONNREFUSED), so a 400 caused by a
    proxy rewriting the request read as a normal finish and published a zero.
    """
    stream = _module().parse_stream(_write_stream(tmp_path, {
        "type": "result", "subtype": "success", "is_error": True, "result": _LOST_API_400}))
    assert stream["transport_error"], stream.get("termination_reason")
    cls, reason = _module().classify_failure(
        False, [{"name": "t1", "outcome": "missed"}], [{"number": 1, "outcome": "missed"}],
        _stream(transport_error=_LOST_API_400), None, rubric_expected=True)
    assert cls == "infrastructure", (cls, reason)


def test_an_answer_that_merely_mentions_an_api_error_is_not_infrastructure(tmp_path):
    """The agent describing an error it handled is a finished run, not an outage."""
    stream = _module().parse_stream(_write_stream(tmp_path, {
        "type": "result", "subtype": "success", "is_error": False,
        "result": "The vendor's API Error field was blank, so I used the ledger total instead."}))
    assert stream["transport_error"] is None
