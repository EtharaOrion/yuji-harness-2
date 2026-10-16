"""The two registered entry points: inference and a separate evaluation.

Both delegate. Inference is scripts/run_task.sh; evaluation is
`harbor trial regrade`, which scores a recorded trial with the bundle's own
verifier in a separate verifier environment.
"""
from __future__ import annotations

import importlib.util
import sys
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _cli():
    spec = importlib.util.spec_from_file_location("harness_cli", REPO / "tools" / "cli.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cli = _cli()


def test_inference_and_evaluation_are_separate_registered_scripts():
    scripts = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]["scripts"]
    assert scripts == {"harness-infer": "tools.cli:infer", "harness-eval": "tools.cli:evaluate"}


def test_every_project_dependency_is_pinned_and_locked():
    project = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]
    assert all("==" in dep for dep in project["dependencies"]), project["dependencies"]
    assert (REPO / "uv.lock").is_file()


def test_eval_refuses_a_dir_that_is_not_a_task(tmp_path, capsys):
    assert cli.evaluate([str(tmp_path), str(tmp_path)]) == 2
    assert "no task.toml" in capsys.readouterr().err


def test_eval_hands_the_bundle_to_harbor_regrade(tmp_path, monkeypatch):
    task = tmp_path / "task"
    task.mkdir()
    (task / "task.toml").write_text("")
    trial = tmp_path / "trial"
    trial.mkdir()
    seen = {}
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/bin/harbor")
    monkeypatch.setattr(cli.os, "execv", lambda path, argv: seen.update(argv=argv))
    cli.evaluate([str(trial), str(task), "--", "--trials-dir", "/x"])
    assert seen["argv"] == ["/bin/harbor", "trial", "regrade", str(trial),
                            "--task-path", str(task), "--trials-dir", "/x"]


def test_infer_hands_its_arguments_to_run_task(monkeypatch):
    seen = {}
    monkeypatch.setattr(cli.os, "execv", lambda path, argv: seen.update(path=path, argv=argv))
    cli.infer(["--stage", "harbor", "tasks/x"])
    assert seen["path"] == str(REPO / "scripts" / "run_task.sh")
    assert seen["argv"][1:] == ["--stage", "harbor", "tasks/x"]
