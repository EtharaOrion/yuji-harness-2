"""summary.json aggregates must reconcile with result.json's per-trial metrics.

The bug these guard: avg_completion_rate averaged the host-side traj_tests
value, on the assumption that result.json's per-trial completion_rate carried
the same quantity. It does not -- that field is the container's Rc (the
unweighted fraction of positive checks passed, written by the bundle's
test_write_reward_json), while traj_tests is a weight-normalised score. The two
never reconciled, and once traj_tests went unmeasured the rollup published 0.0
next to per-trial records of 0.9 and 0.714: the summary said the runs completed
nothing while the records beside it said otherwise.

Nothing else in the suite exercises the aggregation, so these are the only
tests standing between that bug and a repeat.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1]


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "harbor_to_output", _SCRIPTS.parent / "tools" / "delivery" / "harbor_to_output.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


h2o = _load_module()


def _load_delivery():
    spec = importlib.util.spec_from_file_location(
        "make_delivery", _SCRIPTS.parent / "tools" / "delivery" / "make_delivery.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


md = _load_delivery()


def _load_host_rubric():
    spec = importlib.util.spec_from_file_location(
        "host_rubric_pass", _SCRIPTS / "host_rubric_pass.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


hrp = _load_host_rubric()


def _build_job(
    tmp_path: Path, per_trial: list[dict], trial_reward: dict | None = None,
    agent_cost: float | None = None,
) -> tuple[Path, Path]:
    """A minimal Harbor job dir carrying `per_trial` metrics in result.json.

    `trial_reward`, when given, is written as each trial's verifier/reward.json --
    the container-side record the reshaper reads to derive the final reward.
    `agent_cost`, when given, becomes each trial's agent/trajectory.json
    final_metrics.total_cost_usd, uncut, as the agent SDK writes it.
    """
    job = tmp_path / "job"
    job.mkdir()
    (job / "config.json").write_text(
        json.dumps({"agents": [{"name": "claude-code", "model_name": "claude-opus-5"}]})
    )
    (job / "result.json").write_text(
        json.dumps({"id": "job-1", "stats": {"evals": {"e1": {"metrics": per_trial}}}})
    )
    for i in range(len(per_trial)):
        trial = job / f"trial_{i}"
        trial.mkdir()
        (trial / "config.json").write_text(
            json.dumps({"task": {"path": "tasks/demo", "name": "demo"}})
        )
        (trial / "result.json").write_text(json.dumps({"reward": 0.0}))
        if agent_cost is not None:
            agent = trial / "agent"
            agent.mkdir()
            (agent / "trajectory.json").write_text(json.dumps({
                "schema_version": "ATIF-v1.7", "steps": [],
                "final_metrics": {"total_cost_usd": agent_cost},
            }))
        if trial_reward is not None:
            verifier = trial / "verifier"
            verifier.mkdir()
            (verifier / "reward.json").write_text(json.dumps(trial_reward))
    out = tmp_path / "out"
    out.mkdir()
    return job, out


def _convert(tmp_path: Path, per_trial: list[dict]) -> tuple[dict, list[dict]]:
    job, out = _build_job(tmp_path, per_trial)
    written = h2o.convert_job(job, out, ks=[], run_offset=0)
    assert written, "convert_job produced no task output"
    task = written[0]
    summary = json.loads((task / "summary.json").read_text())
    result = json.loads((task / "result.json").read_text())
    recorded: list[dict] = []
    for eval_data in ((result.get("stats") or {}).get("evals") or {}).values():
        recorded.extend(eval_data.get("metrics") or [])
    return summary, recorded


@pytest.mark.parametrize(
    "aggregate,component",
    [("avg_completion_rate", "completion_rate"), ("avg_misbehave_rate", "misbehave_rate")],
)
def test_aggregate_reconciles_with_per_trial_metrics(tmp_path, aggregate, component):
    """The aggregate equals the mean of the values it summarises.

    This is the property the audit re-derives (crucible rollout.py,
    REWARD-COMPONENT-NOT-REDERIVABLE), computed the same way: over the trials
    that actually recorded the component. It reconciles to the precision the
    aggregate is published at -- every metric in the tree carries REWARD_DP
    places -- not to the float the mean happened to land on.
    """
    summary, recorded = _convert(
        tmp_path,
        [
            {"completion_rate": 0.9, "misbehave_rate": 0.0, "reward": 0.0, "scored": 1.0},
            {"completion_rate": 0.5, "misbehave_rate": 0.25, "reward": 0.0, "scored": 1.0},
        ],
    )
    values = [m[component] for m in recorded if component in m]
    rederived = sum(values) / len(values)
    reported = summary["metrics"][aggregate]
    assert reported is not None, f"{aggregate} reported as unmeasured despite recorded values"
    assert float(reported) == pytest.approx(round(rederived, h2o.REWARD_DP))


def test_aggregate_ignores_trials_missing_the_component(tmp_path):
    """A trial that recorded no completion_rate is skipped, not counted as zero.

    Averaging it in as 0.0 would drag the aggregate below what the audit
    re-derives, which skips absent components -- and would understate a run.
    """
    summary, recorded = _convert(
        tmp_path,
        [{"completion_rate": 0.8, "misbehave_rate": 0.0, "reward": 0.0}, {"reward": 0.0}],
    )
    assert summary["metrics"]["avg_completion_rate"] == pytest.approx(0.8)


def test_unmeasured_channel_reports_null_not_zero(tmp_path):
    """An unmeasured Channel A must not be published as a scored 0.0.

    A component that produced no value and one that scored zero are different
    facts; collapsing them is the confusion the reward ledger already refuses.
    """
    summary, _ = _convert(
        tmp_path, [{"completion_rate": 0.9, "misbehave_rate": 0.0, "reward": 0.0}]
    )
    assert summary["metrics"]["avg_traj_tests"] is None


def _convert_with_trial_reward(tmp_path: Path, trial_reward: dict) -> dict:
    """Reshape one trial carrying `trial_reward`, returning its run_1 detail.json."""
    job, out = _build_job(
        tmp_path, [{"completion_rate": 0.9, "misbehave_rate": 0.0, "reward": 0.0}], trial_reward
    )
    written = h2o.convert_job(job, out, ks=[], run_offset=0)
    assert written
    return json.loads((written[0] / "trajectory" / "run_1" / "verifier" / "detail.json").read_text())


def test_scored_zero_records_a_machine_readable_reason(tmp_path):
    """A zero reward must carry why, or it is indistinguishable from a crash.

    The bundle's test.sh records a reason only when Channel A never wrote. A
    *scored* zero -- suite ran, run earned nothing -- previously arrived with no
    explanation, which is what VER-UNATTRIBUTED-ZERO fires on.
    """
    detail = _convert_with_trial_reward(
        tmp_path, {"reward": 0.0, "completion_rate": 0.9, "misbehave_rate": 0.0, "scored": True}
    )
    assert detail.get("zero_reason"), "a scored zero reached detail.json with no recorded cause"


def test_container_supplied_zero_reason_is_not_overwritten(tmp_path):
    """The container knows why it failed; this layer does not. Carry it through."""
    supplied = "unscored: reward_channel_a.json was never written"
    detail = _convert_with_trial_reward(
        tmp_path,
        {"reward": 0.0, "completion_rate": 0.0, "misbehave_rate": 0.0,
         "scored": False, "zero_reason": supplied},
    )
    assert detail["zero_reason"] == supplied


def test_nonzero_reward_carries_no_zero_reason(tmp_path):
    """Only a zero needs explaining; a scored run must not gain a spurious field.

    `producer` is load-bearing and was not always required. harbor_to_output.py
    :672-679 trusts a reward only from "host_rubric_pass" or "container_test";
    anything else is warned about and forced to reward=0 as unscored. Without it
    the 0.75 below is zeroed, the zero_reason guard fires correctly, and this
    test fails while appearing to be about zero_reason rather than about a
    reward that never survived. The sibling tests hide it by supplying 0.0.
    """
    detail = _convert_with_trial_reward(
        tmp_path, {"reward": 0.75, "completion_rate": 0.9, "misbehave_rate": 0.0,
                   "scored": True, "producer": "host_rubric_pass"}
    )
    assert "zero_reason" not in detail


def test_mean_or_none_distinguishes_unmeasured_from_zero():
    assert h2o._mean_or_none([]) is None
    assert h2o._mean_or_none([None, None]) is None
    assert h2o._mean_or_none([0.0]) == 0.0
    assert h2o._mean_or_none([0.9, 0.5]) == pytest.approx(0.7)
    # The old helper's behaviour, kept for counts where zero is the truth.
    assert h2o._mean([]) == 0.0


def test_published_reward_json_is_cut_to_the_trees_precision(tmp_path):
    """Every number in the published tree reads at REWARD_DP, truncated.

    reward.json is the one that kept slipping: the bundle's container-side
    test.sh writes raw float arithmetic, so a reward arrived as
    0.010999785519502404 and the rates as six-place ratios, sitting next to a
    result.json that reported the same quantities at two. Bundles are task
    content, so the precision is settled at the publish boundary instead.

    0.267442 must publish as 0.26, not 0.27. Rounding is the one operation here
    that can make a score larger than the measurement supports; truncation only
    ever understates, which is the safe direction for a number an auditor
    re-derives.
    """
    job, out = _build_job(
        tmp_path,
        [{"completion_rate": 0.9, "misbehave_rate": 0.0, "reward": 0.0}],
        {"reward": 0.010999785519502404, "completion_rate": 0.290698,
         "misbehave_rate": 0.267442, "scored": True},
    )
    written = h2o.convert_job(job, out, ks=[], run_offset=0)
    assert written
    rw = json.loads(
        (written[0] / "trajectory" / "run_1" / "verifier" / "reward.json").read_text()
    )
    for k in ("reward", "completion_rate", "misbehave_rate"):
        assert rw[k] == h2o.norm_reward(rw[k]), \
            f"{k} published at more than {h2o.REWARD_DP} places: {rw[k]}"
    assert rw["completion_rate"] == 0.29
    assert rw["misbehave_rate"] == 0.26, "0.267442 was rounded up, not cut"


def test_norm_reward_cuts_rather_than_rounds():
    """The precision policy itself, stated once.

    The 0.29 case is the one that makes the implementation non-obvious: the
    float nearest 0.29 is 0.28999999999999998, so `int(v * 100) / 100` yields
    0.28 and silently loses a hundredth off every second value.
    """
    assert h2o.norm_reward(0.267442) == 0.26
    assert h2o.norm_reward(0.348837) == 0.34
    assert h2o.norm_reward(0.999) == 0.99
    assert h2o.norm_reward(0.29) == 0.29
    assert h2o.norm_reward(16.0) == 16.0
    # Truncation is toward zero, so a negative is not driven further from it.
    assert h2o.norm_reward(-0.267) == -0.26
    # Non-numeric and bool are still passed through untouched.
    assert h2o.norm_reward(None) is None
    assert h2o.norm_reward(True) is True


def test_result_json_keeps_harbors_pass_at_k_schema(tmp_path):
    """result.json belongs to Harbor, and Harbor re-reads it on every startup.

    `harbor.models.job.result.AgentDatasetStats.pass_at_k` is `dict[int, float]`
    and `Job.__init__` validates any existing result.json before it will start a
    trial. We used to write our own "k=<k>" labels into that field, so the first
    run left behind a job dir no later run could open -- harbor died in pydantic
    with `unable to parse string as an integer` before reaching the agent. The
    labels are ours to keep, but only in the files we own.
    """
    job, out = _build_job(tmp_path, [{"completion_rate": 0.9, "misbehave_rate": 0.0,
                                      "reward": 0.0}])
    written = h2o.convert_job(job, out, ks=[], run_offset=0)
    assert written
    task = written[0]

    result = json.loads((task / "result.json").read_text())
    for eval_data in ((result.get("stats") or {}).get("evals") or {}).values():
        for key in eval_data.get("pass_at_k") or {}:
            assert int(key) >= 1, f"harbor cannot parse pass_at_k key {key!r} as an int"

    # ...and the labelled "k=<k>" form still reaches the .raw summary, which is
    # now its only home -- per_task must not carry it, since a probability
    # keyed 1..N sitting in a per-task block reads as a per-run score.
    passk = json.loads(next(task.glob("pass@*.json")).read_text())
    entry = passk["per_task"][0]
    assert "pass@k" not in entry
    raw_summary = json.loads((task / ".raw").glob("trials_*/summary.json").__next__().read_text())
    assert all(k.startswith("k=") for k in raw_summary["metrics"]["pass@k"])

    # The per-attempt scores ride in both places the layout calls for -- top
    # level and inside the per_task entry -- keyed "pass@<n>", and the two
    # copies are the same dict, so they cannot drift. These are per-trial
    # REWARDS despite the label; the pass@k probabilities that share the name
    # live in the .raw summary keyed "k=<k>", which is what keeps the two
    # metrics from ever appearing identically keyed in one file.
    per_trial = passk["per_trial_rewards"]
    assert entry["per_trial_rewards"] == per_trial
    assert list(per_trial) == [f"pass@{i}" for i in range(1, len(per_trial) + 1)]
    assert list(per_trial.values()) == [e["reward"] for e in raw_summary["attempts"]]


def test_cost_fields_are_truncated_to_the_trees_precision(tmp_path):
    """Costs reach the tree through copied files, bypassing norm_result_metrics.

    total_cost_usd arrives as raw float arithmetic from the agent SDK and
    judge_cost_usd from rubric_judge_cli's pricing, so before norm_metrics_file
    the published dir showed 6.550475500000003 beside rewards already cut to
    two places. Truncation, not rounding: 8.5681125 publishes as 8.56.
    """
    doc = {
        "final_metrics": {"total_cost_usd": 6.550475500000003,
                          "usage": {"cost_usd": 8.5681125}},
        "lines": [{"judge_cost_usd": 0.404303, "completion_rate": 0.395349}],
        # A measurement-shaped number that is NOT a measurement. The task this
        # guards is about decimal conversion, so its rubric quotes figures at
        # three places; keying on field names rather than rounding every float
        # is what keeps them readable.
        "criterion": "reads the fee as the decimal 7849.186",
        "steps": 3,
        "ok": True,
    }
    path = tmp_path / "trajectory.json"
    path.write_text(json.dumps(doc))
    h2o.norm_metrics_file(path)
    out = json.loads(path.read_text())

    assert out["final_metrics"]["total_cost_usd"] == 6.55
    assert out["final_metrics"]["usage"]["cost_usd"] == 8.56   # not 8.57
    assert out["lines"][0]["judge_cost_usd"] == 0.4
    assert out["lines"][0]["completion_rate"] == 0.39          # not 0.4
    assert out["criterion"] == "reads the fee as the decimal 7849.186"
    assert out["steps"] == 3 and out["ok"] is True


def test_norm_metrics_file_leaves_unreadable_files_alone(tmp_path):
    """Precision is cosmetic; it must never cost the artifact itself."""
    path = tmp_path / "not.json"
    path.write_text("not json at all")
    h2o.norm_metrics_file(path)
    assert path.read_text() == "not json at all"
    h2o.norm_metrics_file(tmp_path / "absent.json")  # must not raise


def test_norm_metrics_file_does_not_touch_files_without_metrics(tmp_path):
    """No metric field -> no write, byte for byte.

    This runs over every published JSON, and most carry no measurement at all.
    Rewriting them would reflow Harbor-owned documents (artifacts/manifest.json
    is provenance, kept exactly as written) and bury the files whose numbers
    actually changed under a diff of pure whitespace churn.
    """
    original = '{"collected":["a.csv"],  "steps":3,\n   "nested":{"ok":true}}'
    path = tmp_path / "manifest.json"
    path.write_text(original)
    h2o.norm_metrics_file(path)
    assert path.read_text() == original


def test_ledger_component_values_are_truncated(tmp_path):
    """reward_channel_a.json holds its numbers under the generic key "value".

    The ledger shape is {"weight": w, "value": v}, so matching the PAIR is what
    identifies a measurement -- "value" alone is too common a name to treat as
    one wherever it appears.
    """
    doc = {"reward": 0.57, "rubric": 0.5716,
           "ledger": {"traj_tests": {"status": "unscored", "weight": 3.59, "value": None},
                      "rubric": {"status": "scored", "weight": 4.0, "value": 0.5716}},
           "config": {"value": 0.123456}}      # no sibling weight -> not a metric
    path = tmp_path / "reward_channel_a.json"
    path.write_text(json.dumps(doc))
    h2o.norm_metrics_file(path)
    out = json.loads(path.read_text())

    assert out["rubric"] == 0.57
    assert out["ledger"]["rubric"]["value"] == 0.57
    assert out["ledger"]["traj_tests"]["value"] is None      # unscored stays null
    assert out["ledger"]["traj_tests"]["weight"] == 3.59
    assert out["config"]["value"] == 0.123456


def test_result_json_metrics_are_cut_without_the_rebuild_branch(tmp_path):
    """norm_result_metrics must not depend on the reward_stats rebuild.

    That rebuild only runs when the job has both evals and parsed episodes.
    On any path where it is skipped, Harbor's raw numbers -- reward as
    0.016499678279253607, and reward_stats keyed by the stringified rate
    "0.302326" -- reached the published result.json unchanged.
    """
    res = {
        "stats": {
            "cost_usd": 6.7969325000000005,
            "evals": {
                "some__eval": {
                    "metrics": [{"completion_rate": 0.302326,
                                 "misbehave_rate": 0.267442,
                                 "reward": 0.016499678279253607}],
                    "reward_stats": {
                        "reward": {"0.016499678279253607": ["run_1"]},
                        "completion_rate": {"0.302326": ["run_1"],
                                            "0.309999": ["run_2"]},
                    },
                }
            },
        }
    }
    out = h2o.norm_result_metrics(res)
    ev = out["stats"]["evals"]["some__eval"]

    assert out["stats"]["cost_usd"] == 6.79        # truncated, not 6.80
    assert ev["metrics"][0]["reward"] == 0.01
    assert ev["metrics"][0]["completion_rate"] == 0.3
    assert ev["metrics"][0]["misbehave_rate"] == 0.26
    assert ev["reward_stats"]["reward"] == {"0.01": ["run_1"]}
    # 0.302326 and 0.309999 both cut to 0.3 -- the grouping merges, and no
    # trial name is lost to the collision.
    assert ev["reward_stats"]["completion_rate"] == {"0.3": ["run_1", "run_2"]}


def test_reward_stats_non_numeric_labels_survive(tmp_path):
    """A grouping key that is not a number is a label, not a measurement."""
    res = {"stats": {"evals": {"e": {"reward_stats": {"reward": {"unscored": ["run_3"]}}}}}}
    out = h2o.norm_result_metrics(res)
    assert out["stats"]["evals"]["e"]["reward_stats"]["reward"] == {"unscored": ["run_3"]}


def test_raw_mirror_keeps_full_precision_when_publishing_in_place(tmp_path):
    """.raw must hold the uncut number when the trial dir IS the run dir.

    run_task.sh re-publishes an output dir over itself, so trial_dir/agent and
    out_task/trajectory/run_N/agent resolve to the same path. _copy then hits
    its src == dst short-circuit and returns WITHOUT copying, which makes the
    "published" file Harbor's own. Cutting it before the .raw staging therefore
    truncated the mirror too, and 6.7969325 existed nowhere on disk.

    The separate-directory fixture above cannot catch this -- there _copy makes
    a real copy and the two files are independent -- so this test wires the
    in-place geometry explicitly.
    """
    out_task = tmp_path / "task"
    trial_dir = out_task / "trajectory" / "run_1"        # trial dir IS the run dir
    (trial_dir / "agent").mkdir(parents=True)
    (trial_dir / "verifier").mkdir(parents=True)
    (trial_dir / "result.json").write_text(json.dumps({"id": "t1", "reward": 0.0}))
    (trial_dir / "config.json").write_text(
        json.dumps({"task": {"path": "tasks/demo", "name": "demo"}}))
    (trial_dir / "agent" / "trajectory.json").write_text(json.dumps({
        "schema_version": "ATIF-v1.7", "steps": [],
        "final_metrics": {"total_cost_usd": 6.7969325000000005},
    }))

    ag_file = trial_dir / "agent" / "trajectory.json"
    run_file = out_task / "trajectory" / "run_1" / "agent" / "trajectory.json"
    assert ag_file.resolve() == run_file.resolve(), "fixture is not in-place"

    raw_trials = out_task / ".raw" / "trials_demo"
    raw_trials.mkdir(parents=True)
    h2o.reshape_trial(trial_dir, 1, out_task=out_task, raw_trials=raw_trials,
                      model="m", task_name="demo", task_dir=None, job_id="j1")

    def cost(p):
        return (json.loads(p.read_text()).get("final_metrics") or {}).get("total_cost_usd")

    mirrored = raw_trials / "trajectories" / "m" / "run_1" / "agent" / "trajectory.json"
    assert mirrored.exists(), "no .raw mirror staged"
    assert cost(run_file) == 6.79                    # published: cut, truncated
    assert cost(mirrored) == 6.7969325000000005      # mirror: every digit kept


# ---------------------------------------------------------------------------
# Host-local path masking (make_delivery.mask_local_paths)
#
# Harbor writes the argv it ran with into config.json and quotes the whole
# failing `docker compose ... -f <path>` line into exception.txt, result.json
# and job.log. Every one of those carries the operator's home directory. The
# mask used to anchor only on output/tasks/jobs/input/delivery_output, so a
# path into any OTHER repo directory shipped intact -- `--extra-docker-compose`
# appeared as `~/.../harness/tools/network/egress-proxy/overlay.yaml` in every
# published config.json. These pin what must be cut and, just as importantly,
# what must survive.
# ---------------------------------------------------------------------------

def _mask(text: str, repo="/Users/operator/work/harness", monkeypatch=None) -> str:
    monkeypatch.setattr(md, "REPO_ROOT", Path(repo))
    return md.mask_local_paths(text)


@pytest.mark.parametrize("raw,want", [
    # The regression: a repo path with no delivery anchor in it.
    ("/Users/operator/work/harness/tools/network/egress-proxy/overlay.yaml",
     "tools/network/egress-proxy/overlay.yaml"),
    # Harbor collapses $HOME to `~` when it serialises argv, so the same file
    # reaches disk under two spellings and both have to go.
    ("~/work/harness/tools/network/egress-proxy/overlay.yaml",
     "tools/network/egress-proxy/overlay.yaml"),
    # Still handled by the original anchored rule.
    ("/Users/operator/work/harness/output/t/trajectory/run_1",
     "output/t/trajectory/run_1"),
    # A bare repo root is a cwd or a --project-directory: emptying it would
    # leave a broken value.
    ("cwd=/Users/operator/work/harness", "cwd=."),
    # macOS mkdtemp: the folder hash is derived from the uid.
    ("-f /private/var/folders/1b/d687yx7j39l_8_smp0000gn/T/tmpu3n/x.json",
     "-f $TMPDIR/tmpu3n/x.json"),
    ("/var/folders/1b/d687yx7j39l_8_smp0000gn/T/tmp6/y.json",
     "$TMPDIR/tmp6/y.json"),
    # Home-rooted but outside the repo: head replaced, tail kept.
    ("/Users/someone/.claude/settings.json", "~/.claude/settings.json"),
    ("/home/runner/.local/share/uv/tools/harbor", "~/.local/share/uv/tools/harbor"),
])
def test_mask_cuts_host_paths(raw, want, monkeypatch):
    assert _mask(raw, monkeypatch=monkeypatch) == want


@pytest.mark.parametrize("keep", [
    "/app/solution.docx",          # inside the container: real and reproducible
    "/logs/agent/agent.log",
    "tasks/some-task/environment/docker-compose.yaml",   # already relative
    "output/some-task/trajectory/run_1",
])
def test_mask_leaves_container_and_relative_paths_alone(keep, monkeypatch):
    assert _mask(keep, monkeypatch=monkeypatch) == keep


def test_mask_keeps_json_parseable_and_is_idempotent(monkeypatch):
    doc = {"environment": {"extra_docker_compose": [
        "/Users/operator/work/harness/tools/network/egress-proxy/overlay.yaml"]}}
    once = _mask(json.dumps(doc, indent=4), monkeypatch=monkeypatch)
    assert json.loads(once)["environment"]["extra_docker_compose"] == [
        "tools/network/egress-proxy/overlay.yaml"]
    # The sweep runs again at the end of finance over a tree reshape already
    # masked; a second pass must be a no-op.
    assert _mask(once, monkeypatch=monkeypatch) == once


def test_mask_tree_rewrites_text_and_skips_binaries(tmp_path, monkeypatch):
    monkeypatch.setattr(md, "REPO_ROOT", Path("/Users/operator/work/harness"))
    run = tmp_path / "trajectory" / "run_1"
    (run / "artifacts").mkdir(parents=True)
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"f": "/Users/operator/work/harness/tools/x.yaml"}))
    exc = run / "exception.txt"
    exc.write_text("RuntimeError: docker compose -f ~/work/harness/tools/x.yaml\n")
    # An agent artifact that happens to contain the byte pattern. Rewriting it
    # would change what the run is claimed to have produced.
    art = run / "artifacts" / "report.docx"
    art.write_bytes(b"/Users/operator/work/harness/tools/x.yaml")

    changed = md.mask_tree(tmp_path)

    assert set(changed) == {cfg, exc}
    assert json.loads(cfg.read_text())["f"] == "tools/x.yaml"
    assert "/Users/" not in exc.read_text() and "~/work" not in exc.read_text()
    assert art.read_bytes() == b"/Users/operator/work/harness/tools/x.yaml"


def test_only_trials_limits_what_becomes_a_run(tmp_path):
    """Stale trial dirs in a job dir must not each become a trajectory/run_N.

    Harbor leaves the directory of any trial that died behind, and the glob that
    selects trials cannot tell those from the one the current invocation just
    produced -- so a single N=1 run emitted several runs, each counted as an
    attempt by the aggregates above. run_task.sh names the trials it owns.
    """
    job, out = _build_job(tmp_path, [{"reward": 0.0}, {"reward": 0.0}])
    written = h2o.convert_job(job, out, ks=[], run_offset=0, only_trials={"trial_1"})
    runs = sorted((written[0] / "trajectory").glob("run_*"))
    assert [r.name for r in runs] == ["run_1"]


def test_no_filter_still_converts_every_trial(tmp_path):
    job, out = _build_job(tmp_path, [{"reward": 0.0}, {"reward": 0.0}])
    written = h2o.convert_job(job, out, ks=[], run_offset=0)
    runs = sorted((written[0] / "trajectory").glob("run_*"))
    assert [r.name for r in runs] == ["run_1", "run_2"]


def test_an_empty_only_trials_converts_nothing(tmp_path):
    """"harbor made no trial" must not decay into "convert whatever is lying
    around" -- that is the extra-runs bug with an extra step."""
    job, out = _build_job(tmp_path, [{"reward": 0.0}])
    assert h2o.convert_job(job, out, ks=[], run_offset=0, only_trials=set()) == []


def test_judge_container_reward_publishes_like_the_host_ledger(tmp_path):
    """producer=judge_container is stamped by a bundle whose own reward already
    includes the rubric its judge container graded. It must publish exactly as a
    host-pass ledger does -- rescaled and scored -- not be zeroed as unscored,
    which is what any producer the reshaper does not know gets."""
    published = {}
    for producer in ("host_rubric_pass", "judge_container"):
        root = tmp_path / producer
        root.mkdir()
        job, out = _build_job(
            root, [{"completion_rate": 0.9, "misbehave_rate": 0.0, "reward": 0.0}],
            {"reward": 0.75, "completion_rate": 0.9, "misbehave_rate": 0.0, "producer": producer},
        )
        written = h2o.convert_job(job, out, ks=[], run_offset=0)
        assert written
        published[producer] = json.loads(
            (written[0] / "trajectory" / "run_1" / "verifier" / "reward.json").read_text())
    assert published["judge_container"]["reward"] == published["host_rubric_pass"]["reward"] == 75.0
    assert published["judge_container"].get("producer") == "judge_container"


def test_the_judge_markers_do_not_ship_in_the_published_run(tmp_path):
    """The published tree keeps the shape it had before the judge container.

    judge_container.json, reward_producer.json and judge-container-test.txt are
    read while the run is still going -- the in-container check during the
    verifier, then adopt_container_reward and host_rubric_pass in
    stage_host_rubric, which runs before this converter. What has to outlive the
    run is already folded in: reward.json keeps `producer`, the judge's usage
    stays in judge_tokens.json.
    """
    job, out = _build_job(
        tmp_path, [{"completion_rate": 0.9, "misbehave_rate": 0.0, "reward": 0.0}],
        {"reward": 0.42, "completion_rate": 0.9, "misbehave_rate": 0.0,
         "producer": "judge_container"},
    )
    ver = job / "trial_0" / "verifier"
    for name in h2o.JUDGE_MARKERS:
        (ver / name).write_text("{}")
    written = h2o.convert_job(job, out, ks=[], run_offset=0)
    assert written
    published = written[0] / "trajectory" / "run_1" / "verifier"
    for name in h2o.JUDGE_MARKERS:
        assert not (published / name).exists(), f"{name} shipped in the published run"
    reward = json.loads((published / "reward.json").read_text())
    assert reward["producer"] == "judge_container", "the label itself must survive"
