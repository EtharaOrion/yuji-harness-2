from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from harbor_to_output import bundle_dir_names

# tools/delivery/make_delivery.py -> the checkout root. Every path this
# file masks is measured against it.
REPO_ROOT = _Path(__file__).resolve().parents[2]

# Host-local path masking, inline. Harbor stamps absolute host paths
# (trial_uri, trials_dir, jobs_dir, tracebacks) into bookkeeping files, which
# is how `/Users/<name>/...` strings end up in shipped bundles. _HOST_PATH_RE
# below cuts them; container paths (/workspace, /logs, /tmp) survive verbatim.
_HOME_RE = re.compile(r"(?:file://)?/(?:Users|home)/[^/\s\"'\\]+")

# macOS mkdtemp: /var/folders/<2 chars>/<hash>/T. The hash is derived from the
# uid, so it identifies the operator's account as surely as their name does.
# Harbor writes these into the `docker compose -f ...` line it quotes back in
# every environment-build failure, which is how they reach exception.txt,
# result.json and job.log. /private is the real path, /var the symlink, and
# both spellings turn up in the same message.
_TMPDIR_RE = re.compile(r"(?:/private)?/var/folders/[^/\s\"']+/[^/\s\"']+/[A-Z]")

# Any host path, wherever it was run from: a home or volume root (macOS, Linux,
# Windows, WSL, `~`, file://) followed by its directories, which may hold spaces
# but never start or end with one. Everything up to the last directory a bundle
# points into (_KEEP_FROM) is cut; with none, only the final name survives.
_SEG = r"[^/\\\"'`<>|\s](?:[^/\\\"'`<>|\r\n\t]*[^/\\\"'`<>|\s])?"
_SEP = r"(?:\\\\|\\|/)"
_HOST_PATH_RE = re.compile(
    r"(?<![\w.:/\\~-])(?:file://)?(?:"
    rf"/(?:Users|home|media){_SEP}{_SEG}|/Volumes{_SEP}{_SEG}|/mnt{_SEP}[A-Za-z](?=[/\\])"
    rf"|[A-Za-z]:{_SEP}(?:Users|home|Documents and Settings){_SEP}{_SEG}|~(?=[/\\])"
    rf")(?P<rest>(?:{_SEP}{_SEG})*)"
)
_PART_RE = re.compile(rf"{_SEP}({_SEG})")
_KEEP_FROM = {"output", "delivery_output", "tasks", "jobs", "input", "tools", "scripts", "services"}


def _host_path(m: re.Match) -> str:
    parts = list(_PART_RE.finditer(m.group("rest")))
    if not parts:
        return "~"
    keep = next((p for p in reversed(parts) if p.group(1) in _KEEP_FROM), parts[-1])
    return m.group("rest")[keep.start(1):]


# Harbor's path fields carry the operator's directories whatever machine ran
# them. A value holding a task uuid becomes that uuid alone; a trial name only
# carries a truncated uuid, so it never matches. trials_dir and trial_uri name
# the job, and a task may be run from a folder of any name, so a job field, or
# a path that is a host path, without a uuid of its own takes the task uuid of
# the folder the file sits in. Container and app paths ("/workspace/x") stay.
_UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
_FIELD_RE = re.compile(r'("(path|trials_dir|trial_uri|jobs_dir)"\s*:\s*")([^"]*)(")')


def _task_uuid(path: Path) -> str | None:
    return next((p for p in reversed(path.parts) if _UUID_RE.fullmatch(p)), None)


def _field(m: re.Match, task_uuid: str | None) -> str:
    found = _UUID_RE.findall(m.group(3))
    if not found and task_uuid and (m.group(2) in ("trials_dir", "trial_uri")
                                    or m.group(2) == "path" and _HOST_PATH_RE.match(m.group(3))):
        found = [task_uuid]
    return m.group(1) + found[-1] + m.group(4) if found else m.group(0)

_TEXT_SUFFIXES = {".json", ".jsonl", ".txt", ".md", ".xml", ".yaml", ".yml",
                  ".log", ".toml", ".py", ".csv", ".html", ".cfg", ".ini"}

# ---------------------------------------------------------------------------
# What of the bundle ships.
#
# The public parts only, named explicitly: the task as a solver sees it and the
# tests that score it. Copying the bundle directory wholesale also shipped every
# tree the leak gate withholds, so nothing is copied that is not listed here.
# The bundle grades itself: its tests/ run in its own verifier environment, and
# the harness ships no grader of its own beside them.
PUBLIC_BUNDLE_PARTS = ("task.toml", "instruction.md", "environment", "tests", "solution")

_BUNDLE_IGNORE = shutil.ignore_patterns(
    "__pycache__", ".pytest_cache", "*.pyc", "*.pyo", ".DS_Store", ".env", "*.log",
)


def _repo_needles() -> list[tuple[str, str]]:
    """(needle, replacement) pairs that cut this checkout's root off a path.

    The general host-path rule only keeps a tail from the directories a bundle
    points into; this checkout's own root is cut exactly, before it -- the old
    anchored rule missed every other repo directory, which is why `--extra-docker-compose` shipped as
    `~/Downloads/.../harness/tools/openhands_agent/overlay.yaml` in every
    config.json we published. Anchoring on the repo root instead covers
    `tools/`, `services/`, `scripts/` and any directory added later.

    Two spellings, because Harbor collapses $HOME to `~` when it serialises
    argv but writes the absolute path in a traceback; the `~` form is derived
    from the repo path itself, not from this process's $HOME, so a tree
    reshaped on one machine still masks when swept on another.
    """
    roots = [str(REPO_ROOT)]
    head = _HOME_RE.match(str(REPO_ROOT))
    if head and len(head.group(0)) < len(str(REPO_ROOT)):
        roots.append("~" + str(REPO_ROOT)[len(head.group(0)):])
    pairs = []
    for r in roots:
        for stem in (f"file://{r}", r):
            # A bare root is a cwd or a --project-directory: emptying it would
            # leave a broken value, so it becomes "." rather than "".
            pairs += [(stem + "/", ""), (stem, ".")]
    return sorted(set(pairs), key=lambda kv: len(kv[0]), reverse=True)


def mask_local_paths(text: str, harbor_fields: bool = True, task_uuid: str | None = None) -> str:
    if harbor_fields:
        text = _FIELD_RE.sub(lambda m: _field(m, task_uuid), text)
    for needle, repl in _repo_needles():
        text = text.replace(needle, repl)
    text = _TMPDIR_RE.sub("$TMPDIR", text)
    return _HOST_PATH_RE.sub(_host_path, text)


def mask_tree(root: Path) -> list[Path]:
    """Mask every text file under `root`; return the files that changed."""
    changed: list[Path] = []
    for p in sorted(root.rglob("*")):
        if not (p.is_file() and p.suffix.lower() in _TEXT_SUFFIXES):
            continue
        # surrogateescape, not replace: a log with one non-UTF-8 byte in it
        # must come back byte-identical apart from the paths we masked.
        text = p.read_text(encoding="utf-8", errors="surrogateescape")
        masked = mask_local_paths(text, harbor_fields=".raw" not in p.relative_to(root).parts,
                                  task_uuid=_task_uuid(p))
        if masked != text:
            p.write_text(masked, encoding="utf-8", errors="surrogateescape")
            changed.append(p)
    return changed


def _load(path: Path, default=None):
    try:
        return json.loads(path.read_bytes())
    except Exception:
        return default


def _dump(path: Path, data) -> None:
    path.write_text(mask_local_paths(json.dumps(data, indent=2, ensure_ascii=False)) + "\n")


def _copy_masked(src: Path, dst: Path) -> None:
    """Copy a text file, masking host-local paths (Harbor's trial_uri,
    trials_dir, jobs_dir carry absolute host paths that must not ship)."""
    dst.write_text(mask_local_paths(src.read_text(encoding="utf-8", errors="replace"),
                                    task_uuid=_task_uuid(dst)),
                   encoding="utf-8")


def _short_model(name: str) -> str:
    """`anthropic/claude-opus-5` -> `opus-5` (the trajectories/<model>/ dir).

    The LiteLLM provider prefix is stripped so a ccbridge, zbridge or bedrock
    run of the same model lands in the same directory. Hyphens are kept:
    TASK-STRUCTURE.md names the folder `opus-5`, never `opus.5`.
    """
    if "/" in name:
        name = name.split("/", 1)[1].replace("/", "-")
    return name.removeprefix("claude-")


# Agents whose runs are reference baselines, not model trajectories. The spec
# ships model runs only (TASK-STRUCTURE.md section 6).
_NON_MODEL_AGENTS = {"oracle", "nop"}

# The verifier files a delivered run carries (section 6). reward.json is
# rewritten to {"reward": x}, reward-details.json loses the judge's container
# paths (_scrub_judge_paths); the others are copied verbatim.
_VERIFIER_COPY = ("ctrf.json", "end_env.json", "junit.xml", "reward-details.json",
                  "final_message.md")

# Where the rubric judge ran and logged inside the verifier container. They
# point at files the delivery does not carry, so they are dropped.
_JUDGE_PATH_KEYS = ("cwd", "atif_trajectory")

# Agents harbor imported by path record that path as config.agent.name; the
# delivered result.json names the agent instead.
_AGENT_NAMES = {"tools.openhands_agent.agent:OpenHandsAgent": "openhands-sdk"}


def _scrub_judge_paths(judge) -> None:
    """Drop the judge's container paths from a judge block, in place.

    judge.json nests them under `configured`; reward-details.json's
    rubric.judge carries them at the top level.
    """
    if not isinstance(judge, dict):
        return
    for target in (judge, judge.get("configured")):
        if isinstance(target, dict):
            for key in _JUDGE_PATH_KEYS:
                target.pop(key, None)


def _write_reward_details(src: Path, dst: Path) -> None:
    details = _load(src, None)
    rubric = details.get("rubric") if isinstance(details, dict) else None
    if not isinstance(rubric, dict):
        shutil.copy2(src, dst)
        return
    rubric.pop("judge_logs", None)
    _scrub_judge_paths(rubric.get("judge"))
    _dump(dst, details)


def _write_result(src: Path, dst: Path) -> None:
    result = _load(src, None)
    agent = ((result or {}).get("config") or {}).get("agent") if isinstance(result, dict) else None
    if not (isinstance(agent, dict) and agent.get("name") in _AGENT_NAMES):
        _copy_masked(src, dst)
        return
    agent["name"] = _AGENT_NAMES[agent["name"]]
    _dump(dst, result)

_DP = 4


def _r(x):
    return round(float(x), _DP) if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def _pristine_verifier(run_dir: Path) -> Path:
    """Where the bundle's own test.sh outputs live for this run.

    harbor_to_output snapshots them into client_verifier/ before it rewrites
    verifier/ in place; older trees without the snapshot fall back to verifier/.
    """
    snap = run_dir / "client_verifier"
    return snap if snap.is_dir() else run_dir / "verifier"


def _judge_criteria_meta(task_src: Path | None) -> list[dict]:
    """[[criterion]] blocks of tests/reward/rubric/judge.toml, in file order."""
    if task_src is None:
        return []
    path = task_src / "tests" / "reward" / "rubric" / "judge.toml"
    if not path.is_file():
        # Pre-rewardkit bundles keep the author's rubric at tests/judge.toml,
        # with source weights in `weight` and no is_positive annotation.
        path = task_src / "tests" / "judge.toml"
    if not path.is_file():
        return []
    try:
        import tomllib
        return list(tomllib.loads(path.read_text(encoding="utf-8")).get("criterion") or [])
    except Exception as e:  # malformed judge.toml: report, ship without metadata
        print(f"Warning: cannot parse {path}: {e}", file=sys.stderr)
        return []


def _threshold(task_src: Path | None) -> float:
    tw = _load(task_src / "tests" / "test_weights.json", {}) if task_src else {}
    t = (tw or {}).get("threshold")
    return float(t) if isinstance(t, (int, float)) else 0.7


def _reward_weights(task_src: Path | None) -> dict:
    """weights of the `reward` aggregation in tests/reward/reward.toml."""
    if task_src is not None:
        path = task_src / "tests" / "reward" / "reward.toml"
        if path.is_file():
            try:
                import tomllib
                for block in tomllib.loads(path.read_text(encoding="utf-8")).get("reward") or []:
                    if block.get("name") == "reward" and isinstance(block.get("weights"), dict):
                        return dict(block["weights"])
            except Exception as e:
                print(f"Warning: cannot parse {path}: {e}", file=sys.stderr)
    return {"pytest": 0.5, "rubric": 0.5}


def _pytest_block(ver: Path, score) -> dict:
    grade = _load(ver / "grade_report.json", {}) or {}
    summary = ((_load(ver / "ctrf.json", {}) or {}).get("results") or {}).get("summary") or {}
    passed = summary.get("passed")
    failed = summary.get("failed")
    if failed is None:
        exit_code = None
    else:
        exit_code = 0 if failed == 0 and not summary.get("other") else 1
    return {
        "score": _r(score),
        "passed": passed,
        "failed": failed,
        "exit_code": exit_code,
        "guard_polarity": grade.get("guard_polarity"),
        "gate": grade.get("gate"),
        "missed": list(grade.get("missed") or []),
        "guards_tripped": list(grade.get("guards_tripped") or []),
        "errors": list(grade.get("errors") or []),
        "tests": [
            {
                "name": t.get("test", t.get("name")),
                "weight": t.get("weight"),
                "passed": t.get("passed"),
                "tripped": t.get("tripped"),
                "reason": t.get("reason") or None,
                "error": t.get("error") or None,
            }
            for t in (grade.get("tests") or [])
        ],
    }


def _rubric_block(ver: Path, meta: list[dict]) -> list[dict]:
    details = (_load(ver / "reward-details.json", {}) or {}).get("rubric") or {}
    judged = {str(c.get("id", c.get("name"))): c for c in (details.get("criteria") or [])}
    by_name = {c.get("name"): c for c in (details.get("criteria") or [])}
    rows = []
    order = meta or [{"id": k, "name": v.get("name")} for k, v in judged.items()]
    for m in order:
        cid = str(m.get("id", m.get("name")))
        got = judged.get(cid) or by_name.get(m.get("name")) or {}
        ann = m.get("annotations") or {}
        weight = ann.get("source_weight", m.get("weight", got.get("weight")))
        positive = ann.get("is_positive")
        if positive is None:
            positive = ann.get("polarity", "positive" if (weight or 0) >= 0 else "negative") == "positive"
        satisfied = None
        if got:
            raw = str(got.get("raw", "")).strip().lower()
            satisfied = raw == "yes" if raw in ("yes", "no") else (got.get("value") or 0) >= 1.0
        w = weight if isinstance(weight, (int, float)) else 0
        rows.append({
            "id": cid,
            "name": m.get("name", got.get("name")),
            "criterion": got.get("description", m.get("description")),
            "weight": weight,
            "is_positive": positive,
            "evaluation_target": ann.get("evaluation_target"),
            "satisfied": satisfied,
            "earned": w if positive and satisfied else 0,
            "charged": abs(w) if (not positive) and satisfied else 0,
            "justification": got.get("reasoning"),
            "error": None if got else "criterion not judged",
        })
    return rows


def _failure_mode(errored: bool, passed: bool, combined) -> str:
    if errored:
        return "errored"
    if passed:
        return "pass"
    if not combined:
        return "zero"
    return "partial"


def _model_runs(traj_src: Path) -> list[tuple[int, Path, dict, dict]]:
    """(run_index, run_dir, result, config) of every model run, by run index."""
    runs = []
    for run_dir in traj_src.glob("run_*"):
        m = re.fullmatch(r"run_(\d+)", run_dir.name)
        if not (m and run_dir.is_dir()):
            continue
        result = _load(run_dir / "result.json", {}) or {}
        config = _load(run_dir / "config.json", {}) or result.get("config") or {}
        agent = config.get("agent") or {}
        if (agent.get("name") or "").lower() in _NON_MODEL_AGENTS or not agent.get("model_name"):
            print(f"[delivery] skipping non-model run {run_dir.name} "
                  f"(agent={agent.get('name')!r})")
            continue
        runs.append((int(m.group(1)), run_dir, result, config))
    return sorted(runs, key=lambda r: r[0])


def _write_trajectories(src: Path, dst: Path, task_src: Path | None) -> None:
    """output/<uuid>/trajectory/run_N -> <uuid>/trajectories/<model>/run_N.

    Layout and every file shape follow TASK-STRUCTURE.md sections 1 and 6.
    Numbers come from the bundle's own verifier outputs (rewardkit reward.json,
    reward-details.json, judge.json, grade_report.json, ctrf.json), on the 0-1
    scale, never from the harness's percentage reshaping.
    """
    traj_src = src / "trajectory"
    job_cfg = _load(src / "config.json", {}) or {}
    job_name = job_cfg.get("job_name") or src.name
    threshold = _threshold(task_src)
    weights = _reward_weights(task_src)
    meta = _judge_criteria_meta(task_src)

    by_model: dict[str, list] = {}
    for run in _model_runs(traj_src) if traj_src.is_dir() else []:
        model_full = (run[3].get("agent") or {}).get("model_name", "")
        by_model.setdefault(_short_model(model_full), []).append(run)

    if not by_model:
        print("Warning: no model runs to deliver", file=sys.stderr)
        return

    for short, runs in by_model.items():
        model_dst = dst / "trajectories" / short
        model_dst.mkdir(parents=True)
        per_run = []
        for new_idx, (_old, run_dir, result, config) in enumerate(runs, start=1):
            ver = _pristine_verifier(run_dir)
            dst_run = model_dst / f"run_{new_idx}"
            (dst_run / "agent").mkdir(parents=True)
            (dst_run / "artifacts").mkdir()
            (dst_run / "verifier").mkdir()

            if (run_dir / "agent" / "trajectory.json").is_file():
                shutil.copy2(run_dir / "agent" / "trajectory.json",
                             dst_run / "agent" / "trajectory.json")
            if (run_dir / "artifacts" / "manifest.json").is_file():
                _copy_masked(run_dir / "artifacts" / "manifest.json",
                             dst_run / "artifacts" / "manifest.json")
            if (run_dir / "config.json").is_file():
                _copy_masked(run_dir / "config.json", dst_run / "config.json")
            if (run_dir / "result.json").is_file():
                _write_result(run_dir / "result.json", dst_run / "result.json")

            for name in _VERIFIER_COPY:
                # Snapshots taken before final_message.md joined
                # CLIENT_VERIFIER_FILES lack it; the reshape never touches it
                # in verifier/, so that copy is still the bundle's own.
                src = ver / name
                if not src.is_file() and name == "final_message.md":
                    src = run_dir / "verifier" / name
                if src.is_file() and name == "reward-details.json":
                    _write_reward_details(src, dst_run / "verifier" / name)
                elif src.is_file():
                    shutil.copy2(src, dst_run / "verifier" / name)
                else:
                    print(f"Warning: {run_dir.name}: verifier/{name} missing", file=sys.stderr)

            rewards = _load(ver / "reward.json", {}) or {}
            combined = _r(rewards.get("reward"))
            pytest_score = _r(rewards.get("pytest"))
            rubric_score = _r(rewards.get("rubric"))
            _dump(dst_run / "verifier" / "reward.json",
                  {"reward": combined if combined is not None else 0.0})

            judge = _load(ver / "judge.json", None)
            _scrub_judge_paths(judge)
            judge_model = (judge or {}).get("model") if (judge or {}).get("ran") else None
            if judge is not None and not judge.get("ran"):
                print(f"[make_delivery] WARNING {dst_run.name}: rubric judge did not run "
                      f"({judge.get('reason') or 'no reason given'}); run is pytest-only",
                      file=sys.stderr)
            errored = result.get("exception_info") is not None
            state_failed = (ver / "state_capture_failed.txt").is_file()
            passed = (not errored) and combined is not None and combined >= threshold
            trial_name = result.get("trial_name") or config.get("trial_name") or ""
            trial = trial_name.split("__")[-1]
            model_full = (config.get("agent") or {}).get("model_name", "")

            channels = {k: v for k, v in (("pytest", pytest_score), ("rubric", rubric_score))
                        if v is not None}
            report = {
                "model": short,
                "model_full": model_full,
                "run_index": new_idx,
                "job": job_name,
                "trial": trial,
                "started_at": result.get("started_at"),
                "finished_at": result.get("finished_at"),
                "scale": "0-1",
                "pytest": _pytest_block(ver, pytest_score),
                "rubric": _rubric_block(ver, meta) if rubric_score is not None else [],
                "rubric_score": rubric_score,
                "judge": judge,
                "final_reward": combined,
                "channels": channels,
                "weights": (_load(ver / "reward_producer.json", {}) or {}).get("weights") or weights,
                "state_capture_failed": True if state_failed else None,
                "errored": errored,
            }
            _dump(dst_run / "report.json", report)

            per_run.append({
                "run_index": new_idx,
                "job": job_name,
                "trial": trial,
                "pytest_score": pytest_score,
                "rubric_score": rubric_score,
                "combined_score": combined,
                "passed": passed,
                "errored": errored,
                "failure_mode": _failure_mode(errored, passed, combined),
                "judge_model": judge_model,
                "state_capture_failed": state_failed,
            })

        done = [r for r in per_run if not r["errored"]]

        def _avg(key):
            vals = [r[key] for r in done if r[key] is not None]
            return round(sum(vals) / len(vals), _DP) if vals else None

        _dump(model_dst / "pass_summary.json", {
            "model": short,
            "scale": "0-1",
            "threshold": threshold,
            "runs": len(per_run),
            "runs_completed": len(done),
            "runs_errored": len(per_run) - len(done),
            "errored_run_indices": [r["run_index"] for r in per_run if r["errored"]],
            "runs_with_state_capture_failed": [r["run_index"] for r in per_run
                                               if r["state_capture_failed"]],
            "judge_models": sorted({r["judge_model"] for r in per_run if r["judge_model"]}),
            "runs_without_rubric": [r["run_index"] for r in per_run
                                    if not r["errored"] and r["rubric_score"] is None],
            "average_combined_score": _avg("combined_score"),
            "average_pytest_score": _avg("pytest_score"),
            "average_rubric_score": _avg("rubric_score"),
            "per_run": per_run,
        })
        print(f"[delivery] trajectories/{short}: {len(per_run)} run(s)")


def _copy_public_bundle(task_src: Path, dest: Path) -> list[str]:
    """Copy PUBLIC_BUNDLE_PARTS of `task_src` into `dest`; return what shipped."""
    dest.mkdir(parents=True, exist_ok=True)
    shipped: list[str] = []
    for part in PUBLIC_BUNDLE_PARTS:
        src = task_src / part
        if src.is_dir():
            shutil.copytree(src, dest / part, ignore=_BUNDLE_IGNORE)
        elif src.is_file():
            shutil.copy2(src, dest / part)
        else:
            continue
        shipped.append(part)
    return shipped


def _warn_stray_mounts(compose: Path) -> None:
    """Name any compose line still reaching above the bundle.

    Not rewritten, because only the author knows what the target held: a mount
    of ../../../output resolves to delivery_output/ in a delivered tree and is
    silently created empty by Docker rather than erroring. Said at pack time,
    where it is cheap to fix.
    """
    if not compose.is_file():
        return
    stray = sorted({ln.strip() for ln in compose.read_text(encoding="utf-8").splitlines()
                    if "../../../" in ln and not ln.lstrip().startswith("#")})
    for ln in stray:
        print(f"Warning: {compose} still points outside the bundle: {ln}",
              file=sys.stderr)


def make_delivery(
    task_slug: str,
    output_dir: Path,
    tasks_dir: Path,
    delivery_dir: Path,
) -> None:
    src = output_dir / task_slug
    if not src.exists():
        sys.exit(f"Error: output dir not found: {src}")

    dst = delivery_dir / task_slug
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    task_src = tasks_dir / task_slug
    if not task_src.exists():
        # Bundles are not always directly under --tasks-dir. A collection layout
        # (tasks/<collection>/<task>/) nests them one level deeper, and this is
        # where delivery used to give up and ship an empty data/ dir. Look one
        # level down before reporting the miss. The flat path is still checked
        # first and still wins, so a run that resolves today is unaffected.
        nested = sorted(p for p in tasks_dir.glob(f"*/{task_slug}") if p.is_dir())
        if nested:
            task_src = nested[0]
    if not task_src.exists():
        # The slug carries task.toml's readable name, but the bundle folder is
        # often the bare UUID (staging/<uuid>/). Same flat-then-nested order.
        for name in bundle_dir_names(task_slug)[1:]:
            hits = [tasks_dir / name] + sorted(p for p in tasks_dir.glob(f"*/{name}") if p.is_dir())
            hit = next((p for p in hits if p.is_dir()), None)
            if hit is not None:
                task_src = hit
                break
    if task_src.exists():
        shipped = _copy_public_bundle(task_src, dst)
        print(f"[delivery] bundle parts shipped: {', '.join(shipped) or 'none'}")
        _warn_stray_mounts(dst / "environment" / "docker-compose.yaml")
    else:
        print(f"Warning: task dir not found: {task_src}", file=sys.stderr)
        task_src = None

    _write_trajectories(src, dst, task_src)

    # Safety net: whatever else landed in the bundle (artifacts, logs, task
    # data), no host-local path may ship. This also catches future additions
    # to the bundle without anyone having to remember the masking rule.
    # Only trajectories/ is swept: the bundle parts must stay byte-identical
    # to the source bundle (TASK-STRUCTURE.md section 2).
    traj_root = dst / "trajectories"
    swept = mask_tree(traj_root) if traj_root.is_dir() else []
    for p in swept:
        print(f"masked local paths in: {p.relative_to(dst)}")

    print(f"Done: {dst}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reformat output/<task>/ into delivery_output/<task>/"
    )
    parser.add_argument("task_slug", nargs="?",
                        help="Task slug, e.g. bull-street-lot-expense-claim")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--tasks-dir", default="tasks")
    parser.add_argument("--delivery-dir", default="delivery_output")
    # The masking sweep on its own, over a directory that is already in its
    # final shape. harbor_to_output.py runs it as part of reshape, but two
    # stages write after that -- netaudit's internet_audit.json and finance's
    # receipt -- so run_task.sh calls this again once they are done. Also the
    # way to clean a tree reshaped before these rules covered it.
    parser.add_argument("--mask-only", metavar="DIR",
                        help="mask host-local paths under DIR and exit")
    args = parser.parse_args()

    if args.mask_only:
        root = Path(args.mask_only)
        if not root.is_dir():
            print(f"no such directory: {root}", file=sys.stderr)
            raise SystemExit(1)
        for p in mask_tree(root):
            print(f"masked local paths in: {p}")
        return

    if not args.task_slug:
        parser.error("task_slug is required unless --mask-only is given")

    base = REPO_ROOT
    make_delivery(
        task_slug=args.task_slug,
        output_dir=base / args.output_dir,
        tasks_dir=base / args.tasks_dir,
        delivery_dir=base / args.delivery_dir,
    )


if __name__ == "__main__":
    main()
