from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import NamedTuple
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from harbor_to_output import norm_reward, pct_to_reward, bundle_dir_names

# tools/delivery/make_delivery.py -> the checkout root. Every path this
# file masks is measured against it.
REPO_ROOT = _Path(__file__).resolve().parents[2]

# Host-local path masking, inline. Harbor stamps absolute host paths
# (trial_uri, trials_dir, jobs_dir, tracebacks) into bookkeeping files, which
# is how `/Users/<name>/...` strings end up in shipped bundles. A path holding
# a repo anchor is cut to anchor-relative form (`/Users/x/dev/harness/output/t`
# -> `output/t`, still a usable pointer inside the bundle); any other
# home-rooted path gets its `/Users/<name>` or `/home/<name>` head replaced
# with `~`. Container paths (/workspace, /logs, /tmp) survive verbatim.
_ANCHORED_RE = re.compile(
    r"(?:file://)?(?:/(?:Users|home)|~)/[^\s\"'\\]*?/"
    r"(?=(?:delivery_output|output|input|tasks|jobs)(?:/|[\"'\s]|$))"
)
_HOME_RE = re.compile(r"(?:file://)?/(?:Users|home)/[^/\s\"'\\]+")

# macOS mkdtemp: /var/folders/<2 chars>/<hash>/T. The hash is derived from the
# uid, so it identifies the operator's account as surely as their name does.
# Harbor writes these into the `docker compose -f ...` line it quotes back in
# every environment-build failure, which is how they reach exception.txt,
# result.json and job.log. /private is the real path, /var the symlink, and
# both spellings turn up in the same message.
_TMPDIR_RE = re.compile(r"(?:/private)?/var/folders/[^/\s\"']+/[^/\s\"']+/[A-Z]")

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
PUBLIC_BUNDLE_PARTS = ("task.toml", "instruction.md", "environment", "tests")

_BUNDLE_IGNORE = shutil.ignore_patterns(
    "__pycache__", ".pytest_cache", "*.pyc", "*.pyo", ".DS_Store", ".env", "*.log",
)


def _repo_needles() -> list[tuple[str, str]]:
    """(needle, replacement) pairs that cut this checkout's root off a path.

    _ANCHORED_RE only fires on a path whose tail starts at one of the five
    directories a delivered bundle points into. Everything else in the repo is
    invisible to it -- which is why `--extra-docker-compose` shipped as
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


def mask_local_paths(text: str) -> str:
    text = _ANCHORED_RE.sub("", text)
    for needle, repl in _repo_needles():
        text = text.replace(needle, repl)
    text = _TMPDIR_RE.sub("$TMPDIR", text)
    return _HOME_RE.sub("~", text)


def mask_tree(root: Path) -> list[Path]:
    """Mask every text file under `root`; return the files that changed."""
    changed: list[Path] = []
    for p in sorted(root.rglob("*")):
        if not (p.is_file() and p.suffix.lower() in _TEXT_SUFFIXES):
            continue
        # surrogateescape, not replace: a log with one non-UTF-8 byte in it
        # must come back byte-identical apart from the paths we masked.
        text = p.read_text(encoding="utf-8", errors="surrogateescape")
        masked = mask_local_paths(text)
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
    dst.write_text(mask_local_paths(src.read_text(encoding="utf-8", errors="replace")),
                   encoding="utf-8")


def _short_model(name: str) -> str:
    # Strip the LiteLLM provider prefix (bedrock/, anthropic/, gemini/, ...)
    # so a ccbridge, zbridge or bedrock run of the same model lands in the
    # same trajectory/<alias>/ directory for easy comparison. Without this,
    # Path join on "bedrock/claude-opus-5" nests the model under a provider
    # subdir, so the same alias produces different delivery trees per bridge.
    if "/" in name:
        name = name.split("/", 1)[1].replace("/", "-")
    short = name.removeprefix("claude-")
    return re.sub(r"-(\d+)$", r".\1", short)


def _run_label(dir_name: str) -> str:
    return dir_name.replace("_", " ", 1)


def _rubric_cleared(item: dict) -> bool:
    positive = item.get("is_positive", True)
    passed = bool(item.get("passed"))
    return passed if positive else not passed


def _build_judge_usage(tokens: dict) -> dict:
    return {
        "input_tokens": tokens.get("judge_input_tokens", 0),
        "cache_read_input_tokens": tokens.get("judge_input_cache_tokens", 0),
        # `judge_cache_write_tokens` is the current name; `judge_output_cache_tokens`
        # is the old one and is still read so artifacts written before the rename
        # keep costing correctly. Both hold cache CREATION, which is input-side.
        "cache_creation_input_tokens": tokens.get(
            "judge_cache_write_tokens",
            tokens.get("judge_output_cache_tokens", 0)),
        "output_tokens": tokens.get("judge_output_tokens", 0),
        "cost_usd": tokens.get("judge_cost_usd", 0),
    }


def _copy_public_bundle(task_src: Path, dest: Path) -> list[str]:
    """Copy PUBLIC_BUNDLE_PARTS of `task_src` into `dest`; return what shipped."""
    dest.mkdir(parents=True)
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
        shipped = _copy_public_bundle(task_src, dst / "data")
        print(f"[delivery] bundle parts shipped: {', '.join(shipped) or 'none'}")
        _warn_stray_mounts(dst / "data" / "environment" / "docker-compose.yaml")
    else:
        print(f"Warning: task dir not found: {task_src}", file=sys.stderr)
        (dst / "data").mkdir()

    traj_src = src / "trajectory"
    traj_dst = dst / "trajectory"
    traj_dst.mkdir()

    pass_sum = _load(src / "pass_summary.json", {})
    per_run_clean = [
        {k: v for k, v in run.items() if k != "include_multimodal"}
        for run in (pass_sum.get("per_run") or [])
    ]
    delivery_pass = {k: v for k, v in pass_sum.items() if k != "per_run"}
    delivery_pass["per_run"] = per_run_clean
    _dump(traj_dst / "pass_summary.json", delivery_pass)

    model_name: str = pass_sum.get("model", "")

    for run_dir in sorted(traj_src.glob("run_*")):
        report = _load(run_dir / "report.json", {})
        run_model = report.get("model") or model_name
        model_dst = traj_dst / _short_model(run_model)
        dst_run = model_dst / _run_label(run_dir.name)
        dst_run.mkdir(parents=True)

        (dst_run / "agent").mkdir()
        src_traj = run_dir / "agent" / "trajectory.json"
        if src_traj.exists():
            shutil.copy2(src_traj, dst_run / "agent" / "trajectory.json")

        arts_src = run_dir / "artifacts"
        if arts_src.exists():
            shutil.copytree(arts_src, dst_run / "artifacts")
        else:
            (dst_run / "artifacts").mkdir()

        if (run_dir / "config.json").exists():
            _copy_masked(run_dir / "config.json", dst_run / "config.json")
        judge_tokens = _load(run_dir / "verifier" / "judge_tokens.json", {})
        # A bundle's judge may write judge_tokens.json as a LIST of per-call
        # entries; older judges wrote one dict. Merge a
        # list into the dict shape _build_judge_usage expects: token counts
        # and cost sum across calls, model comes from the first entry.
        if isinstance(judge_tokens, list):
            calls = [t for t in judge_tokens if isinstance(t, dict)]
            merged = {"model_name": calls[0].get("model_name", "") if calls else ""}
            for key in ("judge_input_tokens", "judge_output_tokens",
                        "judge_input_cache_tokens", "judge_output_cache_tokens",
                        "judge_cache_write_tokens", "judge_cost_usd"):
                vals = [t[key] for t in calls if isinstance(t.get(key), (int, float))]
                if vals:
                    merged[key] = sum(vals)
            judge_tokens = merged
        report.pop("include_multimodal", None)
        report["judge_model"] = judge_tokens.get("model_name", "")
        report["judge_usage"] = _build_judge_usage(judge_tokens)
        _dump(dst_run / "report.json", report)

        if (run_dir / "result.json").exists():
            _copy_masked(run_dir / "result.json", dst_run / "result.json")

        ver_src = run_dir / "verifier"
        ver_dst = dst_run / "verifier"
        ver_dst.mkdir()

        if (ver_src / "ctrf.json").exists():
            shutil.copy2(ver_src / "ctrf.json", ver_dst / "ctrf.json")

        for src in (run_dir / "logs" / "verifier-stdout.txt",
                    ver_src / "test-stdout.txt"):
            if src.exists():
                shutil.copy2(src, ver_dst / "test-stdout.txt")
                break

        reward_src = _load(ver_src / "reward.json", {}) or {}
        _dump(ver_dst / "reward.json", {
            "reward": norm_reward(reward_src.get("reward", 0)),
            **{k: reward_src[k] for k in ("comparable", "caveats") if k in reward_src},
        })

        rubric = report.get("rubric", [])
        # harbor_to_output writes this key PRESENT and NULL when the rubric
        # channel went ungraded (the host rubric pass refuses to grade an empty
        # trajectory), so `.get(key, 0)` returned None and the division below
        # raised TypeError. Coercing it to 0 would be worse than the crash: it
        # would publish "this run scored 0%" for a run that was never scored at
        # all, which is precisely the claim the reward ledger declines to make.
        # Unscored stays null, and the delivery bundle says so.
        rubric_pct = report.get("rubric_weights_percentage")
        score_data = {
            "model": report.get("model", run_model),
            "run_index": report.get("run_index", 1),
            "rubric_weights_percentage": rubric_pct,
            "total": len(rubric),
            "passed": sum(1 for r in rubric if _rubric_cleared(r)),
            "failed": sum(1 for r in rubric if not _rubric_cleared(r)),
            "reward": pct_to_reward(rubric_pct),
            "scored": rubric_pct is not None,
            "rubric": rubric,
            "judge_model": judge_tokens.get("model_name", ""),
            "judge_usage": report.get("judge_usage", {}),
        }
        _dump(ver_dst / "score.json", score_data)

    # Safety net: whatever else landed in the bundle (artifacts, logs, task
    # data), no host-local path may ship. This also catches future additions
    # to the bundle without anyone having to remember the masking rule.
    swept = mask_tree(dst)
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

    base = Path(__file__).resolve().parent.parent
    make_delivery(
        task_slug=args.task_slug,
        output_dir=base / args.output_dir,
        tasks_dir=base / args.tasks_dir,
        delivery_dir=base / args.delivery_dir,
    )


if __name__ == "__main__":
    main()
