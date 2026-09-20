#!/usr/bin/env python3
"""codexbridge: the whole verifier, inside its own container.

Every scored channel runs in here now -- rubric, Channel A, the state dump and
the ledger -- and none of them runs in `main`, the container the agent had root
in. The harness's grading script (/harness/scoring/tests/evaluate.sh) is what
runs, one copy shared by every bundle, so the numbers cannot drift from what
main used to produce or between one bundle and the next; this image just
supplies what it needs (python, pytest, the mcp client, codex) and the mounts
it reads.

The rubric was moved first, for a different reason: on the host,
`codex exec --sandbox read-only` can still READ the whole disk -- other runs
under output/, answer files under tasks/*/tests -- because read-only only
forbids writes.

  GET  /healthz   200 when an evaluation could run: token set, codex login
                  installed, codex + pytest + the bundle script present, and
                  /logs/verifier writable. Compose's healthcheck calls it
                  (`codexbridge.py --health`) and harbor's `up --wait` holds the
                  trial on it, so a broken grader fails before the agent phase
                  instead of after it.
  POST /evaluate  {"trajectory": {...}} + header x-judge-token. Writes the
                  trajectory where the bundle scripts expect it, runs
                  the shared evaluate.sh, and reports what it wrote. The reports
                  land in /logs/verifier, which is harbor's own per-trial
                  directory on the host -- main never relays them.

The token is made per run by scripts/run_task.sh and reaches two places only:
this container's environment and the verifier step's (task.toml [verifier.env];
harbor applies that to the test script, never to the agent). The agent shares a
network with this service but never holds the token.

The login arrives as a read-only mount and is COPIED into CODEX_HOME at start.
codex refreshes its access token by rewriting auth.json; the copy is what it
rewrites, so nothing in here can change the host's file.

Stdlib only.
"""
from __future__ import annotations

import hmac
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.request import ProxyHandler, build_opener

HERE = Path(__file__).resolve().parent
GRADED_IN = "judge-container"
LOG_TAIL_CHARS = 4000

# Where the grading script lives, and the file its steps read the trajectory
# from. Both are fixed: the caller says what to grade, never how. main builds
# the trajectory with the bundle's own parser (test.sh step 1) and posts it, so
# Channel A sees exactly the evidence it saw before the move.
#
# The script is the harness's, shared by every bundle and mounted here with the
# rest of services/scoring. What is per-bundle -- test_outputs.py,
# test_weights.json, rubric.json, state_dump.py, the answer files -- it reads
# from /tests.
EVALUATE_SH = Path(os.environ.get("JUDGE_EVALUATE_SH", "/harness/scoring/tests/evaluate.sh"))
TRAJECTORY_PATH = Path(os.environ.get("JUDGE_TRAJECTORY_PATH", "/tmp/agent_trajectory.json"))
VERIFIER_DIR = Path(os.environ.get("JUDGE_VERIFIER_DIR", "/logs/verifier"))

_state: dict[str, str | None] = {"credential_error": "codex login not installed yet",
                                 "login_checked": None, "login_error": None}
_grade_lock = threading.Lock()
_codex_version: str | None = None


def _port() -> int:
    return int(os.environ.get("JUDGE_PORT", "8770"))


def _model() -> str:
    return os.environ.get("JUDGE_MODEL", "gpt-5.6-sol")


def _judge_cli() -> Path:
    return Path(os.environ.get("JUDGE_CLI", HERE / "rubric_judge_cli.py"))


def _max_body() -> int:
    return int(os.environ.get("JUDGE_MAX_BODY_BYTES", str(64 * 1024 * 1024)))


def _grade_timeout() -> float:
    return float(os.environ.get("JUDGE_GRADE_TIMEOUT_SEC", "1500"))


def install_credential() -> str | None:
    """Copy the mounted login into CODEX_HOME. None on success, else the reason."""
    src = Path(os.environ.get("CODEX_AUTH_SRC", "/run/codex-auth/auth.json"))
    if not src.is_file() or src.stat().st_size == 0:
        return f"codex login not mounted at {src} (scripts/run_task.sh sets CODEX_AUTH_FILE)"
    # Non-empty is not the same as carrying a credential. Measured: `codex login
    # status` answers "Logged in using ChatGPT" for {"tokens": null} and for {},
    # so the CLI will not report this one. Silent on anything unparseable.
    try:
        doc = json.loads(src.read_text())
        tok = doc.get("tokens") or {}
        empty = isinstance(doc, dict) and isinstance(tok, dict) \
            and not (tok.get("access_token") or doc.get("OPENAI_API_KEY"))
    except (OSError, ValueError, AttributeError):
        empty = False
    if empty:
        return (f"the codex login mounted at {src} carries no credential "
                "(no tokens, no API key); run `codex login` on the host")
    home = Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex")
    try:
        home.mkdir(parents=True, exist_ok=True)
        dest = home / "auth.json"
        shutil.copyfile(src, dest)
        dest.chmod(0o600)
    except OSError as exc:
        return f"could not install codex login into {home}: {exc}"
    return None


def codex_login_error() -> str | None:
    """Ask the CLI whether the mounted login actually works.

    install_credential() only proves a file arrived. An expired or revoked login
    passes that and then fails every rubric criterion at grading time. Only an
    explicit "not logged in" is treated as a fault; anything else we could not
    run or parse is left alone, so a slow or odd CLI never blocks a good run.
    """
    if _state["login_checked"]:
        return _state["login_error"]
    _state["login_checked"] = "1"
    try:
        proc = subprocess.run(["codex", "login", "status"], capture_output=True,
                              text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if "not logged in" in f"{proc.stdout} {proc.stderr}".lower():
        _state["login_error"] = ("the mounted codex login is not usable "
                                 "(`codex login status` says: not logged in); "
                                 "run `codex login` on the host and retry")
    return _state["login_error"]


def not_ready() -> str | None:
    """Why a grade cannot run right now, or None."""
    if not os.environ.get("JUDGE_TOKEN"):
        return "JUDGE_TOKEN is not set (scripts/run_task.sh creates one per run)"
    if _state["credential_error"]:
        return _state["credential_error"]
    if not shutil.which("codex"):
        return "codex CLI not found on PATH"
    login = codex_login_error()
    if login:
        return login
    if not _judge_cli().is_file():
        return f"rubric grader missing at {_judge_cli()}"
    if importlib.util.find_spec("pytest") is None:
        return "pytest is not installed; Channel A cannot run here"
    if importlib.util.find_spec("mcp") is None:
        return "the mcp client is not installed; the state dump cannot run here"
    if not EVALUATE_SH.is_file():
        return (f"{EVALUATE_SH} is not mounted; services/scoring must be mounted at "
                "/harness/scoring in this container (see the judge service in the "
                "bundle's compose file)")
    if not os.access(VERIFIER_DIR, os.W_OK):
        return f"{VERIFIER_DIR} is not writable; harbor's per-trial log dir must be mounted here"
    return None


def codex_version() -> str | None:
    global _codex_version
    if _codex_version is None:
        try:
            out = subprocess.run(["codex", "--version"], capture_output=True,
                                 text=True, timeout=30)
            _codex_version = (out.stdout or out.stderr).strip() or "unknown"
        except (OSError, subprocess.SubprocessError):
            _codex_version = "unknown"
    return _codex_version


def _load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def evaluate(trajectory: dict) -> dict:
    """Run the bundle's whole grading script over one run.

    The command is fixed here rather than taken from the request: `main` is the
    container the agent worked in, and an endpoint that ran what main asked for
    would hand the agent's leftovers a way to grade themselves. All the caller
    supplies is the trajectory.

    Reports are written by the script itself, into harbor's per-trial
    /logs/verifier mount, so nothing has to be relayed back through main.
    """
    TRAJECTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    TRAJECTORY_PATH.write_text(json.dumps(trajectory))

    # A run with no tool calls and nothing said is a run that never started --
    # an auth failure, a rate limit, a container that died in agent setup. It
    # used to be graded anyway: the whole rubric was bought from codex against
    # empty evidence and a reward of 0 published, which is indistinguishable
    # from an agent that tried and failed. Refuse instead, and say which it was.
    steps = trajectory.get("steps") or []
    final = (trajectory.get("final_message") or "").strip()
    if not steps and not final:
        VERIFIER_DIR.mkdir(parents=True, exist_ok=True)
        (VERIFIER_DIR / "no_agent_activity.txt").write_text(
            "the agent produced no tool calls and no final message; nothing was graded\n")
        return {
            "ok": False,
            "reason": "no agent activity: 0 tool calls, no final message — "
                      "the agent phase did not run, so nothing was graded",
            "returncode": None,
            "reward": None,
            "rubric_criteria": 0,
            "written": ["no_agent_activity.txt"],
            "log_tail": "",
            "graded_in": GRADED_IN,
            "model": _model(),
            "codex_version": None,
        }

    env = dict(os.environ)
    env.setdefault("JUDGE_MODEL", _model())
    env["COMPLEXMCP_TRAJECTORY"] = env["ATLAS_TRAJECTORY"] = str(TRAJECTORY_PATH)
    env["MCPATLAS_TRAJECTORY"] = str(TRAJECTORY_PATH)
    try:
        proc = subprocess.run(["bash", str(EVALUATE_SH)], capture_output=True, text=True,
                              timeout=_grade_timeout(), cwd="/tmp", env=env)
        rc, log = proc.returncode, (proc.stdout or "") + (proc.stderr or "")
    except subprocess.TimeoutExpired:
        rc, log = None, f"evaluation timed out after {_grade_timeout():.0f}s"
    except OSError as exc:
        rc, log = None, f"could not run {EVALUATE_SH}: {exc}"

    written = sorted(p.name for p in VERIFIER_DIR.glob("*")) if VERIFIER_DIR.is_dir() else []
    reward = _load_json(VERIFIER_DIR / "reward.json")
    breakdown = _load_json(VERIFIER_DIR / "rubric_breakdown.json") or {}
    rubric_rows = breakdown.get("per_criterion") or breakdown.get("results") or []

    reason = None
    if rc != 0:
        reason = f"evaluation exited {rc}" if rc is not None else "evaluation did not finish"
    elif reward is None:
        reason = "no reward.json was written"
    return {
        "ok": reason is None,
        "reason": reason,
        "returncode": rc,
        "reward": reward,
        "rubric_criteria": len(rubric_rows),
        "written": written,
        "log_tail": log[-LOG_TAIL_CHARS:],
        "graded_in": GRADED_IN,
        "model": _model(),
        "codex_version": codex_version(),
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "codexbridge"

    def log_message(self, fmt, *args):
        sys.stderr.write("[codexbridge] " + (fmt % args) + "\n")

    def _send(self, code: int, doc: dict) -> None:
        body = json.dumps(doc).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path != "/healthz":
            return self._send(404, {"error": "not found"})
        why = not_ready()
        self._send(503 if why else 200,
                   {"status": "unavailable" if why else "ok", "reason": why,
                    "model": _model()})

    def do_POST(self):
        if self.path != "/evaluate":
            return self._send(404, {"error": "not found"})
        why = not_ready()
        if why:
            return self._send(503, {"error": why})
        sent = self.headers.get("x-judge-token", "").encode()
        if not hmac.compare_digest(sent, os.environ["JUDGE_TOKEN"].encode()):
            return self._send(401, {"error": "bad or missing x-judge-token"})
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if length <= 0:
            return self._send(411, {"error": "Content-Length required"})
        if length > _max_body():
            self.close_connection = True
            return self._send(413, {"error": f"body {length} bytes exceeds {_max_body()}"})
        try:
            req = json.loads(self.rfile.read(length))
            trajectory = req["trajectory"]
            if not isinstance(trajectory, dict):
                raise TypeError("trajectory must be a JSON object")
        except (ValueError, KeyError, TypeError) as exc:
            return self._send(400, {"error": f"bad request: {exc}"})
        with _grade_lock:
            try:
                doc = evaluate(trajectory)
            except Exception as exc:  # the verifier must hear about it, not time out
                return self._send(500, {"error": f"evaluation crashed: {exc!r}"})
        self._send(200, doc)


def make_server(host: str, port: int) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), Handler)


def health_probe() -> int:
    """Exit status for compose's healthcheck. Never proxied: it is loopback."""
    opener = build_opener(ProxyHandler({}))
    try:
        with opener.open(f"http://127.0.0.1:{_port()}/healthz", timeout=5) as resp:
            return 0 if resp.status == 200 else 1
    except Exception as exc:
        print(f"[codexbridge] unhealthy: {exc}", file=sys.stderr)
        return 1


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if args[:1] == ["--health"]:
        return health_probe()
    _state["credential_error"] = install_credential()
    host = os.environ.get("JUDGE_HOST", "0.0.0.0")
    srv = make_server(host, _port())
    print(f"[codexbridge] listening on {host}:{_port()} model={_model()} "
          f"ready={not_ready() or 'yes'}", flush=True)
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
