"""`load_dotenv` decides what configuration exists, so it is asserted directly.

These live apart from test_network_isolation.py, which skips itself wholesale
when `tasks/` is absent -- and `tasks/` is gitignored, so it is absent on every
CI checkout. Nothing here needs a bundle: `load_dotenv` is lifted out of
run_task.sh by regex and driven against a `.env` written for the test, so the
assertions are about the function rather than about this machine's file.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUN_TASK = REPO / "scripts" / "run_task.sh"


def _drive(tmp_path, env_text: str, probes: dict[str, str]) -> str:
    """Run the real load_dotenv over `env_text`; return its combined output."""
    body = RUN_TASK.read_text()
    m = re.search(r"^DOTENV_FORBIDDEN_KEYS=.*?^}", body, re.S | re.M)
    assert m, "load_dotenv / DOTENV_FORBIDDEN_KEYS not found in run_task.sh"

    (tmp_path / ".env").write_text(env_text)
    driver = tmp_path / "drive.sh"
    driver.write_text(
        "set -u\n"
        f'REPO="{tmp_path}"\n'
        + m.group(0) + "\n"
        "load_dotenv\n"
        + "".join(f'echo "{label}=[${{{key}:-}}]"\n'
                  for label, key in probes.items())
    )
    out = subprocess.run(["bash", str(driver)], capture_output=True, text=True)
    return out.stdout + out.stderr


def test_dotenv_cannot_disable_network_isolation(tmp_path):
    """The switch that cost a benchmark result.

    NETWORK_ISOLATION_OFF=1 sat in .env, so every run on the machine was an open
    run and nothing said so. It is a per-RUN decision, and load_dotenv now
    refuses to read it from the file.
    """
    env_file = REPO / ".env"
    if env_file.is_file():
        live = [ln for ln in env_file.read_text().splitlines()
                if not ln.lstrip().startswith("#")]
        assert not any(ln.startswith("NETWORK_ISOLATION_OFF=") for ln in live), (
            ".env sets NETWORK_ISOLATION_OFF; every run on this machine is an "
            "open run and the audit is the only thing left standing"
        )

    # DOTENV_PROBE_KEY is a name nothing else could have set, so an inherited
    # value cannot mask the result -- load_dotenv is fill-if-unset by design.
    combined = _drive(
        tmp_path,
        "NETWORK_ISOLATION_OFF=1\nDOTENV_PROBE_KEY=loaded\n",
        {"ISO": "NETWORK_ISOLATION_OFF", "PROBE": "DOTENV_PROBE_KEY"},
    )

    assert "ISO=[]" in combined, combined
    assert "REFUSED" in combined, (
        "the key was dropped silently; configuration that vanishes looks "
        "exactly like configuration that works\n" + combined
    )
    # Every other key must still load, or the guard has broken .env instead of
    # narrowing it.
    assert "PROBE=[loaded]" in combined, combined


def test_dotenv_reads_quoted_values_without_widening_the_refusal(tmp_path):
    """Quoting makes a value loadable; it must not make a key permissible.

    The charset filter drops any value holding a brace, comma or space, so JSON
    settings sat in .env doing nothing. Quoting now carries them -- which puts a
    second path to `export` beside the refusal list, so the refusal is asserted
    here too: it keys off the NAME, and no amount of quoting reaches it.
    """
    combined = _drive(
        tmp_path,
        'DOTENV_BARE_JSON={"a":"b"}\n'            # bare: still unsupported
        "DOTENV_SQ_JSON='{\"a\":\"b\"}'\n"        # the env.template form
        'DOTENV_DQ_VAL="has space, and comma"\n'  # finance_reporter parity
        'NETWORK_ISOLATION_OFF="1"\n',            # quoted: still refused
        {"BARE": "DOTENV_BARE_JSON", "SQ": "DOTENV_SQ_JSON",
         "DQ": "DOTENV_DQ_VAL", "ISO": "NETWORK_ISOLATION_OFF"},
    )

    assert 'SQ=[{"a":"b"}]' in combined, (
        "quoted JSON still does not survive .env -- this is the form "
        "env.template documents for ZB_MODEL_ALIAS_JSON\n" + combined)
    assert "DQ=[has space, and comma]" in combined, (
        "finance_reporter.py:119 strips both quote styles; a value one parser "
        "drops and the other loads is the Odoo gate's false positive\n" + combined)
    assert "ISO=[]" in combined and "REFUSED" in combined, (
        "quoting the value smuggled a refused KEY past the list\n" + combined)
    assert "BARE=[]" in combined and "DOTENV_BARE_JSON" in combined, (
        "an unquoted value outside the charset must still be skipped, and named\n"
        + combined)
