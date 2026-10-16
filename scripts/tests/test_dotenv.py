"""`load_dotenv` decides what configuration exists, so it is asserted directly.

Nothing here needs a bundle: `load_dotenv` and the retired-settings refusal that
follows it are lifted out of run_task.sh by regex and driven against a `.env`
written for the test, so the assertions are about the functions rather than
about this machine's file.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUN_TASK = REPO / "scripts" / "run_task.sh"


def _drive(tmp_path, env_text: str, probes: dict[str, str]) -> subprocess.CompletedProcess:
    """Run the real load_dotenv and retired-key refusal over `env_text`."""
    body = RUN_TASK.read_text()
    m = re.search(r"^load_dotenv\(\) \{.*?^done\n", body, re.S | re.M)
    assert m, "load_dotenv / the retired-settings loop not found in run_task.sh"

    (tmp_path / ".env").write_text(env_text)
    driver = tmp_path / "drive.sh"
    driver.write_text(
        "set -u\n"
        f'REPO="{tmp_path}"\n'
        + m.group(0) + "\n"
        + "".join(f'echo "{label}=[${{{key}:-}}]"\n'
                  for label, key in probes.items())
    )
    env = {"PATH": "/usr/bin:/bin"}
    return subprocess.run(["bash", str(driver)], capture_output=True, text=True, env=env)


def test_a_retired_network_switch_in_dotenv_is_refused_not_ignored(tmp_path):
    """NETWORK_ISOLATION_OFF used to open the network for every run on the
    machine. The network is the bundle's task.toml now, so the key does nothing
    -- and a key that does nothing while looking set is configuration that
    vanishes. The run refuses and names it."""
    out = _drive(tmp_path, "NETWORK_ISOLATION_OFF=1\nDOTENV_PROBE_KEY=loaded\n",
                 {"PROBE": "DOTENV_PROBE_KEY"})
    assert out.returncode == 2, out.stdout + out.stderr
    assert "NETWORK_ISOLATION_OFF is set" in out.stderr, out.stderr


def test_dotenv_loads_ordinary_keys(tmp_path):
    out = _drive(tmp_path, "DOTENV_PROBE_KEY=loaded\n", {"PROBE": "DOTENV_PROBE_KEY"})
    assert out.returncode == 0, out.stderr
    assert "PROBE=[loaded]" in out.stdout, out.stdout


def test_dotenv_reads_quoted_values(tmp_path):
    """The charset filter drops any value holding a brace, comma or space, so
    JSON settings sat in .env doing nothing. Quoting carries them."""
    out = _drive(
        tmp_path,
        'DOTENV_BARE_JSON={"a":"b"}\n'            # bare: still unsupported
        "DOTENV_SQ_JSON='{\"a\":\"b\"}'\n"        # the env.template form
        'DOTENV_DQ_VAL="has space, and comma"\n',  # finance_reporter parity
        {"BARE": "DOTENV_BARE_JSON", "SQ": "DOTENV_SQ_JSON", "DQ": "DOTENV_DQ_VAL"},
    )
    combined = out.stdout + out.stderr
    assert 'SQ=[{"a":"b"}]' in combined, (
        "quoted JSON still does not survive .env -- this is the form "
        "env.template documents for ZB_MODEL_ALIAS_JSON\n" + combined)
    assert "DQ=[has space, and comma]" in combined, (
        "finance_reporter.py:119 strips both quote styles; a value one parser "
        "drops and the other loads is the Odoo gate's false positive\n" + combined)
    assert "BARE=[]" in combined and "DOTENV_BARE_JSON" in combined, (
        "an unquoted value outside the charset must still be skipped, and named\n"
        + combined)


def test_a_quoted_retired_key_is_still_refused(tmp_path):
    out = _drive(tmp_path, 'NETWORK_ISOLATION_OFF="1"\n', {})
    assert out.returncode == 2 and "NETWORK_ISOLATION_OFF" in out.stderr, out.stderr
