"""README.md is generated from the harness code and must match it."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def test_readme_is_regenerated_from_the_code():
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "gen_readme.py"), "--check"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
