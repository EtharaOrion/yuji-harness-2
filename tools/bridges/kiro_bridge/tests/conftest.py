"""Put the repo root on sys.path so `import kiro_bridge` works when pytest is
run from inside kiro_bridge/ (the package is not pip-installed; from the repo
root it is already importable)."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
