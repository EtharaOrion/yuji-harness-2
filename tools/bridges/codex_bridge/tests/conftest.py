"""Put tools/bridges on sys.path so `import codex_bridge` works from anywhere
(the package is not pip-installed; the runner uses the same shim)."""

from __future__ import annotations

import sys
from pathlib import Path

_BRIDGES = Path(__file__).resolve().parents[2]
if str(_BRIDGES) not in sys.path:
    sys.path.insert(0, str(_BRIDGES))
