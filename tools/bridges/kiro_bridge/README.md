# kiro-bridge

Dependency-free Python bridge for running Kiro CLI models and evaluations through the operator's existing `kiro-cli` login.

## Install

```bash
python -m pip install .
kiro-bridge doctor --json
kiro-bridge models
```

Python API:

```python
from kiro_bridge import KiroBridge

with KiroBridge() as kiro:
    result = kiro.run_agent(
        "Inspect the project and fix the failing tests",
        cwd="/path/to/project",
        model="claude-opus-5",
        effort="high",
    )
    print(result.transcript())
```

Anthropic-compatible local server:

```bash
kiro-bridge serve --host 127.0.0.1 --port 8798 --pin-model claude-opus-5
```

See the bundle-level `../README.md` for OpenHands/Harbor integration, reasoning capture, evaluator routing, and security guidance.
