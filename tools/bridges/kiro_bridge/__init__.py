"""Kiro bridge: run Kiro CLI agents and Kiro-judged evaluations on a Kiro login.

    from kiro_bridge import KiroBridge

    with KiroBridge() as kiro:   # uses the `kiro-cli login` credentials
        run = kiro.run_agent("Fix the failing test", cwd="/repo")
        verdict = kiro.evaluate(task, criteria, {"transcript": run.transcript()})
"""

from kiro_bridge.agent import AgentResult, Usage, render_transcript, run_agent
from kiro_bridge.anthropic_server import AnthropicServer, CallRecord
from kiro_bridge.bridge import (
    DoctorReport,
    KiroBridge,
    close_shared_bridge,
    credential_problem,
    shared_bridge,
)
from kiro_bridge.config import KiroBridgeError, KiroConfig
from kiro_bridge.evaluate import (
    CriterionVerdict,
    EvalResult,
    complete,
    complete_with_reasoning,
    evaluate,
    parse_verdicts,
    score_verdicts,
)
from kiro_bridge.mcp_http import McpToolServer
from kiro_bridge.tools import CustomTool, ExecResult, sandbox_tools

__version__ = "0.1.0"

__all__ = [
    "AgentResult",
    "AnthropicServer",
    "CallRecord",
    "CriterionVerdict",
    "CustomTool",
    "DoctorReport",
    "EvalResult",
    "ExecResult",
    "KiroBridge",
    "KiroBridgeError",
    "KiroConfig",
    "McpToolServer",
    "Usage",
    "close_shared_bridge",
    "complete",
    "complete_with_reasoning",
    "credential_problem",
    "evaluate",
    "parse_verdicts",
    "render_transcript",
    "run_agent",
    "sandbox_tools",
    "score_verdicts",
    "shared_bridge",
]
