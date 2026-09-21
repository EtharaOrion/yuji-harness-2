#!/usr/bin/env python3
"""OpenHands adapter for yuji-atlas: drop-in replacement for claude-code agent.

Runs an OpenHands agent against a yuji-atlas task bundle. Uses the same auth
(CLAUDE_CODE_OAUTH_TOKEN) and same trajectory format
(/logs/agent/claude-code.txt in Anthropic-messages JSONL) as the claude-code
agent path, so no downstream changes are needed in test.sh, judge_client.py,
grade.py, or combine_channels.py.

Invocation (from harness/scripts/run_task.sh AGENT=openhands case):
    python3 /opt/openhands-adapter/entrypoint.py

Environment reads:
  CLAUDE_CODE_OAUTH_TOKEN   Claude OAuth token for opus-5 agent (required)
  BUNDLE_ROOT               Path to the mounted /tests + /workspace pairing
                            (default: derived from /tests parent)
  TASK_INSTRUCTION_PATH     Default: /tests/instruction.md
  AGENT_LOG_PATH            Default: /logs/agent/claude-code.txt
                            (writing here means downstream parsers pick up
                            the trajectory without changes)
  REPORT_OUTPUT_PATH        Default: /workspace/out/report.md
  MAX_ITERATIONS            Default: 100
  MODEL                     Default: anthropic/claude-opus-5

Writes:
  <AGENT_LOG_PATH>          JSONL file, one Anthropic-messages event per line
  <REPORT_OUTPUT_PATH>      Final agent-produced report

MCP servers are read from the bundle's task.toml [environment.mcp_servers]
list and passed to the OpenHands agent via mcp_config with the same
streamable-http URLs the claude-code agent uses.

Exit codes:
  0 = success (report produced, trajectory written)
  1 = configuration error (missing token, missing instruction file)
  2 = OpenHands runtime error (agent crashed or hit iteration limit)
"""

from __future__ import annotations

import json
import os
import sys
import time
import tomllib
from pathlib import Path

CLAUDE_CODE_OAUTH_TOKEN = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN", "")
BUNDLE_ROOT = Path(os.environ.get("BUNDLE_ROOT", "/tests")).parent
TASK_INSTRUCTION_PATH = Path(os.environ.get("TASK_INSTRUCTION_PATH", "/tests/instruction.md"))
TASK_TOML_PATH = Path(os.environ.get("TASK_TOML_PATH", str(BUNDLE_ROOT / "task.toml")))
AGENT_LOG_PATH = Path(os.environ.get("AGENT_LOG_PATH", "/logs/agent/openhands.txt"))
TRAJECTORY_JSON_PATH = Path(os.environ.get("TRAJECTORY_JSON_PATH", "/logs/agent/openhands_trajectory.json"))
REPORT_OUTPUT_PATH = Path(os.environ.get("REPORT_OUTPUT_PATH", "/workspace/out/report.md"))
MAX_ITERATIONS = int(os.environ.get("MAX_ITERATIONS", "100"))
MODEL = os.environ.get("MODEL", "anthropic/claude-opus-5")


def load_mcp_servers_from_task_toml(task_toml_path: Path) -> dict:
    if not task_toml_path.exists():
        return {}
    with task_toml_path.open("rb") as handle:
        data = tomllib.load(handle)
    servers = data.get("environment", {}).get("mcp_servers", []) or []
    mcp_config = {"mcpServers": {}}
    for server in servers:
        name = server.get("name")
        url = server.get("url")
        if name and url:
            mcp_config["mcpServers"][name] = {"url": url}
    return mcp_config


def serialize_message_to_anthropic_jsonl(message) -> dict:
    role = getattr(message, "role", None) or (message.get("role") if isinstance(message, dict) else None)
    content_blocks = []
    raw_content = getattr(message, "content", None) if not isinstance(message, dict) else message.get("content")
    if isinstance(raw_content, str):
        content_blocks.append({"type": "text", "text": raw_content})
    elif isinstance(raw_content, list):
        for block in raw_content:
            block_dict = block if isinstance(block, dict) else _block_to_dict(block)
            if block_dict:
                content_blocks.append(block_dict)
    return {"type": role or "assistant", "content": content_blocks}


def _block_to_dict(block) -> dict:
    block_type = getattr(block, "type", None) or (block.get("type") if isinstance(block, dict) else None)
    if not block_type:
        return {}
    if block_type == "text":
        return {"type": "text", "text": getattr(block, "text", "") or ""}
    if block_type == "tool_use":
        return {
            "type": "tool_use",
            "id": getattr(block, "id", None) or "",
            "name": getattr(block, "name", None) or "",
            "input": getattr(block, "input", None) or {},
        }
    if block_type == "tool_result":
        return {
            "type": "tool_result",
            "tool_use_id": getattr(block, "tool_use_id", None) or "",
            "content": getattr(block, "content", None) or "",
        }
    return {"type": block_type}


def main() -> int:
    if not CLAUDE_CODE_OAUTH_TOKEN:
        print("ERROR: CLAUDE_CODE_OAUTH_TOKEN not set", file=sys.stderr)
        return 1

    if not TASK_INSTRUCTION_PATH.exists():
        print(f"ERROR: instruction.md not found at {TASK_INSTRUCTION_PATH}", file=sys.stderr)
        return 1

    try:
        from openhands.sdk import LLM, Agent, Conversation, Event, LLMConvertibleEvent
        from openhands.sdk.tool import Tool
        from openhands.tools.file_editor import FileEditorTool
        from openhands.tools.terminal import TerminalTool
        from pydantic import SecretStr
    except ImportError as exc:
        print(f"ERROR: openhands-agent-sdk not installed: {exc}", file=sys.stderr)
        return 1

    instruction = TASK_INSTRUCTION_PATH.read_text()
    mcp_config = load_mcp_servers_from_task_toml(TASK_TOML_PATH)

    print(f"openhands-adapter: model={MODEL} max_iters={MAX_ITERATIONS} mcp_servers={len(mcp_config.get('mcpServers', {}))}")

    AGENT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    log_fh = AGENT_LOG_PATH.open("w")

    llm = LLM(
        usage_id="agent",
        model=MODEL,
        api_key=SecretStr(CLAUDE_CODE_OAUTH_TOKEN),
    )

    tools = [Tool(name=TerminalTool.name), Tool(name=FileEditorTool.name)]

    agent = Agent(llm=llm, tools=tools, mcp_config=mcp_config)

    def trajectory_callback(event: Event) -> None:
        if isinstance(event, LLMConvertibleEvent):
            try:
                message = event.to_llm_message()
                record = serialize_message_to_anthropic_jsonl(message)
                log_fh.write(json.dumps(record) + "\n")
                log_fh.flush()
            except Exception as exc:
                print(f"warn: trajectory serialization failed: {exc}", file=sys.stderr)

    conversation = Conversation(
        agent=agent,
        callbacks=[trajectory_callback],
        workspace=str(REPORT_OUTPUT_PATH.parent.parent),
    )

    started_at = time.time()
    try:
        conversation.send_message(instruction)
        conversation.run()
    except Exception as exc:
        print(f"ERROR: OpenHands agent failed: {exc}", file=sys.stderr)
        log_fh.close()
        return 2
    finally:
        elapsed = time.time() - started_at
        cost = getattr(llm.metrics, "accumulated_cost", None)
        print(f"openhands-adapter: elapsed={elapsed:.1f}s cost=${cost or 0:.4f}")

    log_fh.close()

    try:
        AGENT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with AGENT_LOG_PATH.open("r") as read_fh:
            events = []
            for line in read_fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        trajectory = {
            "agent": "openhands",
            "model": MODEL,
            "elapsed_sec": round(time.time() - started_at, 2),
            "cost_usd": round(getattr(llm.metrics, "accumulated_cost", 0.0) or 0.0, 4),
            "mcp_servers_configured": len(mcp_config.get("mcpServers", {})),
            "event_count": len(events),
            "events": events,
        }
        TRAJECTORY_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
        TRAJECTORY_JSON_PATH.write_text(json.dumps(trajectory, indent=2))
        print(f"openhands-adapter: trajectory json -> {TRAJECTORY_JSON_PATH}")
    except Exception as exc:
        print(f"warn: trajectory.json emission failed: {exc}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
