from __future__ import annotations

import json
import logging
import os
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_API_KEY = "sk-ant-bedrock-judge-local"
MAX_REQUEST_BYTES = 10 * 1024 * 1024
STOP_REASON_MAP = {
    "end_turn": "end_turn",
    "stop_sequence": "stop_sequence",
    "max_tokens": "max_tokens",
    "tool_use": "tool_use",
    "content_filtered": "refusal",
    "guardrail_intervened": "refusal",
}


class BedrockBridgeError(Exception):
    pass


def resolve_bedrock_arn(alias: str) -> str:
    """Look up `alias` in BEDROCK_MODEL_IDS; return the full ARN or alias unchanged."""
    if alias.startswith("arn:aws:"):
        return alias
    raw = os.environ.get("BEDROCK_MODEL_IDS", "")
    for pair in raw.split(","):
        key, _, val = pair.partition("=")
        if key.strip() == alias and val.strip():
            return val.strip()
    raise BedrockBridgeError(
        f"alias '{alias}' not in BEDROCK_MODEL_IDS "
        f"(aliases: {[p.split('=',1)[0].strip() for p in raw.split(',') if '=' in p]})"
    )


class BedrockJudgeServer:
    """One-process Anthropic Messages API server backed by AWS Bedrock Converse.

    `pinned_model` is the judge model alias from JUDGE_MODEL / KIRO_JUDGE_MODEL;
    every /v1/messages request is served by that model regardless of the model
    the caller sent. This matches how kiro_bridge pins the judge key, so the
    rubric judge gets a fixed model irrespective of what judge.toml asks for.

    Backend: boto3's bedrock-runtime.converse. Credentials come from the host's
    AWS_BEARER_TOKEN_BEDROCK (preferred) or standard AWS SigV4 chain. Nothing
    AWS leaves the host; the verifier only ever sees ANTHROPIC_BASE_URL.
    """

    def __init__(self, *, pinned_model: str, host: str = "127.0.0.1",
                 port: int = 0, api_key: str | None = None,
                 aws_region: str | None = None) -> None:
        self.pinned_model = pinned_model
        self.host = host
        self._requested_port = port
        self.api_key = api_key or DEFAULT_API_KEY
        self.aws_region = aws_region or os.environ.get("BEDROCK_AWS_REGION") \
            or os.environ.get("AWS_REGION") or "us-east-1"
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._resolved_arn = resolve_bedrock_arn(pinned_model)
        self._client: Any = None

    @property
    def port(self) -> int:
        if self._server is None:
            raise BedrockBridgeError("server not started")
        return self._server.server_address[1]

    @property
    def resolved_arn(self) -> str:
        return self._resolved_arn

    def start(self) -> BedrockJudgeServer:
        try:
            import boto3
        except ImportError as exc:
            raise BedrockBridgeError(
                "boto3 is not importable; install it on the host "
                "(pip install boto3) to run the Bedrock judge proxy"
            ) from exc
        self._client = boto3.client("bedrock-runtime", region_name=self.aws_region)
        self._server = ThreadingHTTPServer((self.host, self._requested_port),
                                           _make_handler(self))
        self._thread = threading.Thread(target=self._server.serve_forever,
                                        name="bedrock-judge-proxy", daemon=True)
        self._thread.start()
        logger.info("bedrock judge proxy serving on %s:%d -> %s (region %s)",
                    self.host, self.port, self._resolved_arn, self.aws_region)
        return self

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def converse(self, body: dict[str, Any]) -> dict[str, Any]:
        converse_messages, system_blocks = _anthropic_to_converse_messages(body)
        inference_cfg: dict[str, Any] = {}
        if (mt := body.get("max_tokens")) is not None:
            inference_cfg["maxTokens"] = int(mt)
        if (t := body.get("temperature")) is not None:
            inference_cfg["temperature"] = float(t)
        if (tp := body.get("top_p")) is not None:
            inference_cfg["topP"] = float(tp)
        if (stops := body.get("stop_sequences")):
            inference_cfg["stopSequences"] = list(stops)
        kwargs: dict[str, Any] = {
            "modelId": self._resolved_arn,
            "messages": converse_messages,
        }
        if system_blocks:
            kwargs["system"] = system_blocks
        if inference_cfg:
            kwargs["inferenceConfig"] = inference_cfg
        resp = self._client.converse(**kwargs)
        return _converse_to_anthropic(resp, requested_model=body.get("model", self.pinned_model))


def _anthropic_to_converse_messages(body: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Translate Anthropic Messages API body into Converse (messages, system).

    Anthropic sends `system` as a sibling field; Converse takes it as its own
    top-level `system` list. Content may be a bare string OR an array of
    content blocks ({"type": "text", "text": "..."}, ...); Converse always
    takes `content` as a list of {"text": "..."} blocks for text parts.
    Only text content is forwarded in this proxy, matching what the rubric
    judge sends.
    """
    out: list[dict[str, Any]] = []
    for msg in body.get("messages") or []:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role not in ("user", "assistant"):
            continue
        out.append({"role": role, "content": _content_to_converse_blocks(msg.get("content"))})

    system_blocks: list[dict[str, Any]] = []
    sysval = body.get("system")
    if isinstance(sysval, str):
        if sysval.strip():
            system_blocks.append({"text": sysval})
    elif isinstance(sysval, list):
        for b in sysval:
            if isinstance(b, dict) and b.get("type") == "text" and b.get("text"):
                system_blocks.append({"text": b["text"]})
    return out, system_blocks


def _content_to_converse_blocks(content: Any) -> list[dict[str, Any]]:
    if content is None:
        return []
    if isinstance(content, str):
        return [{"text": content}] if content else []
    if not isinstance(content, list):
        return [{"text": str(content)}]
    blocks: list[dict[str, Any]] = []
    for b in content:
        if isinstance(b, str):
            if b:
                blocks.append({"text": b})
        elif isinstance(b, dict) and b.get("type") == "text" and b.get("text"):
            blocks.append({"text": b["text"]})
    return blocks


def _converse_to_anthropic(resp: dict[str, Any], *, requested_model: str) -> dict[str, Any]:
    """Translate Bedrock Converse response into Anthropic Messages API shape.

    Converse `output.message.content` is a list of blocks; we re-wrap each
    text block as {"type": "text", "text": "..."}. Converse `stopReason`
    uses snake-case tokens (`end_turn`, `max_tokens`, ...) which happen to
    overlap with Anthropic's own in most cases; see STOP_REASON_MAP.
    """
    msg = resp.get("output", {}).get("message", {}) or {}
    raw_content = msg.get("content") or []
    content_blocks: list[dict[str, Any]] = []
    for b in raw_content:
        if not isinstance(b, dict):
            continue
        if (text := b.get("text")):
            content_blocks.append({"type": "text", "text": text})
    if not content_blocks:
        content_blocks.append({"type": "text", "text": ""})
    usage = resp.get("usage") or {}
    stop = STOP_REASON_MAP.get(resp.get("stopReason", "end_turn"), "end_turn")
    msg_id = f"msg_bedrock_{secrets.token_hex(8)}"
    return {
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "content": content_blocks,
        "model": requested_model,
        "stop_reason": stop,
        "stop_sequence": None,
        "usage": {
            "input_tokens": int(usage.get("inputTokens", 0) or 0),
            "output_tokens": int(usage.get("outputTokens", 0) or 0),
            "cache_read_input_tokens": int(usage.get("cacheReadInputTokens", 0) or 0),
            "cache_creation_input_tokens": int(usage.get("cacheWriteInputTokens", 0) or 0),
        },
    }


def _make_handler(server: BedrockJudgeServer) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            logger.debug("%s - %s", self.address_string(), format % args)

        def _respond(self, status: int, body: dict[str, Any]) -> None:
            payload = json.dumps(body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _error(self, status: int, type_: str, message: str) -> None:
            self._respond(status, {
                "type": "error",
                "error": {"type": type_, "message": message},
            })

        def do_GET(self) -> None:
            if self.path.rstrip("/") in ("/healthz", "/health", "/"):
                self._respond(200, {"status": "ok",
                                    "pinned_model": server.pinned_model,
                                    "arn": server.resolved_arn,
                                    "region": server.aws_region})
                return
            if self.path.startswith("/v1/models"):
                self._respond(200, {"data": [{"id": server.pinned_model,
                                              "object": "model", "owned_by": "bedrock"}]})
                return
            self._error(404, "not_found_error", f"no route for GET {self.path}")

        def do_POST(self) -> None:
            if not self.path.startswith("/v1/messages"):
                self._error(404, "not_found_error", f"no route for POST {self.path}")
                return
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0 or length > MAX_REQUEST_BYTES:
                self._error(400, "invalid_request_error",
                            f"invalid Content-Length: {length}")
                return
            try:
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise ValueError("request body must be a JSON object")
            except (json.JSONDecodeError, ValueError) as exc:
                self._error(400, "invalid_request_error", f"bad JSON: {exc}")
                return
            started = time.monotonic()
            try:
                reply = server.converse(body)
            except BedrockBridgeError as exc:
                self._error(400, "invalid_request_error", str(exc))
                return
            except Exception as exc:  # noqa: BLE001
                logger.exception("bedrock converse call failed")
                self._error(502, "api_error",
                            f"bedrock converse failed: {type(exc).__name__}: {exc}")
                return
            elapsed_ms = int((time.monotonic() - started) * 1000)
            logger.info("bedrock converse ok: %dms, in=%d out=%d",
                        elapsed_ms,
                        reply.get("usage", {}).get("input_tokens", 0),
                        reply.get("usage", {}).get("output_tokens", 0))
            self._respond(200, reply)

    return Handler
