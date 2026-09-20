#!/usr/bin/env python3
"""Compress the AGENT's prompt messages, and nothing else.

WHY THIS EXISTS INSTEAD OF `headroom proxy`
===========================================
The headroom proxy rewrites the whole Anthropic request, and that killed two
runs. Measured through it, with the payload Claude Code actually sends:

    61 tools in  -> 60 out          the `tool_search_tool_regex` entry deleted
    server_tool_use / tool_search_tool_result blocks   stripped

Claude Code carries ~206 MCP tools here, so partway through a session it
switches to deferred tool loading (measured: assistant turn 10 of run_8). From
that turn on the conversation references a tool the rewritten request no longer
declares, and the API answers `400 Tool reference 'tool_search_tool_regex' not
found in available tools`. The session ends; harbor publishes a reward of 0.

So this proxy does the opposite of clever. It forwards the request byte for
byte except for one field, `messages`, and only when the result passes a
structural check. Everything else -- tools, system, betas, metadata, headers,
the response stream -- is relayed untouched.

WHAT "SAFE" MEANS HERE
======================
Compression may shorten text. It may not change the SHAPE of the conversation.
`_skeleton()` captures every non-text block (tool_use and its id, tool_result
and its tool_use_id, thinking, image, server_tool_use, anything else) per
message; if the skeleton moves, the original body is sent instead. That is what
would have caught the failure above before it reached the API.

On any doubt at all -- unparsable body, a raising compressor, a result that is
larger, a moved skeleton -- the ORIGINAL bytes go upstream. After
AGENT_COMPRESS_DISABLE_AFTER consecutive failures the proxy stops trying for
the rest of the run and becomes a plain relay, because a compressor that is
failing is not worth a per-request gamble on a paid agent phase.
"""
from __future__ import annotations

import copy
import json
import os
import sys
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# This container IS the compressor, so the shared module's own opt-in switch is
# on by definition. grader_compress reads it at call time.
os.environ.setdefault("GRADER_HEADROOM_ENABLED", "true")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from grader_compress import compress_messages  # type: ignore
except Exception as exc:  # pragma: no cover - the image always ships it
    compress_messages = None
    print(f"[agent_compress] grader_compress unavailable ({exc}); relaying only", flush=True)

UPSTREAM = os.environ.get("ANTHROPIC_TARGET_API_URL", "https://api.anthropic.com").rstrip("/")
PORT = int(os.environ.get("PORT", "8787"))
DISABLE_AFTER = int(os.environ.get("AGENT_COMPRESS_DISABLE_AFTER", "3"))
TIMEOUT = float(os.environ.get("AGENT_COMPRESS_TIMEOUT", "900"))

# Hop-by-hop headers plus the two we always recompute. Relaying Host or a stale
# Content-Length is how a proxy turns a valid request into a 400 of its own.
_HOP = frozenset({
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailer", "transfer-encoding", "upgrade", "host", "content-length",
})

_state_lock = threading.Lock()
_failures = 0
_disabled = False


def _note_failure(why: str) -> None:
    global _failures, _disabled
    with _state_lock:
        _failures += 1
        print(f"[agent_compress] passing through uncompressed: {why}", flush=True)
        if not _disabled and _failures >= DISABLE_AFTER:
            _disabled = True
            print(f"[agent_compress] {_failures} failures; compression OFF for the "
                  "rest of this run, relaying only", flush=True)


def _note_success() -> None:
    global _failures
    with _state_lock:
        _failures = 0


def _skeleton(message):
    """Everything about a message that compression must not change.

    Text is deliberately absent: shortening it is the whole point. Block ids
    are included because a tool_result whose tool_use_id no longer matches a
    tool_use is a 400 from the API, not a smaller prompt.
    """
    if not isinstance(message, dict):
        return ("?",)
    content = message.get("content")
    blocks = []
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                blocks.append(("?", None))
                continue
            kind = block.get("type")
            if kind == "text":
                continue
            blocks.append((kind, block.get("id") or block.get("tool_use_id") or block.get("name")))
    elif not isinstance(content, str) and content is not None:
        blocks.append(("?", None))
    return (message.get("role"), tuple(blocks))


def _cache_markers(messages):
    """Positional cache_control markers, so they survive compression.

    Prompt caching needs a byte-stable prefix, not an uncompressed one, but the
    marker has to be somewhere. Compression flattens a block list to a bare
    string, which leaves the breakpoint nowhere to live, and losing it silently
    forfeits the discount at exactly the point that mattered.
    """
    out = {}
    for i, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("cache_control") is not None:
                out[i] = block["cache_control"]
                break
    return out


def _restore_markers(messages, markers):
    for i, marker in markers.items():
        if i >= len(messages) or not isinstance(messages[i], dict):
            continue
        content = messages[i].get("content")
        if isinstance(content, str):
            messages[i]["content"] = [{"type": "text", "text": content, "cache_control": marker}]
        elif isinstance(content, list) and content and isinstance(content[-1], dict):
            if not any(isinstance(b, dict) and b.get("cache_control") for b in content):
                content[-1]["cache_control"] = marker


def compress_request(raw: bytes) -> bytes:
    """Return the body to send upstream: compressed if provably safe, else `raw`."""
    if _disabled or compress_messages is None:
        return raw
    try:
        body = json.loads(raw)
    except Exception:
        return raw   # not JSON; nothing to do, and not an error worth counting
    if not isinstance(body, dict):
        return raw
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        return raw

    try:
        probe = copy.deepcopy(messages)
        markers = _cache_markers(probe)
        for message in probe:                      # the module stands down on
            if isinstance(message, dict):          # cached traffic; this path
                content = message.get("content")   # keeps caching AND asks for
                if isinstance(content, list):      # compression, so strip the
                    for block in content:          # markers off the copy and
                        if isinstance(block, dict):# put them back after.
                            block.pop("cache_control", None)
        out = compress_messages(body.get("model", ""), probe)
    except Exception as exc:
        _note_failure(f"compressor raised {exc!r}")
        return raw

    if not isinstance(out, list) or len(out) != len(messages):
        _note_failure("compressor changed the number of messages")
        return raw
    for before, after in zip(messages, out):
        if _skeleton(before) != _skeleton(after):
            _note_failure("compressor changed the shape of a message "
                          f"({_skeleton(before)} -> {_skeleton(after)})")
            return raw

    _restore_markers(out, markers)
    body["messages"] = out
    try:
        new = json.dumps(body).encode()
    except Exception as exc:
        _note_failure(f"result would not serialise ({exc!r})")
        return raw
    if len(new) >= len(raw):
        return raw   # no saving is not a failure, it is the common case
    _note_success()
    print(f"[agent_compress] {len(raw)} -> {len(new)} bytes "
          f"(-{100.0 * (len(raw) - len(new)) / len(raw):.1f}%)", flush=True)
    return new


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "agent-compress"

    def log_message(self, *args):  # the access log is squid's job
        pass

    def do_GET(self):
        if self.path.rstrip("/") in ("/health", "/healthz"):
            payload = json.dumps({"ok": True, "upstream": UPSTREAM,
                                  "compressing": not _disabled}).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        self._proxy(b"")

    def do_POST(self):
        length = int(self.headers.get("content-length") or 0)
        self._proxy(self.rfile.read(length) if length else b"")

    def _proxy(self, raw: bytes) -> None:
        body = compress_request(raw) if raw else raw
        headers = {k: v for k, v in self.headers.items() if k.lower() not in _HOP}
        if raw:
            headers["Content-Length"] = str(len(body))
        req = urllib.request.Request(UPSTREAM + self.path, data=body or None,
                                     headers=headers, method=self.command)
        try:
            resp = urllib.request.urlopen(req, timeout=TIMEOUT)
        except urllib.error.HTTPError as err:
            # An upstream error is an answer, not a failure of ours: relay it
            # whole so the agent sees what the API actually said.
            resp = err
        except Exception as exc:
            payload = json.dumps({"type": "error", "error": {
                "type": "api_error", "message": f"agent-compress could not reach upstream: {exc}"}}).encode()
            self.send_response(502)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        with resp:
            self.send_response(resp.status)
            passed = {}
            for key, value in resp.headers.items():
                if key.lower() in _HOP:
                    continue
                passed[key.lower()] = value
                self.send_header(key, value)
            streaming = "content-length" not in passed
            if streaming:
                self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            while True:
                chunk = resp.read(8192)
                if not chunk:
                    break
                if streaming:
                    self.wfile.write(b"%X\r\n" % len(chunk) + chunk + b"\r\n")
                else:
                    self.wfile.write(chunk)
                self.wfile.flush()
            if streaming:
                self.wfile.write(b"0\r\n\r\n")


def main() -> None:
    print(f"[agent_compress] listening on :{PORT}, upstream {UPSTREAM}, "
          f"compressor {'ready' if compress_messages else 'ABSENT'}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
