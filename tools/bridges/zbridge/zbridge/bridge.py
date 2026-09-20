"""FastAPI app: Anthropic /v1/messages -> z.ai GLM Coding Plan.

Contract per PLAN.md §3, §5-§7. Two streaming modes:
  - passthrough  (ZB_BUFFER_AND_RETRY=0)  chunk-by-chunk translation
  - buffered    (ZB_BUFFER_AND_RETRY=1)  full capture, retry-on-drop, atomic replay (default)
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import random
import secrets
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from .credentials import CredentialsError, EnvKeyProvider, KeyProvider
from .errors import (
    ClassifiedError,
    ErrorKind,
    classify_glm_error,
    forward_upstream_headers,
    to_anthropic_error_response,
)
from .sse_translator import SseTranslator
from .stream_tee import StreamTee
from .translate import (
    CACHE_WRITE_ATTRIBUTIONS,
    DEFAULT_CACHE_BLOCK_TOKENS,
    TranslationError,
    anthropic_to_glm_request,
    glm_to_anthropic_response,
    map_usage,
)

_LOG = logging.getLogger("zbridge")

DEFAULT_UPSTREAM = "https://api.z.ai/api/coding/paas/v4/chat/completions"
DEFAULT_MODEL = "glm-5.3"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "").strip() or default)
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip() or default)
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name, "").strip().lower()
    if not v:
        return default
    return v not in ("0", "false", "no", "off")


class Config:
    def __init__(self):
        self.upstream_url = os.environ.get("ZB_UPSTREAM_URL", DEFAULT_UPSTREAM)
        self.default_model = os.environ.get("ZB_DEFAULT_MODEL", DEFAULT_MODEL)
        self.model_alias = self._load_alias()
        self.bridge_secret = os.environ.get("ZB_BRIDGE_SECRET", "").strip()
        self.stream_log_path = os.environ.get("ZB_STREAM_LOG_PATH", "").strip() or None
        self.buffer_and_retry = _env_bool("ZB_BUFFER_AND_RETRY", True)
        self.buffer_retries = _env_int("ZB_STREAM_BUFFER_RETRIES", 3)
        self.read_timeout_nonstream = _env_float("ZB_READ_TIMEOUT_NONSTREAM_S", 180.0)
        self.read_timeout_stream = _env_float("ZB_READ_TIMEOUT_STREAM_S", 600.0)
        self.max_inline_retries = _env_int("ZB_MAX_INLINE_RETRIES", 3)
        self.max_inline_wait = _env_int("ZB_MAX_INLINE_WAIT_S", 30)
        self.thinking_sig_key = os.environ.get("ZB_THINKING_SIG_KEY", "").strip().encode() or None
        self.preserve_thinking = _env_bool("ZB_PRESERVE_THINKING_IN_CONTEXT", False)
        self.ping_interval = _env_float("ZB_PING_INTERVAL_S", 10.0)
        self.connect_timeout = _env_float("ZB_CONNECT_TIMEOUT_S", 30.0)
        self.cache_write_attribution = self._load_cache_attribution()
        self.cache_block_tokens = _env_int("ZB_CACHE_BLOCK_TOKENS", DEFAULT_CACHE_BLOCK_TOKENS)

    @staticmethod
    def _load_cache_attribution() -> str:
        v = os.environ.get("ZB_CACHE_WRITE_ATTRIBUTION", "").strip().lower() or "block"
        if v not in CACHE_WRITE_ATTRIBUTIONS:
            _LOG.warning(
                "ZB_CACHE_WRITE_ATTRIBUTION=%r is not one of %s; falling back to 'block'",
                v, CACHE_WRITE_ATTRIBUTIONS,
            )
            return "block"
        return v

    def _load_alias(self) -> dict[str, str]:
        raw = os.environ.get("ZB_MODEL_ALIAS_JSON", "").strip()
        if not raw:
            return {}
        try:
            v = json.loads(raw)
            return v if isinstance(v, dict) else {}
        except (json.JSONDecodeError, TypeError):
            return {}


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

_AUTH_HEADER_NAMES = ("x-zbridge-secret", "x-api-key", "authorization")


def _extract_client_secret(request: Request) -> str:
    for h in _AUTH_HEADER_NAMES:
        v = request.headers.get(h, "")
        if not v:
            continue
        if h == "authorization" and v.lower().startswith("bearer "):
            return v[len("bearer "):].strip()
        return v.strip()
    return ""


def _auth_ok(cfg: Config, request: Request) -> bool:
    if not cfg.bridge_secret:
        return True  # Unauthenticated mode (with startup warning)
    return _extract_client_secret(request) == cfg.bridge_secret


def _unauth_response() -> JSONResponse:
    return JSONResponse(
        {"type": "error", "error": {"type": "authentication_error", "message": "invalid or missing bridge secret"}},
        status_code=401,
    )


# ---------------------------------------------------------------------------
# Retry helper (jittered exp backoff)
# ---------------------------------------------------------------------------

def _backoff_seconds(attempt: int, base: float = 1.0, factor: float = 2.0, cap: float = 60.0) -> float:
    wait = min(cap, base * (factor ** attempt))
    jitter = wait * random.uniform(-0.2, 0.2)
    return max(0.1, wait + jitter)


# ---------------------------------------------------------------------------
# Non-streaming path (T15a)
# ---------------------------------------------------------------------------

async def _forward_nonstream(
    cfg: Config,
    key_provider: KeyProvider,
    glm_body: dict[str, Any],
    client: httpx.AsyncClient,
) -> Response:
    attempt = 0
    last_error: ClassifiedError | None = None
    last_headers: dict[str, str] = {}

    while attempt <= cfg.max_inline_retries:
        try:
            api_key = key_provider.get()
        except CredentialsError as e:
            return JSONResponse(
                {"type": "error", "error": {"type": "authentication_error", "message": str(e)}},
                status_code=500,
            )

        try:
            resp = await client.post(
                cfg.upstream_url,
                json=glm_body,
                headers={
                    "authorization": f"Bearer {api_key}",
                    "content-type": "application/json",
                    "accept": "application/json",
                    "accept-language": "en-US,en",
                },
                timeout=httpx.Timeout(cfg.read_timeout_nonstream, connect=cfg.connect_timeout),
            )
        except httpx.HTTPError as e:
            last_error = ClassifiedError(
                kind=ErrorKind.UPSTREAM_5XX, http_status=502,
                anthropic_error_type="api_error",
                message=f"upstream network error: {e}",
            )
            _LOG.warning("upstream network error attempt=%d: %s", attempt, e)
            if attempt >= cfg.max_inline_retries:
                break
            await asyncio.sleep(_backoff_seconds(attempt))
            attempt += 1
            continue

        upstream_hdrs = dict(resp.headers)

        if 200 <= resp.status_code < 300:
            try:
                glm_body_out = resp.json()
            except json.JSONDecodeError:
                return _err_response(ClassifiedError(
                    kind=ErrorKind.UNKNOWN, http_status=502,
                    anthropic_error_type="api_error",
                    message="upstream returned non-JSON success body",
                ), upstream_hdrs)
            try:
                anth = glm_to_anthropic_response(
                    glm_body_out,
                    thinking_sig_key=cfg.thinking_sig_key,
                    cache_write_attribution=cfg.cache_write_attribution,
                    cache_block_tokens=cfg.cache_block_tokens,
                )
            except Exception as e:  # noqa: BLE001
                _LOG.exception("response translation failed")
                return _err_response(ClassifiedError(
                    kind=ErrorKind.UNKNOWN, http_status=500,
                    anthropic_error_type="api_error",
                    message=f"zbridge: response translation failed: {e}",
                ), upstream_hdrs)
            fwd_hdrs = forward_upstream_headers(upstream_hdrs)
            return JSONResponse(anth, status_code=200, headers=fwd_hdrs)

        # Non-2xx: classify
        classified = classify_glm_error(resp.status_code, resp.content, upstream_hdrs)
        last_error = classified
        last_headers = upstream_hdrs
        _LOG.info("upstream error status=%d kind=%s", resp.status_code, classified.kind.value)

        if classified.kind.is_retryable and attempt < cfg.max_inline_retries:
            wait = min(cfg.max_inline_wait, _backoff_seconds(attempt))
            await asyncio.sleep(wait)
            attempt += 1
            continue

        return _err_response(classified, upstream_hdrs)

    # Loop exhausted
    if last_error is None:
        last_error = ClassifiedError(
            kind=ErrorKind.UNKNOWN, http_status=502,
            anthropic_error_type="api_error", message="zbridge: retries exhausted",
        )
    return _err_response(last_error, last_headers)


def _err_response(classified: ClassifiedError, upstream_hdrs: dict[str, str]) -> JSONResponse:
    status, body, err_hdrs = to_anthropic_error_response(classified)
    fwd = forward_upstream_headers(upstream_hdrs, error_kind=classified.kind.value)
    fwd.update(err_hdrs)
    return JSONResponse(body, status_code=status, headers=fwd)


# ---------------------------------------------------------------------------
# Streaming path (T15b) — passthrough + buffered modes
# ---------------------------------------------------------------------------

_PING_FRAME = b"event: ping\ndata: {\"type\":\"ping\"}\n\n"
_SSE_MEDIA = "text/event-stream"


async def _forward_stream_passthrough(
    cfg: Config,
    key_provider: KeyProvider,
    glm_body: dict[str, Any],
    client: httpx.AsyncClient,
    request_id: str,
    tee: StreamTee | None,
) -> Response:
    api_key = key_provider.get()

    async def gen():
        translator = SseTranslator(
            model=cfg.default_model,
            thinking_sig_key=cfg.thinking_sig_key,
            usage_mapper=lambda u: map_usage(
                u, cfg.cache_write_attribution, cfg.cache_block_tokens
            ),
        )
        try:
            async with client.stream(
                "POST", cfg.upstream_url,
                json=glm_body,
                headers={
                    "authorization": f"Bearer {api_key}",
                    "content-type": "application/json",
                    "accept": "text/event-stream",
                    "accept-language": "en-US,en",
                },
                timeout=httpx.Timeout(None, connect=cfg.connect_timeout, read=cfg.read_timeout_stream),
            ) as up:
                if not (200 <= up.status_code < 300):
                    body_bytes = b""
                    async for chunk in up.aiter_bytes():
                        body_bytes += chunk
                        if len(body_bytes) > 65536:
                            break
                    classified = classify_glm_error(up.status_code, body_bytes, dict(up.headers))
                    yield _sse_error_frame(classified.anthropic_error_type, classified.message)
                    if tee is not None:
                        tee.event(kind="error", event="error", delta=classified.message[:200])
                    return
                async for chunk in up.aiter_bytes():
                    for out in translator.feed(chunk):
                        if tee is not None:
                            tee.feed_bytes = getattr(tee, "feed_bytes", None)  # sentinel; no-op
                        yield out
                for out in translator.close():
                    yield out
        except httpx.HTTPError as e:
            _LOG.warning("passthrough stream failed: %s", e)
            for out in translator.close():
                yield out
            yield _sse_error_frame("api_error", f"zbridge: passthrough failure: {e}")
            if tee is not None:
                tee.event(kind="error", event="error", delta=str(e)[:200])

    return StreamingResponse(
        gen(),
        media_type=_SSE_MEDIA,
        headers={"zbridge-stream-mode": "passthrough"},
    )


async def _capture_stream(
    cfg: Config,
    key_provider: KeyProvider,
    glm_body: dict[str, Any],
    client: httpx.AsyncClient,
) -> tuple[str, bytes, dict[str, str], ClassifiedError | None]:
    """Capture the FULL GLM SSE stream. Retry on mid-stream drop up to buffer_retries.

    Returns (kind, buffered_bytes, upstream_headers, classified_error_or_None)
    where kind in {"ok", "http_error", "network_drop_exhausted"}.
    """
    attempt = 0
    last_hdrs: dict[str, str] = {}
    while attempt <= cfg.buffer_retries:
        try:
            api_key = key_provider.get()
        except CredentialsError as e:
            return "http_error", b"", {}, ClassifiedError(
                kind=ErrorKind.UNKNOWN, http_status=500,
                anthropic_error_type="authentication_error", message=str(e),
            )

        buf = bytearray()
        saw_terminal = False
        tail = b""
        try:
            async with client.stream(
                "POST", cfg.upstream_url,
                json=glm_body,
                headers={
                    "authorization": f"Bearer {api_key}",
                    "content-type": "application/json",
                    "accept": "text/event-stream",
                    "accept-language": "en-US,en",
                },
                timeout=httpx.Timeout(None, connect=cfg.connect_timeout, read=cfg.read_timeout_stream),
            ) as up:
                last_hdrs = dict(up.headers)
                if not (200 <= up.status_code < 300):
                    body = b""
                    async for c in up.aiter_bytes():
                        body += c
                        if len(body) > 65536:
                            break
                    classified = classify_glm_error(up.status_code, body, last_hdrs)
                    if classified.kind.is_retryable and attempt < cfg.buffer_retries:
                        await asyncio.sleep(_backoff_seconds(attempt))
                        attempt += 1
                        continue
                    return "http_error", b"", last_hdrs, classified

                async for chunk in up.aiter_bytes():
                    buf += chunk
                    tail = (tail + chunk)[-512:]
                    if b"[DONE]" in tail or b'"finish_reason"' in tail:
                        saw_terminal = True
        except httpx.HTTPError as e:
            _LOG.warning("buffered capture drop attempt=%d: %s", attempt, e)

        if saw_terminal or b"[DONE]" in buf or b'"finish_reason"' in buf:
            return "ok", bytes(buf), last_hdrs, None

        attempt += 1
        if attempt > cfg.buffer_retries:
            break
        await asyncio.sleep(_backoff_seconds(attempt))

    return "network_drop_exhausted", b"", last_hdrs, ClassifiedError(
        kind=ErrorKind.UPSTREAM_5XX, http_status=502,
        anthropic_error_type="api_error",
        message="zbridge: upstream stream incomplete after retries",
    )


async def _forward_stream_buffered(
    cfg: Config,
    key_provider: KeyProvider,
    glm_body: dict[str, Any],
    client: httpx.AsyncClient,
    request_id: str,
    tee: StreamTee | None,
) -> Response:
    async def gen():
        capture_task = asyncio.create_task(_capture_stream(cfg, key_provider, glm_body, client))
        try:
            while not capture_task.done():
                try:
                    await asyncio.wait_for(asyncio.shield(capture_task), timeout=cfg.ping_interval)
                except TimeoutError:
                    yield _PING_FRAME
                    continue
            kind, buffered, upstream_hdrs, classified = capture_task.result()
        except asyncio.CancelledError:
            capture_task.cancel()
            raise

        if kind == "ok":
            translator = SseTranslator(
                model=cfg.default_model,
                thinking_sig_key=cfg.thinking_sig_key,
                usage_mapper=lambda u: map_usage(
                    u, cfg.cache_write_attribution, cfg.cache_block_tokens
                ),
            )
            for out in translator.feed(buffered):
                if tee is not None:
                    tee.event(kind="stream", event="chunk", delta=str(len(out)))
                yield out
            for out in translator.close():
                yield out
            if tee is not None:
                tee.event(kind="status", event="message_stop")
        else:
            err_type = classified.anthropic_error_type if classified else "api_error"
            err_msg = classified.message if classified else "zbridge: buffered stream failed"
            yield _sse_error_frame(err_type, err_msg)
            if tee is not None:
                tee.event(kind="error", event="error", delta=err_msg[:200])

    return StreamingResponse(
        gen(),
        media_type=_SSE_MEDIA,
        headers={"zbridge-stream-mode": "buffered"},
    )


def _sse_error_frame(err_type: str, message: str) -> bytes:
    return (
        b"event: error\ndata: "
        + json.dumps({"type": "error", "error": {"type": err_type, "message": message}}).encode("utf-8")
        + b"\n\n"
    )


# ---------------------------------------------------------------------------
# App builder
# ---------------------------------------------------------------------------

def build_app(
    key_provider: KeyProvider | None = None,
    config: Config | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> FastAPI:
    cfg = config or Config()
    key_prov: KeyProvider = key_provider or EnvKeyProvider()

    if not cfg.bridge_secret:
        _LOG.warning(
            "ZB_BRIDGE_SECRET is not set — the bridge is UNAUTHENTICATED; any "
            "local process can spend the z.ai subscription. Set it to lock down."
        )
    if cfg.thinking_sig_key is None:
        _LOG.warning(
            "ZB_THINKING_SIG_KEY is empty — thinking-block signatures will be blank. "
            "Set to a stable value for reproducible trajectory bytes."
        )

    own_client = http_client is None
    client_instance = http_client or httpx.AsyncClient()

    @asynccontextmanager
    async def _lifespan(_app: FastAPI):
        try:
            yield
        finally:
            if own_client:
                with contextlib.suppress(Exception):
                    await client_instance.aclose()

    app = FastAPI(title="zbridge", version="0.1.0", lifespan=_lifespan)
    app.state._client = client_instance

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        return JSONResponse({
            "ok": True,
            "upstream": cfg.upstream_url,
            "model_default": cfg.default_model,
            "stream_mode": "buffered" if cfg.buffer_and_retry else "passthrough",
            "cache_write_attribution": cfg.cache_write_attribution,
            "cache_block_tokens": cfg.cache_block_tokens,
        })

    @app.post("/v1/messages")
    async def messages(request: Request) -> Response:
        if not _auth_ok(cfg, request):
            return _unauth_response()

        raw = await request.body()
        try:
            body = json.loads(raw)
        except json.JSONDecodeError as e:
            return JSONResponse(
                {"type": "error", "error": {"type": "invalid_request_error", "message": f"invalid JSON: {e}"}},
                status_code=400,
            )

        # Translate request
        try:
            glm_body = anthropic_to_glm_request(
                body,
                model_alias=cfg.model_alias,
                preserve_thinking=cfg.preserve_thinking,
            )
        except TranslationError as e:
            return JSONResponse(
                {"type": "error", "error": {"type": e.error_type, "message": str(e)}},
                status_code=400,
            )
        except Exception as e:  # noqa: BLE001
            _LOG.exception("request translation failed")
            return JSONResponse(
                {"type": "error", "error": {"type": "invalid_request_error",
                                             "message": f"zbridge: request translation failed: {e}"}},
                status_code=400,
            )

        # Ensure default model if none set
        glm_body.setdefault("model", cfg.default_model)

        streaming = bool(glm_body.get("stream"))
        req_id = secrets.token_hex(8)

        tee = None
        if cfg.stream_log_path:
            tee = StreamTee(
                path=cfg.stream_log_path, source="agent",
                request_id=req_id, model=glm_body.get("model", ""),
            )
            if streaming:
                tee.event(kind="status", event="message_start")

        try:
            if streaming:
                if cfg.buffer_and_retry:
                    return await _forward_stream_buffered(
                        cfg, key_prov, glm_body, app.state._client, req_id, tee)
                return await _forward_stream_passthrough(
                    cfg, key_prov, glm_body, app.state._client, req_id, tee)
            return await _forward_nonstream(cfg, key_prov, glm_body, app.state._client)
        finally:
            if tee is not None and not streaming:
                tee.close()

    return app
