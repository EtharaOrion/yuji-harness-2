"""RED tests for zbridge.errors — GLM error classification + header forwarding.

Every row in PLAN.md §6 has at least one test.
"""
from __future__ import annotations

import pytest

from zbridge.errors import (
    ClassifiedError,
    ErrorKind,
    classify_glm_error,
    forward_upstream_headers,
    to_anthropic_error_response,
)


# ----- Classification -------------------------------------------------------

@pytest.mark.parametrize("code", ["1000", "1001", "1003"])
def test_auth_codes_map_to_oauth_token_invalid(code):
    c = classify_glm_error(401, {"error": {"code": code, "message": "auth"}}, {})
    assert c.kind == ErrorKind.OAUTH_TOKEN_INVALID
    assert c.http_status == 401
    assert c.anthropic_error_type == "authentication_error"


def test_1301_maps_to_waf_blocked_with_hint():
    c = classify_glm_error(400, {"error": {"code": "1301", "message": "content policy"}}, {})
    assert c.kind == ErrorKind.WAF_BLOCKED
    assert c.http_status == 400
    assert c.anthropic_error_type == "invalid_request_error"
    assert "tool" in c.message.lower() or "syntax" in c.message.lower()


def test_1302_transient_throttle_retryable():
    c = classify_glm_error(429, {"error": {"code": "1302", "message": "rate limited"}}, {})
    assert c.kind == ErrorKind.TRANSIENT_THROTTLE
    assert c.kind.is_retryable is True


def test_1305_overloaded_retryable():
    c = classify_glm_error(429, {"error": {"code": "1305", "message": "overloaded"}}, {})
    assert c.kind == ErrorKind.OVERLOADED
    assert c.kind.is_retryable is True


@pytest.mark.parametrize("code", ["1308", "1309", "1315", "1321"])
def test_1308_1321_subscription_cap_not_retryable(code):
    c = classify_glm_error(429, {"error": {"code": code, "message": "quota"}}, {})
    assert c.kind == ErrorKind.SUBSCRIPTION_CAP
    assert c.kind.is_retryable is False


def test_1113_billing_error():
    c = classify_glm_error(429, {"error": {"code": "1113", "message": "balance"}}, {})
    assert c.kind == ErrorKind.BILLING_ERROR
    assert c.anthropic_error_type == "permission_error"
    assert c.http_status == 403


def test_1211_unknown_model_maps_to_not_found_404():
    c = classify_glm_error(400, {"error": {"code": "1211", "message": "unknown model"}}, {})
    assert c.kind == ErrorKind.UNKNOWN_MODEL
    assert c.http_status == 404
    assert c.anthropic_error_type == "not_found_error"


def test_500_upstream_5xx_retryable():
    c = classify_glm_error(500, b"internal server error", {})
    assert c.kind == ErrorKind.UPSTREAM_5XX
    assert c.kind.is_retryable is True


def test_unknown_business_code_falls_back_to_unknown():
    c = classify_glm_error(400, {"error": {"code": "9999", "message": "?"}}, {})
    assert c.kind == ErrorKind.UNKNOWN


def test_classify_handles_bytes_body():
    body_bytes = b'{"error":{"code":"1302","message":"rl"}}'
    c = classify_glm_error(429, body_bytes, {})
    assert c.kind == ErrorKind.TRANSIENT_THROTTLE


def test_classify_handles_none_body_uses_status():
    c = classify_glm_error(500, None, {})
    assert c.kind == ErrorKind.UPSTREAM_5XX


# ----- Anthropic error envelope ---------------------------------------------

def test_to_anthropic_error_envelope_shape():
    c = ClassifiedError(
        kind=ErrorKind.SUBSCRIPTION_CAP,
        http_status=429,
        anthropic_error_type="rate_limit_error",
        message="quota exhausted",
    )
    status, body, headers = to_anthropic_error_response(c)
    assert status == 429
    assert body["type"] == "error"
    assert body["error"]["type"] == "rate_limit_error"
    assert body["error"]["message"] == "quota exhausted"
    assert headers.get("zbridge-error-kind") == "subscription_cap"


def test_error_envelope_retry_after_header_when_present():
    c = ClassifiedError(
        kind=ErrorKind.TRANSIENT_THROTTLE, http_status=429,
        anthropic_error_type="rate_limit_error", message="try later",
        retry_after_seconds=5,
    )
    status, body, headers = to_anthropic_error_response(c)
    assert headers.get("retry-after") == "5" or headers.get("Retry-After") == "5"


# ----- Header forwarding ----------------------------------------------------

def test_forward_upstream_headers_request_id():
    upstream = {"x-request-id": "req-abc-123", "content-type": "application/json"}
    fwd = forward_upstream_headers(upstream)
    assert fwd["zbridge-upstream-x-request-id"] == "req-abc-123"


def test_forward_upstream_headers_ratelimit_variants():
    upstream = {
        "x-ratelimit-limit": "1000",
        "x-ratelimit-remaining": "998",
        "x-ratelimit-reset": "60",
    }
    fwd = forward_upstream_headers(upstream)
    assert fwd["zbridge-upstream-x-ratelimit-limit"] == "1000"
    assert fwd["zbridge-upstream-x-ratelimit-remaining"] == "998"
    assert fwd["zbridge-upstream-x-ratelimit-reset"] == "60"


def test_forward_upstream_headers_retry_after_and_error_kind():
    upstream = {"retry-after": "12"}
    fwd = forward_upstream_headers(upstream, error_kind="transient_throttle")
    # retry-after passes through (with or without prefix — implementation choice; we accept either)
    assert (fwd.get("retry-after") == "12"
            or fwd.get("zbridge-upstream-retry-after") == "12")
    assert fwd["zbridge-error-kind"] == "transient_throttle"


def test_forward_upstream_headers_strips_chunking_artifacts():
    upstream = {
        "content-encoding": "gzip",
        "transfer-encoding": "chunked",
        "content-length": "42",
        "connection": "keep-alive",
    }
    fwd = forward_upstream_headers(upstream)
    for k in ("content-encoding", "transfer-encoding", "content-length", "connection"):
        assert k not in fwd
        assert f"zbridge-upstream-{k}" not in fwd
