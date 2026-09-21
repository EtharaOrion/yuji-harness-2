"""The ccbridge changes this harness made on top of the reference bridge.

Pure functions and a stubbed credential store; nothing here reaches Anthropic.
Run with:  cd tools/bridges/ccbridge && uv run --extra dev pytest -q
"""
from __future__ import annotations

import json
import time

import pytest

from claude_oauth import bridge, credentials
from claude_oauth.credentials import CredentialProvider, OAuthCredentials


# --------------------------------------------------------------- thinking ---

def _normalized(body: dict) -> dict:
    return bridge.normalize_body_for_anthropic_direct(json.loads(json.dumps(body)))


def test_adaptive_thinking_passes_through_untouched():
    """What OpenHands SDK 1.49 via LiteLLM sends for claude-opus-5. Rewritten to
    the enabled shape, opus-5 ignores display and returns every block empty."""
    body = {"model": "claude-opus-5", "max_tokens": 16384,
            "thinking": {"type": "adaptive", "display": "summarized"},
            "output_config": {"effort": "high"}}
    assert _normalized(body) == body


def test_the_callers_display_choice_is_left_alone():
    for thinking in ({"type": "adaptive", "display": "omitted"}, {"type": "adaptive"}):
        assert _normalized({"max_tokens": 100, "thinking": thinking})["thinking"] == thinking


def test_convert_mode_restores_the_reference_rewrite(monkeypatch):
    monkeypatch.setenv("WCB_CC_ADAPTIVE_THINKING", "convert")
    out = _normalized({"model": "claude-opus-5", "max_tokens": 16384,
                       "thinking": {"type": "adaptive", "display": "summarized"},
                       "output_config": {"effort": "high"}})
    assert "output_config" not in out
    assert out["thinking"]["type"] == "enabled"
    # 32000 would not fit under max_tokens; the clamp keeps it valid.
    assert bridge.MIN_THINKING_BUDGET <= out["thinking"]["budget_tokens"] < out["max_tokens"]


def test_an_enabled_budget_that_does_not_fit_is_clamped():
    out = _normalized({"max_tokens": 16384, "thinking": {"type": "enabled", "budget_tokens": 32000}})
    assert out["thinking"]["budget_tokens"] == 8192
    assert out["thinking"]["display"] == "summarized"


def test_an_enabled_budget_that_fits_is_left_alone():
    out = _normalized({"max_tokens": 64000, "thinking": {"type": "enabled", "budget_tokens": 32000}})
    assert out["thinking"]["budget_tokens"] == 32000


def test_no_room_to_think_drops_thinking_rather_than_sending_it_invalid():
    out = _normalized({"max_tokens": 1024, "thinking": {"type": "enabled", "budget_tokens": 2048}})
    assert "thinking" not in out


def test_a_request_without_thinking_is_untouched():
    body = {"max_tokens": 10, "messages": []}
    assert _normalized(body) == body


# ------------------------------------------------------------ credentials ---

def _creds(token: str, expires_in_s: float) -> str:
    return json.dumps({"claudeAiOauth": {
        "accessToken": token, "refreshToken": f"rt-{token}",
        "expiresAt": int((time.time() + expires_in_s) * 1000), "scopes": ["user:inference"],
    }})


@pytest.fixture()
def stores(monkeypatch):
    """Every credential source, each returning whatever the test puts in it."""
    box: dict[str, str | None] = {"file": None, "keychain": None, "secret": None, "cache": None}
    monkeypatch.delenv("CLAUDE_CODE_CREDENTIALS", raising=False)
    monkeypatch.delenv("WCB_CC_CREDS_PATH", raising=False)
    monkeypatch.setattr(credentials, "_read_credentials_file", lambda: box["file"])
    monkeypatch.setattr(credentials, "_read_keychain_macos", lambda: box["keychain"])
    monkeypatch.setattr(credentials, "_read_secretservice_linux", lambda: box["secret"])
    monkeypatch.setattr(credentials, "_read_cache_file", lambda: box["cache"])
    return box


def test_the_freshest_store_wins(stores):
    stores["keychain"] = _creds("stale", -60)
    stores["cache"] = _creds("fresh", 3600)
    assert credentials.load_credentials().access_token == "fresh"


def test_an_explicit_override_wins_outright(stores, monkeypatch):
    stores["keychain"] = _creds("fresh", 3600)
    monkeypatch.setenv("CLAUDE_CODE_CREDENTIALS", _creds("pinned", 60))
    assert credentials.load_credentials().access_token == "pinned"


def test_expiry_rereads_the_stores_before_refreshing(stores, monkeypatch):
    """The CLI already refreshed: use its token, spend no refresh token."""
    stores["keychain"] = _creds("old", 3600)
    provider = CredentialProvider()
    assert provider.get_access_token() == "old"
    provider._creds = OAuthCredentials.from_claude_payload(json.loads(_creds("old", -60)))
    stores["keychain"] = _creds("cli-refreshed", 3600)
    monkeypatch.setattr(credentials, "refresh_credentials",
                        lambda c: pytest.fail("refreshed although the store had a live token"))
    assert provider.get_access_token() == "cli-refreshed"


def test_refresh_happens_only_when_every_store_is_expired(stores, monkeypatch):
    stores["keychain"] = _creds("old", -60)
    seen = []

    def fake_refresh(c):
        seen.append(c.refresh_token)
        return OAuthCredentials("new", "rt-new", int((time.time() + 3600) * 1000), [])

    monkeypatch.setattr(credentials, "refresh_credentials", fake_refresh)
    monkeypatch.setattr(credentials, "write_cache", lambda c: None)
    assert CredentialProvider().get_access_token() == "new"
    assert seen == ["rt-old"]
