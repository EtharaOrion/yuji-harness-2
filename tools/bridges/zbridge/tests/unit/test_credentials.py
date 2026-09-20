"""RED tests for zbridge.credentials — KeyProvider protocol + EnvKeyProvider."""
from __future__ import annotations

import pytest

from zbridge.credentials import EnvKeyProvider, KeyProvider


def test_env_key_provider_reads_configured_env(monkeypatch):
    monkeypatch.setenv("ZB_ZAI_API_KEY", "test-key-abc")
    p = EnvKeyProvider()
    assert p.get() == "test-key-abc"


def test_env_key_provider_custom_var(monkeypatch):
    monkeypatch.setenv("MY_KEY", "custom-abc")
    p = EnvKeyProvider(env_var="MY_KEY")
    assert p.get() == "custom-abc"


def test_env_key_provider_missing_env_raises(monkeypatch):
    monkeypatch.delenv("ZB_ZAI_API_KEY", raising=False)
    p = EnvKeyProvider()
    with pytest.raises(Exception):  # Any exception acceptable — specific class up to impl
        p.get()


def test_env_key_provider_conforms_to_protocol():
    p: KeyProvider = EnvKeyProvider()  # will only type-check at runtime; verifies Protocol import
    assert hasattr(p, "get") and callable(p.get)
