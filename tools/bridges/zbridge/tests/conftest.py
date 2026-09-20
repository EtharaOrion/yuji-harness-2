"""Shared pytest fixtures for zbridge tests."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "synth"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture
def load_json(fixtures_dir):
    def _load(name: str) -> dict:
        return json.loads((fixtures_dir / name).read_text())
    return _load


@pytest.fixture
def load_sse(fixtures_dir):
    def _load(name: str) -> bytes:
        return (fixtures_dir / name).read_bytes()
    return _load
