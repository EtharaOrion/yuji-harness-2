"""RED tests for zbridge.stream_tee — JSONL schema parity with ccbridge."""
from __future__ import annotations

import json

from zbridge.stream_tee import StreamTee

CCBRIDGE_SCHEMA_KEYS = {"ts", "seq", "source", "request_id", "model", "kind", "event", "delta"}


def test_event_writes_valid_jsonl_row(tmp_path):
    p = tmp_path / "tee.jsonl"
    tee = StreamTee(path=str(p), source="agent", request_id="req-1", model="glm-5.3")
    tee.event(kind="text", event="delta", delta="hello")
    tee.close()
    lines = p.read_text().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert set(row.keys()) == CCBRIDGE_SCHEMA_KEYS


def test_seq_increments_across_events(tmp_path):
    p = tmp_path / "tee.jsonl"
    tee = StreamTee(path=str(p), source="agent", request_id="req-1", model="glm-5.3")
    tee.event(kind="status", event="message_start")
    tee.event(kind="text", event="delta", delta="a")
    tee.event(kind="text", event="delta", delta="b")
    tee.close()
    rows = [json.loads(line) for line in p.read_text().splitlines()]
    seqs = [r["seq"] for r in rows]
    assert seqs == sorted(seqs) and seqs[0] == 0 and len(set(seqs)) == len(seqs)


def test_none_path_is_no_op(tmp_path):
    tee = StreamTee(path=None, source="agent", request_id="req-1", model="glm-5.3")
    tee.event(kind="text", event="delta", delta="hi")  # must not raise
    tee.close()


def test_ts_is_iso_or_float(tmp_path):
    p = tmp_path / "tee.jsonl"
    tee = StreamTee(path=str(p), source="agent", request_id="req-1", model="glm-5.3")
    tee.event(kind="status", event="message_start")
    tee.close()
    row = json.loads(p.read_text().splitlines()[0])
    ts = row["ts"]
    # Accept float (unix seconds) or ISO string
    assert isinstance(ts, (int, float)) or (isinstance(ts, str) and len(ts) >= 10)
