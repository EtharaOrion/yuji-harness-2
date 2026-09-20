"""T1 — usage mapping: OpenAI prompt_tokens semantics -> Anthropic usage semantics.

The bug these lock down: z.ai's `prompt_tokens` INCLUDES
`prompt_tokens_details.cached_tokens`, but Anthropic's `input_tokens` EXCLUDES
`cache_read_input_tokens`. Assigning one to the other while also emitting the
cache field makes any consumer that reconstructs
`prompt_tokens = input + cache_read + cache_creation` (LiteLLM does) double-count
every cached token, inflating token counts and cost.

Numbers below are real: the probe/trajectory figures are recorded in the test
names and comments so a regression is traceable to observed upstream behaviour.
"""
from __future__ import annotations

import pytest

from zbridge.translate import (
    DEFAULT_CACHE_BLOCK_TOKENS,
    glm_to_anthropic_response,
    map_usage,
)


def recon(usage: dict) -> int:
    """LiteLLM's Anthropic transform: prompt_tokens = input + read + creation."""
    return (
        usage["input_tokens"]
        + usage.get("cache_read_input_tokens", 0)
        + usage.get("cache_creation_input_tokens", 0)
    )


# ---------------------------------------------------------------------------
# The core invariant
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "prompt,cached",
    [
        (5384, 0),        # probe call 1: forced cache miss
        (5384, 5376),     # probe call 2/3: cache hit, byte-identical prompt
        (7858, 1088),     # flaskr trajectory turn 0
        (141403, 140928), # flaskr trajectory turn 163 (recorded as 282331 pre-fix)
        (8, 3),           # tiny, below one cache block
        (1, 0),
        (0, 0),
    ],
)
@pytest.mark.parametrize("attribution", ["block", "none"])
def test_reconstructed_prompt_tokens_equals_upstream(prompt, cached, attribution):
    """input + cache_read + cache_creation must round-trip to upstream prompt_tokens."""
    usage = map_usage(
        {"prompt_tokens": prompt, "completion_tokens": 7,
         "prompt_tokens_details": {"cached_tokens": cached}},
        attribution=attribution,
    )
    assert recon(usage) == prompt
    assert usage["output_tokens"] == 7


@pytest.mark.parametrize(
    "prompt,cached",
    [(5384, 5376), (7858, 1088), (141403, 140928), (8, 3), (64, 0), (65, 64), (128, 64)],
)
def test_input_tokens_never_zero_when_prompt_has_fresh_tokens(prompt, cached):
    """Explicit requirement: the cache-write split must not zero out input_tokens."""
    usage = map_usage(
        {"prompt_tokens": prompt, "prompt_tokens_details": {"cached_tokens": cached}},
        attribution="block",
    )
    assert usage["input_tokens"] >= 1


def test_no_double_count_on_cache_hit():
    """The exact regression: 5384 upstream must not become 10760 downstream."""
    usage = map_usage(
        {"prompt_tokens": 5384, "completion_tokens": 16,
         "prompt_tokens_details": {"cached_tokens": 5376}},
    )
    assert recon(usage) == 5384, "cached tokens counted twice"
    assert usage["input_tokens"] != 5384, "input_tokens still carries the cached prefix"


# ---------------------------------------------------------------------------
# Attribution modes
# ---------------------------------------------------------------------------

def test_block_attribution_splits_on_cache_block_boundary():
    # prompt 5384, cached 0 -> cacheable prefix = floor(5383/64)*64 = 5376,
    # leaving an 8-token trailing partial block as real input. 5376 is exactly the
    # cached_tokens the probe observed on the next call.
    usage = map_usage(
        {"prompt_tokens": 5384, "prompt_tokens_details": {"cached_tokens": 0}},
        attribution="block",
    )
    assert usage["cache_creation_input_tokens"] == 5376
    assert usage["input_tokens"] == 8
    assert usage["cache_read_input_tokens"] == 0


def test_block_attribution_on_cache_hit_writes_only_the_new_blocks():
    # trajectory turn 163: 141403 total, 140928 already cached.
    # cacheable = floor(141402/64)*64 = 141376 -> 448 newly written, 27 fresh input.
    usage = map_usage(
        {"prompt_tokens": 141403, "prompt_tokens_details": {"cached_tokens": 140928}},
        attribution="block",
    )
    assert usage["cache_read_input_tokens"] == 140928
    assert usage["cache_creation_input_tokens"] == 448
    assert usage["input_tokens"] == 27
    assert recon(usage) == 141403


def test_none_attribution_reports_no_cache_writes():
    usage = map_usage(
        {"prompt_tokens": 5384, "prompt_tokens_details": {"cached_tokens": 5376}},
        attribution="none",
    )
    assert usage["cache_creation_input_tokens"] == 0
    assert usage["input_tokens"] == 8
    assert recon(usage) == 5384


def test_block_smaller_than_one_block_yields_no_write():
    usage = map_usage(
        {"prompt_tokens": 40, "prompt_tokens_details": {"cached_tokens": 0}},
        attribution="block",
    )
    assert usage["cache_creation_input_tokens"] == 0
    assert usage["input_tokens"] == 40


def test_custom_block_size_is_honoured():
    usage = map_usage(
        {"prompt_tokens": 1000, "prompt_tokens_details": {"cached_tokens": 0}},
        attribution="block", block_tokens=256,
    )
    assert usage["cache_creation_input_tokens"] == 768  # floor(999/256)*256
    assert usage["input_tokens"] == 232
    assert recon(usage) == 1000


def test_zero_block_size_degrades_to_no_write_instead_of_dividing_by_zero():
    usage = map_usage(
        {"prompt_tokens": 500, "prompt_tokens_details": {"cached_tokens": 100}},
        attribution="block", block_tokens=0,
    )
    assert usage["cache_creation_input_tokens"] == 0
    assert usage["input_tokens"] == 400


# ---------------------------------------------------------------------------
# Upstream-shape robustness
# ---------------------------------------------------------------------------

def test_cache_fields_omitted_when_upstream_reports_no_caching():
    """Never invent cache numbers for a provider that says nothing about caching."""
    usage = map_usage({"prompt_tokens": 5, "completion_tokens": 1})
    assert "cache_read_input_tokens" not in usage
    assert "cache_creation_input_tokens" not in usage
    assert usage["input_tokens"] == 5


def test_cache_fields_present_when_upstream_reports_zero_cached():
    usage = map_usage(
        {"prompt_tokens": 5, "prompt_tokens_details": {"cached_tokens": 0}},
    )
    assert usage["cache_read_input_tokens"] == 0
    assert usage["cache_creation_input_tokens"] == 0


def test_flat_cached_tokens_form_accepted():
    usage = map_usage({"prompt_tokens": 100, "cached_tokens": 64})
    assert usage["cache_read_input_tokens"] == 64
    assert recon(usage) == 100


def test_cached_exceeding_prompt_is_clamped():
    """Upstream nonsense must not produce negative input_tokens."""
    usage = map_usage(
        {"prompt_tokens": 100, "prompt_tokens_details": {"cached_tokens": 999}},
    )
    assert usage["input_tokens"] >= 0
    assert usage["cache_read_input_tokens"] == 100
    assert recon(usage) == 100


def test_missing_and_null_usage_do_not_raise():
    for bad in (None, {}, {"prompt_tokens": None, "completion_tokens": None}):
        usage = map_usage(bad)
        assert usage["input_tokens"] == 0
        assert usage["output_tokens"] == 0


def test_non_numeric_usage_values_do_not_raise():
    usage = map_usage({"prompt_tokens": "abc", "completion_tokens": "x"})
    assert usage["input_tokens"] == 0
    assert usage["output_tokens"] == 0


def test_default_block_size_is_the_observed_zai_granularity():
    assert DEFAULT_CACHE_BLOCK_TOKENS == 64


# ---------------------------------------------------------------------------
# End-to-end through the response translator
# ---------------------------------------------------------------------------

def test_response_translator_applies_the_fixed_mapping():
    out = glm_to_anthropic_response({
        "id": "x", "model": "glm-5.3",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5384, "completion_tokens": 16,
                  "prompt_tokens_details": {"cached_tokens": 5376}},
    })
    assert recon(out["usage"]) == 5384
    assert out["usage"]["input_tokens"] == 8
    assert out["usage"]["cache_read_input_tokens"] == 5376
    assert out["usage"]["cache_creation_input_tokens"] == 0


def test_response_translator_honours_attribution_override():
    body = {
        "id": "x", "model": "glm-5.3",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5384, "completion_tokens": 16,
                  "prompt_tokens_details": {"cached_tokens": 0}},
    }
    blocked = glm_to_anthropic_response(body, cache_write_attribution="block")
    plain = glm_to_anthropic_response(body, cache_write_attribution="none")
    assert blocked["usage"]["cache_creation_input_tokens"] == 5376
    assert plain["usage"]["cache_creation_input_tokens"] == 0
    assert recon(blocked["usage"]) == recon(plain["usage"]) == 5384
