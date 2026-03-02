"""Tests for _build_top_hypotheses in fast_agent.py and engine.py."""

from __future__ import annotations

import pytest

from x_creative.core.types import Hypothesis, VerifyStatus
from x_creative.creativity.engine import (
    _build_top_hypotheses as engine_build,
)
from x_creative.saga.fast_agent import (
    _build_top_hypotheses as saga_build,
)


def _hyp(
    hypothesis_id: str,
    *,
    final_score: float | None = None,
    description: str = "desc",
    source_domain: str = "bio",
    verify_status: VerifyStatus | None = None,
) -> Hypothesis:
    return Hypothesis(
        id=hypothesis_id,
        description=description,
        source_domain=source_domain,
        source_structure="struct",
        analogy_explanation="analogy",
        observable="obs",
        final_score=final_score,
        verify_status=verify_status,
    )


# Run every test against both copies of the function
@pytest.fixture(params=[engine_build, saga_build], ids=["engine", "saga"])
def build_fn(request):
    return request.param


class TestBuildTopHypotheses:
    def test_returns_top_n(self, build_fn):
        hyps = [_hyp(f"h{i}", final_score=float(i)) for i in range(10)]
        result = build_fn(hyps, top_n=3)
        assert len(result) == 3
        # Highest scores first
        assert result[0]["score"] == 9.0
        assert result[1]["score"] == 8.0
        assert result[2]["score"] == 7.0

    def test_returns_all_when_fewer_than_n(self, build_fn):
        hyps = [_hyp("h1", final_score=5.0)]
        result = build_fn(hyps, top_n=5)
        assert len(result) == 1

    def test_empty_list(self, build_fn):
        assert build_fn([], top_n=5) == []

    def test_uses_composite_score_when_no_final_score(self, build_fn):
        h = _hyp("h1")
        # composite_score() with no scores returns 0.0
        result = build_fn([h])
        assert len(result) == 1
        assert result[0]["score"] == 0.0

    def test_final_score_preferred_over_composite(self, build_fn):
        h = _hyp("h1", final_score=8.5)
        result = build_fn([h])
        assert result[0]["score"] == 8.5

    def test_description_truncated_to_80(self, build_fn):
        long_desc = "A" * 200
        h = _hyp("h1", final_score=1.0, description=long_desc)
        result = build_fn([h])
        assert len(result[0]["description"]) <= 80

    def test_verify_status_serialized(self, build_fn):
        h = _hyp("h1", final_score=1.0, verify_status=VerifyStatus.PASSED)
        result = build_fn([h])
        assert result[0]["verify_status"] == "passed"

    def test_verify_status_none(self, build_fn):
        h = _hyp("h1", final_score=1.0, verify_status=None)
        result = build_fn([h])
        assert result[0]["verify_status"] == ""

    def test_output_fields(self, build_fn):
        h = _hyp("h1", final_score=7.0, source_domain="physics")
        result = build_fn([h])
        item = result[0]
        assert set(item.keys()) == {"id", "description", "score", "source_domain", "verify_status"}
        assert item["id"] == "h1"
        assert item["source_domain"] == "physics"

    def test_score_rounded_to_2_decimals(self, build_fn):
        h = _hyp("h1", final_score=7.123456)
        result = build_fn([h])
        assert result[0]["score"] == 7.12

    def test_default_top_n_is_5(self, build_fn):
        hyps = [_hyp(f"h{i}", final_score=float(i)) for i in range(10)]
        result = build_fn(hyps)
        assert len(result) == 5

    def test_sorting_mixed_final_and_composite(self, build_fn):
        """Hypothesis with final_score=8 should rank above one with no final_score."""
        h1 = _hyp("h1", final_score=8.0)
        h2 = _hyp("h2")  # composite_score() -> 0.0 (no scores set)
        result = build_fn([h2, h1], top_n=2)
        assert result[0]["id"] == "h1"
        assert result[1]["id"] == "h2"
