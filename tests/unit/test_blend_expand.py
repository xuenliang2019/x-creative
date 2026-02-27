"""Tests for blend_expand operator."""

import pytest
from unittest.mock import AsyncMock

from x_creative.core.types import Hypothesis
from x_creative.creativity.blend import blend_expand


def _hyp(hid: str, domain: str = "thermo") -> Hypothesis:
    return Hypothesis(
        id=hid, description=f"hyp {hid}", source_domain=domain,
        source_structure="s", analogy_explanation="a", observable="obs",
    )


class TestBlendExpand:
    @pytest.mark.asyncio
    async def test_returns_hypothesis_with_blend_network(self) -> None:
        mock_router = AsyncMock()
        mock_router.complete.return_value = AsyncMock(
            content='[{"generic_space": "shared structure", "blend_description": "merged concept", "cross_space_mappings": [{"source_element": "entropy", "blend_element": "disorder", "projection_type": "compression"}], "emergent_structures": [{"description": "new pattern", "emergence_type": "completion", "observable_link": "obs_x"}], "description": "blended hypothesis", "observable": "blend_obs"}]'
        )

        results = await blend_expand(
            hypothesis_a=_hyp("a", "thermo"),
            hypothesis_b=_hyp("b", "info"),
            router=mock_router,
        )

        assert len(results) >= 1
        assert results[0].blend_network is not None
        assert results[0].observable != ""
        assert results[0].expansion_type == "blend"

    @pytest.mark.asyncio
    async def test_returns_empty_on_parse_error(self) -> None:
        mock_router = AsyncMock()
        mock_router.complete.return_value = AsyncMock(content="not json")

        results = await blend_expand(
            hypothesis_a=_hyp("a"),
            hypothesis_b=_hyp("b"),
            router=mock_router,
        )
        assert results == []

    @pytest.mark.asyncio
    async def test_skips_blends_missing_observable(self) -> None:
        mock_router = AsyncMock()
        mock_router.complete.return_value = AsyncMock(
            content='[{"generic_space": "shared", "blend_description": "blend", '
            '"cross_space_mappings": [], "emergent_structures": [], '
            '"description": "blended hyp", "observable": "   "}]'
        )

        results = await blend_expand(
            hypothesis_a=_hyp("a"),
            hypothesis_b=_hyp("b"),
            router=mock_router,
        )

        assert results == []


class TestBlendExpandRouting:
    @pytest.mark.asyncio
    async def test_blend_expand_uses_task_routing(self) -> None:
        """blend_expand should call router.complete with task='blend_expansion'."""
        mock_router = AsyncMock()
        mock_router.complete.return_value = AsyncMock(
            content='[{"generic_space": "shared", "blend_description": "blend", '
            '"cross_space_mappings": [], "emergent_structures": [], '
            '"description": "blended hyp", "observable": "x_blend"}]'
        )

        ha = Hypothesis(
            id="a", description="hyp a", source_domain="thermo",
            source_structure="s", analogy_explanation="a", observable="x",
        )
        hb = Hypothesis(
            id="b", description="hyp b", source_domain="info_theory",
            source_structure="s", analogy_explanation="a", observable="y",
        )

        await blend_expand(ha, hb, mock_router)

        mock_router.complete.assert_called_once()
        _, kwargs = mock_router.complete.call_args
        assert kwargs["task"] == "blend_expansion"
        assert isinstance(kwargs["messages"], list)
        assert kwargs["messages"][0]["role"] == "user"
