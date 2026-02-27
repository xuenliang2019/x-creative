"""Tests for TargetDomainResolver."""

from unittest.mock import AsyncMock, patch

import pytest


class TestTargetDomainResolver:
    @pytest.mark.asyncio
    async def test_explicit_override(self):
        from x_creative.answer.target_resolver import TargetDomainResolver
        from x_creative.core.types import ProblemFrame

        resolver = TargetDomainResolver()
        frame = ProblemFrame(description="test")
        plugin = await resolver.resolve(frame, target_override="open_source_development")
        assert plugin is not None
        assert plugin.id == "open_source_development"

    @pytest.mark.asyncio
    async def test_exact_match_high_confidence(self):
        from x_creative.answer.target_resolver import TargetDomainResolver
        from x_creative.core.types import ProblemFrame

        frame = ProblemFrame(
            description="test",
            domain_hint={"domain_id": "open_source_development", "confidence": 0.85},
        )
        resolver = TargetDomainResolver()
        plugin = await resolver.resolve(frame, target_override="auto")
        assert plugin is not None
        assert plugin.id == "open_source_development"

    @pytest.mark.asyncio
    async def test_nonexistent_override_raises(self):
        from x_creative.answer.target_resolver import TargetDomainResolver
        from x_creative.core.types import ProblemFrame

        resolver = TargetDomainResolver()
        frame = ProblemFrame(description="test")
        with pytest.raises(ValueError, match="not found"):
            await resolver.resolve(frame, target_override="nonexistent_xyz")

    @pytest.mark.asyncio
    async def test_low_confidence_falls_to_ephemeral(self):
        from x_creative.answer.target_resolver import TargetDomainResolver
        from x_creative.core.types import ProblemFrame

        frame = ProblemFrame(
            description="How to grow tomatoes",
            domain_hint={"domain_id": "agriculture", "confidence": 0.4},
        )
        resolver = TargetDomainResolver()

        # Mock semantic match to return None (no match) and ephemeral to return a plugin
        with patch.object(resolver, "_semantic_match", new_callable=AsyncMock, return_value=None):
            with patch.object(resolver, "_generate_ephemeral", new_callable=AsyncMock) as mock_gen:
                from x_creative.core.plugin import TargetDomainPlugin

                mock_gen.return_value = TargetDomainPlugin(
                    id="ephemeral_agriculture",
                    name="Agriculture",
                    description="Farming",
                )
                plugin = await resolver.resolve(frame, target_override="auto")

        assert plugin is not None
        assert plugin.id == "ephemeral_agriculture"
