"""Tests for SourceDomainSelector."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from x_creative.core.plugin import TargetDomainPlugin
from x_creative.core.types import Domain, DomainStructure, ProblemFrame, TargetMapping


def _make_domain(domain_id: str, name: str) -> Domain:
    return Domain(
        id=domain_id,
        name=name,
        description=f"Description of {name}",
        structures=[
            DomainStructure(
                id=f"{domain_id}_s1", name=f"{name} structure",
                description="A structure", key_variables=["x", "y"], dynamics="linear",
            )
        ],
        target_mappings=[
            TargetMapping(structure=f"{domain_id}_s1", target="some target", observable="measure X")
        ],
    )


def _make_plugin_with_n_domains(n: int) -> TargetDomainPlugin:
    domains = [_make_domain(f"domain_{i}", f"Domain {i}").model_dump() for i in range(n)]
    return TargetDomainPlugin(id="test", name="Test", description="Test domain", source_domains=domains)


class TestSourceDomainSelector:
    @pytest.mark.asyncio
    async def test_select_returns_domains(self):
        from x_creative.answer.source_selector import SourceDomainSelector

        plugin = _make_plugin_with_n_domains(25)
        frame = ProblemFrame(description="test question")
        # Use a mocked router to avoid network calls in unit tests.
        mock_router = MagicMock()
        mock_router.complete = AsyncMock(side_effect=Exception("LLM unavailable in unit tests"))
        selector = SourceDomainSelector(router=mock_router)
        domains = await selector.select(frame, plugin, min_domains=5, max_domains=25)
        assert len(domains) >= 5
        assert len(domains) <= 25
        assert all(isinstance(d, Domain) for d in domains)

    @pytest.mark.asyncio
    async def test_select_respects_max(self):
        from x_creative.answer.source_selector import SourceDomainSelector

        plugin = _make_plugin_with_n_domains(40)
        frame = ProblemFrame(description="test")
        mock_router = MagicMock()
        mock_router.complete = AsyncMock(side_effect=Exception("LLM unavailable in unit tests"))
        selector = SourceDomainSelector(router=mock_router)
        domains = await selector.select(frame, plugin, min_domains=10, max_domains=20)
        assert len(domains) <= 20

    @pytest.mark.asyncio
    async def test_select_few_domains_returns_all(self):
        from x_creative.answer.source_selector import SourceDomainSelector

        plugin = _make_plugin_with_n_domains(5)
        frame = ProblemFrame(description="test")
        selector = SourceDomainSelector()
        domains = await selector.select(frame, plugin, min_domains=18, max_domains=30)
        assert len(domains) == 5

    @pytest.mark.asyncio
    async def test_select_empty_plugin(self):
        from x_creative.answer.source_selector import SourceDomainSelector

        plugin = TargetDomainPlugin(id="empty", name="Empty", description="No domains")
        frame = ProblemFrame(description="test")
        selector = SourceDomainSelector()
        domains = await selector.select(frame, plugin)
        assert domains == []


class TestSourceDomainSelectorFeasibility:
    """Tests for Stage B mapping feasibility filtering."""

    @pytest.mark.asyncio
    async def test_filter_removes_low_feasibility_domains(self) -> None:
        """Domains with low mapping feasibility should be filtered out."""
        from unittest.mock import AsyncMock, MagicMock
        from x_creative.answer.source_selector import SourceDomainSelector

        mock_router = MagicMock()
        mock_result = MagicMock()
        mock_result.content = '[{"domain_id": "domain_a", "feasibility": 0.8}, {"domain_id": "domain_b", "feasibility": 0.2}]'
        mock_router.complete = AsyncMock(return_value=mock_result)

        selector = SourceDomainSelector(
            router=mock_router,
            feasibility_threshold=0.4,
        )

        domains = [
            Domain(id="domain_a", name="A", description="Good domain", structures=[]),
            Domain(id="domain_b", name="B", description="Poor domain", structures=[]),
        ]
        frame = ProblemFrame(description="Test problem")

        result = await selector._filter_by_mapping_feasibility(frame, domains)

        assert len(result) == 1
        assert result[0].id == "domain_a"

    @pytest.mark.asyncio
    async def test_filter_keeps_all_on_failure(self) -> None:
        """If LLM fails, keep all domains (graceful degradation)."""
        from unittest.mock import AsyncMock, MagicMock
        from x_creative.answer.source_selector import SourceDomainSelector

        mock_router = MagicMock()
        mock_router.complete = AsyncMock(side_effect=Exception("LLM error"))

        selector = SourceDomainSelector(
            router=mock_router,
            feasibility_threshold=0.4,
        )

        domains = [
            Domain(id="a", name="A", description="d", structures=[]),
            Domain(id="b", name="B", description="d", structures=[]),
        ]
        frame = ProblemFrame(description="Test")

        result = await selector._filter_by_mapping_feasibility(frame, domains)
        assert len(result) == 2  # All kept on failure

    @pytest.mark.asyncio
    async def test_filter_never_returns_empty(self) -> None:
        """Even if all domains score below threshold, return all (never empty)."""
        from unittest.mock import AsyncMock, MagicMock
        from x_creative.answer.source_selector import SourceDomainSelector

        mock_router = MagicMock()
        mock_result = MagicMock()
        mock_result.content = '[{"domain_id": "a", "feasibility": 0.1}, {"domain_id": "b", "feasibility": 0.1}]'
        mock_router.complete = AsyncMock(return_value=mock_result)

        selector = SourceDomainSelector(
            router=mock_router,
            feasibility_threshold=0.4,
        )

        domains = [
            Domain(id="a", name="A", description="d", structures=[]),
            Domain(id="b", name="B", description="d", structures=[]),
        ]
        frame = ProblemFrame(description="Test")

        result = await selector._filter_by_mapping_feasibility(frame, domains)
        assert len(result) == 2  # All kept since filtering would leave empty

    @pytest.mark.asyncio
    async def test_public_filter_by_mapping_feasibility_uses_target_domains(self) -> None:
        from unittest.mock import AsyncMock
        from x_creative.answer.source_selector import SourceDomainSelector

        plugin = _make_plugin_with_n_domains(2)
        frame = ProblemFrame(description="Test")
        selector = SourceDomainSelector()

        mock_filter = AsyncMock(return_value=[plugin.get_domain_library().get("domain_0")])
        selector._filter_by_mapping_feasibility = mock_filter  # type: ignore[method-assign]

        result = await selector.filter_by_mapping_feasibility(frame, plugin)

        assert len(result) == 1
        assert result[0].id == "domain_0"
        assert mock_filter.await_count == 1
        passed_domains = mock_filter.await_args.args[1]
        assert len(passed_domains) == 2
