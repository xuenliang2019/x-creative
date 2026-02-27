"""Tests for domain generator service."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from x_creative.domain_manager.services.generator import (
    DomainGeneratorService,
    DomainRecommendation,
)
from x_creative.domain_manager.services.search import SearchResult


class TestDomainGeneratorService:
    """Tests for DomainGeneratorService."""

    @pytest.fixture
    def mock_router(self):
        """Create mock model router."""
        router = MagicMock()
        router.complete = AsyncMock()
        router.close = AsyncMock()
        return router

    @pytest.fixture
    def service(self, mock_router):
        """Create service with mock router."""
        with patch(
            "x_creative.domain_manager.services.generator.ModelRouter",
            return_value=mock_router,
        ):
            return DomainGeneratorService()

    @pytest.mark.asyncio
    async def test_recommend_domains(self, service, mock_router):
        """Test domain recommendation."""
        mock_router.complete.return_value = MagicMock(
            content='''```json
{
    "recommendations": [
        {
            "domain_name": "临界现象",
            "domain_name_en": "Critical Phenomena",
            "relevance_reason": "相变理论可用于预测用户增长拐点",
            "potential_structures": ["相变点", "临界指数"]
        }
    ]
}
```'''
        )

        search_results = [
            SearchResult(
                title="Critical Phenomena in Physics",
                url="https://example.com",
                description="Study of phase transitions",
            )
        ]

        recommendations = await service.recommend_domains(
            "寻找预测用户增长拐点的新视角",
            search_results,
        )

        assert len(recommendations) >= 1
        assert recommendations[0].domain_name == "临界现象"

    @pytest.mark.asyncio
    async def test_extract_structures(self, service, mock_router):
        """Test structure extraction."""
        mock_router.complete.return_value = MagicMock(
            content='''```json
{
    "structures": [
        {
            "id": "phase_transition",
            "name": "相变",
            "description": "系统在临界点发生状态突变",
            "key_variables": ["order_parameter", "temperature"],
            "dynamics": "临界点附近涨落增大"
        }
    ]
}
```'''
        )

        search_results = [
            SearchResult(
                title="Phase Transitions",
                url="https://example.com",
                description="Study of state changes",
            )
        ]

        structures = await service.extract_structures("热力学", search_results)
        assert len(structures) >= 1
        assert structures[0].id == "phase_transition"

    @pytest.mark.asyncio
    async def test_generate_mapping(self, service, mock_router):
        """Test target mapping generation."""
        mock_router.complete.return_value = MagicMock(
            content='''```json
{
    "target": "用户增长状态转换信号",
    "observable": "活跃度突变、留存率结构突变"
}
```'''
        )

        from x_creative.core.types import DomainStructure

        structure = DomainStructure(
            id="phase_transition",
            name="相变",
            description="系统在临界点发生状态突变",
            key_variables=["order_parameter", "temperature"],
            dynamics="临界点附近涨落增大",
        )

        search_results = [
            SearchResult(
                title="Phase Transitions in Complex Systems",
                url="https://example.com",
                description="Applying physics to complex systems",
            )
        ]

        mapping = await service.generate_mapping(structure, search_results)
        assert mapping.target == "用户增长状态转换信号"
        assert "活跃度" in mapping.observable

    @pytest.mark.asyncio
    async def test_close(self, service, mock_router):
        """Test closing the router."""
        await service.close()
        mock_router.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_domain_similarity_finds_similar(self, service, mock_router):
        """Test that similar domains are detected."""
        mock_router.complete.return_value = MagicMock(
            content='''```json
{
    "is_similar": true,
    "similar_domain_id": "game_theory",
    "reason": "博弈论和Game Theory是同一学科"
}
```'''
        )

        rec = DomainRecommendation(
            domain_name="博弈论",
            domain_name_en="Game Theory",
            relevance_reason="策略互动分析",
            potential_structures=["纳什均衡"],
        )

        existing_domains = [
            ("game_theory", "博弈论系统", "Game Theory"),
            ("thermodynamics", "热力学", "Thermodynamics"),
        ]

        result = await service.check_domain_similarity(rec, existing_domains)

        assert result.is_extension is True
        assert result.existing_domain_id == "game_theory"
        assert result.existing_domain_name == "博弈论系统"

    @pytest.mark.asyncio
    async def test_check_domain_similarity_no_similar(self, service, mock_router):
        """Test that dissimilar domains are not marked as extensions."""
        mock_router.complete.return_value = MagicMock(
            content='''```json
{
    "is_similar": false,
    "similar_domain_id": null,
    "reason": "量子力学与现有领域无关"
}
```'''
        )

        rec = DomainRecommendation(
            domain_name="量子力学",
            domain_name_en="Quantum Mechanics",
            relevance_reason="量子效应",
            potential_structures=["叠加态"],
        )

        existing_domains = [
            ("game_theory", "博弈论", "Game Theory"),
        ]

        result = await service.check_domain_similarity(rec, existing_domains)

        assert result.is_extension is False
        assert result.existing_domain_id is None

    @pytest.mark.asyncio
    async def test_filter_duplicate_structures(self, service, mock_router):
        """Test filtering out duplicate structures."""
        mock_router.complete.return_value = MagicMock(
            content='''```json
{
    "duplicate_indices": [1],
    "reasons": {"1": "与现有结构 entropy_increase 语义相同"}
}
```'''
        )

        from x_creative.core.types import DomainStructure

        new_structures = [
            DomainStructure(
                id="heat_flow",
                name="热传导",
                description="热量从高温流向低温",
                key_variables=["temperature_gradient"],
                dynamics="热量自发流动",
            ),
            DomainStructure(
                id="entropy_law",
                name="熵增定律",
                description="孤立系统熵只增不减",
                key_variables=["entropy"],
                dynamics="熵增加",
            ),
        ]

        existing_structures = [
            DomainStructure(
                id="entropy_increase",
                name="熵增原理",
                description="系统总熵趋于增大",
                key_variables=["entropy", "disorder"],
                dynamics="不可逆过程熵增",
            ),
        ]

        result = await service.filter_duplicate_structures(
            new_structures, existing_structures
        )

        # Should filter out the second structure (index 1)
        assert len(result) == 1
        assert result[0].id == "heat_flow"
