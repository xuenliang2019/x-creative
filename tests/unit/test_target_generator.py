"""Tests for target generator service."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from x_creative.target_manager.services.target_generator import TargetGeneratorService


class TestTargetGeneratorService:
    """Tests for TargetGeneratorService."""

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
            "x_creative.target_manager.services.target_generator.ModelRouter",
            return_value=mock_router,
        ):
            return TargetGeneratorService()

    @pytest.mark.asyncio
    async def test_generate_constraints(self, service, mock_router):
        """Test generating constraints."""
        mock_router.complete.return_value = MagicMock(
            content='''```json
{
    "constraints": [
        {
            "name": "data_quality",
            "description": "数据质量必须有保证",
            "severity": "critical",
            "check_prompt": "检查数据来源"
        },
        {
            "name": "scalability",
            "description": "方案必须可扩展",
            "severity": "important",
            "check_prompt": null
        }
    ]
}
```'''
        )

        result = await service.generate_constraints("测试领域", "测试描述")
        assert len(result) == 2
        assert result[0].name == "data_quality"
        assert result[0].severity == "critical"
        assert result[1].name == "scalability"

    @pytest.mark.asyncio
    async def test_generate_evaluation_criteria(self, service, mock_router):
        """Test generating evaluation criteria."""
        mock_router.complete.return_value = MagicMock(
            content='''```json
{
    "evaluation_criteria": [
        "准确率 (Accuracy)",
        "召回率 (Recall)",
        "F1 Score"
    ]
}
```'''
        )

        result = await service.generate_evaluation_criteria("测试领域", "测试描述")
        assert len(result) == 3
        assert "准确率" in result[0]

    @pytest.mark.asyncio
    async def test_generate_anti_patterns(self, service, mock_router):
        """Test generating anti-patterns."""
        mock_router.complete.return_value = MagicMock(
            content='''```json
{
    "anti_patterns": [
        "忽略数据预处理",
        "过度拟合训练数据"
    ]
}
```'''
        )

        result = await service.generate_anti_patterns("测试领域", "测试描述")
        assert len(result) == 2
        assert "忽略" in result[0]

    @pytest.mark.asyncio
    async def test_generate_terminology(self, service, mock_router):
        """Test generating terminology."""
        mock_router.complete.return_value = MagicMock(
            content='''```json
{
    "terminology": {
        "过拟合": "模型在训练集上表现好但泛化能力差",
        "特征工程": "从原始数据中提取有用特征的过程"
    }
}
```'''
        )

        result = await service.generate_terminology("测试领域", "测试描述")
        assert len(result) == 2
        assert "过拟合" in result
        assert "特征工程" in result

    @pytest.mark.asyncio
    async def test_generate_stale_ideas(self, service, mock_router):
        """Test generating stale ideas."""
        mock_router.complete.return_value = MagicMock(
            content='''```json
{
    "stale_ideas": [
        "简单线性回归",
        "基于规则的系统"
    ]
}
```'''
        )

        result = await service.generate_stale_ideas("测试领域", "测试描述")
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_generate_full_metadata(self, service, mock_router):
        """Test parallel generation of all metadata."""
        # Mock all 5 calls (generate_full_metadata calls gather with 5 tasks)
        mock_router.complete.return_value = MagicMock(
            content='''```json
{
    "constraints": [{"name": "c1", "description": "d1", "severity": "critical", "check_prompt": null}],
    "evaluation_criteria": ["指标1"],
    "anti_patterns": ["反模式1"],
    "terminology": {"术语": "定义"},
    "stale_ideas": ["想法1"]
}
```'''
        )

        result = await service.generate_full_metadata("测试领域", "测试描述")
        assert "constraints" in result
        assert "evaluation_criteria" in result
        assert "anti_patterns" in result
        assert "terminology" in result
        assert "stale_ideas" in result

    @pytest.mark.asyncio
    async def test_rewrite_target_mappings(self, service, mock_router):
        """Test rewriting target mappings."""
        mock_router.complete.return_value = MagicMock(
            content='''```json
{
    "mappings": [
        {
            "structure": "entropy_increase",
            "target": "新目标概念",
            "observable": "新可观测指标"
        }
    ]
}
```'''
        )

        from x_creative.core.types import Domain, DomainStructure, TargetMapping

        domain = Domain(
            id="thermo",
            name="热力学",
            name_en="Thermodynamics",
            description="研究热力学",
            structures=[
                DomainStructure(
                    id="entropy_increase",
                    name="熵增",
                    description="熵只增不减",
                    key_variables=["entropy"],
                    dynamics="单向增加",
                )
            ],
            target_mappings=[
                TargetMapping(
                    structure="entropy_increase",
                    target="原目标",
                    observable="原指标",
                )
            ],
        )

        result = await service.rewrite_target_mappings(
            domain, "新领域", "新描述", "旧领域",
        )
        assert len(result) == 1
        assert result[0].target == "新目标概念"
        assert result[0].observable == "新可观测指标"

    @pytest.mark.asyncio
    async def test_batch_rewrite_source_domains(self, service, mock_router):
        """Test batch rewriting of source domains."""
        mock_router.complete.return_value = MagicMock(
            content='''```json
{
    "mappings": [
        {
            "structure": "s1",
            "target": "新目标",
            "observable": "新指标"
        }
    ]
}
```'''
        )

        from x_creative.core.types import Domain, DomainStructure, TargetMapping

        domains = [
            Domain(
                id=f"domain_{i}",
                name=f"领域{i}",
                name_en=f"Domain{i}",
                description=f"描述{i}",
                structures=[
                    DomainStructure(
                        id="s1", name="结构", description="描述",
                        key_variables=["v1"], dynamics="动态",
                    )
                ],
                target_mappings=[
                    TargetMapping(structure="s1", target="旧目标", observable="旧指标")
                ],
            )
            for i in range(3)
        ]

        progress_calls = []

        def on_progress(current, total):
            progress_calls.append((current, total))

        result = await service.batch_rewrite_source_domains(
            domains, "新领域", "新描述", "旧领域",
            progress_callback=on_progress,
        )

        assert len(result) == 3
        assert progress_calls == [(1, 3), (2, 3), (3, 3)]
        for d in result:
            assert d.target_mappings[0].target == "新目标"

    @pytest.mark.asyncio
    async def test_close(self, service, mock_router):
        """Test closing the router."""
        await service.close()
        mock_router.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_manager(self, mock_router):
        """Test async context manager."""
        with patch(
            "x_creative.target_manager.services.target_generator.ModelRouter",
            return_value=mock_router,
        ):
            async with TargetGeneratorService() as svc:
                assert svc is not None
            mock_router.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_constraints_parse_failure(self, service, mock_router):
        """Test graceful handling of parse failure."""
        mock_router.complete.return_value = MagicMock(
            content="This is not valid JSON at all"
        )

        result = await service.generate_constraints("测试", "测试")
        assert result == []
