"""Tests for TargetDomainPlugin."""

import pytest
from pydantic import ValidationError


class TestDomainConstraint:
    """Tests for DomainConstraint."""

    def test_create_minimal(self):
        from x_creative.core.plugin import DomainConstraint

        constraint = DomainConstraint(
            name="low_signal_to_noise",
            description="跨领域类比中有效信号稀疏，信噪比通常 < 0.1",
            severity="critical",
        )

        assert constraint.name == "low_signal_to_noise"
        assert constraint.severity == "critical"
        assert constraint.check_prompt is None

    def test_create_with_check_prompt(self):
        from x_creative.core.plugin import DomainConstraint

        constraint = DomainConstraint(
            name="no_lookahead",
            description="不能使用未来信息",
            severity="critical",
            check_prompt="检查假说中是否存在使用未来数据的迹象",
        )

        assert constraint.check_prompt is not None

    def test_invalid_severity(self):
        from x_creative.core.plugin import DomainConstraint

        with pytest.raises(ValidationError):
            DomainConstraint(
                name="test",
                description="test",
                severity="invalid",  # Must be critical/important/advisory
            )


class TestValidatorConfig:
    """Tests for ValidatorConfig."""

    def test_create_valid(self):
        from x_creative.core.plugin import ValidatorConfig

        config = ValidatorConfig(
            name="novelty_checker",
            prompt_template="Check if {idea} is novel",
        )

        assert config.name == "novelty_checker"
        assert config.prompt_template == "Check if {idea} is novel"
        assert config.model is None
        assert config.threshold == 5.0  # default value

    def test_default_threshold(self):
        from x_creative.core.plugin import ValidatorConfig

        config = ValidatorConfig(
            name="test",
            prompt_template="test",
        )

        assert config.threshold == 5.0

    def test_threshold_validation_bounds(self):
        from x_creative.core.plugin import ValidatorConfig

        # Valid: within bounds
        config = ValidatorConfig(
            name="test",
            prompt_template="test",
            threshold=0.0,
        )
        assert config.threshold == 0.0

        config = ValidatorConfig(
            name="test",
            prompt_template="test",
            threshold=10.0,
        )
        assert config.threshold == 10.0

        # Invalid: below lower bound
        with pytest.raises(ValidationError):
            ValidatorConfig(
                name="test",
                prompt_template="test",
                threshold=-0.1,
            )

        # Invalid: above upper bound
        with pytest.raises(ValidationError):
            ValidatorConfig(
                name="test",
                prompt_template="test",
                threshold=10.1,
            )


class TestTargetDomainPlugin:
    """Tests for TargetDomainPlugin."""

    def test_create_minimal(self):
        from x_creative.core.plugin import TargetDomainPlugin

        plugin = TargetDomainPlugin(
            id="drug_discovery",
            name="药物发现",
            description="研究新药靶点和分子设计",
        )

        assert plugin.id == "drug_discovery"
        assert plugin.constraints == []
        assert plugin.evaluation_criteria == []

    def test_create_full(self):
        from x_creative.core.plugin import DomainConstraint, TargetDomainPlugin

        plugin = TargetDomainPlugin(
            id="open_source_development",
            name="开源软件开发选题",
            description="开源项目创新研究",
            constraints=[
                DomainConstraint(
                    name="adoption_barrier",
                    description="新工具需克服用户惯性",
                    severity="critical",
                ),
            ],
            evaluation_criteria=["易用性", "社区活跃度"],
            anti_patterns=["重复造轮子", "过度工程化"],
            terminology={"viral": "病毒式传播"},
            stale_ideas=["又一个TODO应用", "又一个Markdown编辑器"],
        )

        assert len(plugin.constraints) == 1
        assert "易用性" in plugin.evaluation_criteria

    def test_get_constraint_by_name(self):
        from x_creative.core.plugin import DomainConstraint, TargetDomainPlugin

        plugin = TargetDomainPlugin(
            id="test",
            name="Test",
            description="Test domain",
            constraints=[
                DomainConstraint(name="c1", description="d1", severity="critical"),
                DomainConstraint(name="c2", description="d2", severity="advisory"),
            ],
        )

        c = plugin.get_constraint("c1")
        assert c is not None
        assert c.severity == "critical"

        assert plugin.get_constraint("nonexistent") is None

    def test_get_critical_constraints(self):
        from x_creative.core.plugin import DomainConstraint, TargetDomainPlugin

        plugin = TargetDomainPlugin(
            id="test",
            name="Test",
            description="Test domain",
            constraints=[
                DomainConstraint(name="c1", description="d1", severity="critical"),
                DomainConstraint(name="c2", description="d2", severity="advisory"),
                DomainConstraint(name="c3", description="d3", severity="important"),
                DomainConstraint(name="c4", description="d4", severity="critical"),
            ],
        )

        critical = plugin.get_critical_constraints()

        assert len(critical) == 2
        assert all(c.severity == "critical" for c in critical)
        critical_names = [c.name for c in critical]
        assert "c1" in critical_names
        assert "c4" in critical_names


class TestTargetDomainLoader:
    """Tests for loading target domain plugins."""

    def test_load_builtin_open_source_development(self):
        from x_creative.core.plugin import load_target_domain

        plugin = load_target_domain("open_source_development")

        assert plugin is not None
        assert plugin.id == "open_source_development"
        assert len(plugin.constraints) > 0

    def test_load_nonexistent_domain(self):
        from x_creative.core.plugin import load_target_domain

        plugin = load_target_domain("nonexistent_domain_xyz")
        assert plugin is None

    def test_list_available_domains(self):
        from x_creative.core.plugin import list_target_domains

        domains = list_target_domains()
        assert "open_source_development" in domains
