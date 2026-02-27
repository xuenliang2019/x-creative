"""Tests for target YAML manager."""

import pytest
from pathlib import Path

from x_creative.target_manager.services.target_yaml_manager import TargetYAMLManager


@pytest.fixture
def tmp_domains_dir(tmp_path):
    """Create a temporary domains directory."""
    domains_dir = tmp_path / "target_domains"
    domains_dir.mkdir()
    return domains_dir


@pytest.fixture
def manager(tmp_domains_dir):
    """Create a TargetYAMLManager with temp directory."""
    return TargetYAMLManager(domains_dir=tmp_domains_dir)


@pytest.fixture
def sample_yaml(tmp_domains_dir):
    """Create a sample target domain YAML file."""
    content = """\
id: test_domain
name: 测试领域
description: 这是一个测试领域

constraints:
- name: test_constraint
  description: 测试约束
  severity: critical
  check_prompt: 检查测试

evaluation_criteria:
- 指标一
- 指标二

anti_patterns:
- 反模式一

terminology:
  术语A: 定义A
  术语B: 定义B

stale_ideas:
- 陈旧想法一

source_domains:
- id: physics
  name: 物理学
  name_en: Physics
  description: 物理学基础
  structures:
  - id: force_law
    name: 力学定律
    description: 力等于质量乘以加速度
    key_variables: [force, mass, acceleration]
    dynamics: F=ma
  target_mappings:
  - structure: force_law
    target: 测试目标
    observable: 测试指标
"""
    yaml_file = tmp_domains_dir / "test_domain.yaml"
    yaml_file.write_text(content, encoding="utf-8")
    return yaml_file


class TestTargetYAMLManager:
    """Tests for TargetYAMLManager."""

    def test_list_target_domains_empty(self, manager):
        """Test listing with no domains."""
        result = manager.list_target_domains()
        assert result == []

    def test_list_target_domains(self, manager, sample_yaml):
        """Test listing domains returns correct data."""
        result = manager.list_target_domains()
        assert len(result) == 1
        assert result[0]["id"] == "test_domain"
        assert result[0]["name"] == "测试领域"
        assert result[0]["domain_count"] == 1
        assert result[0]["constraint_count"] == 1
        assert result[0]["structure_count"] == 1

    def test_load_target_domain(self, manager, sample_yaml):
        """Test loading a target domain as plugin."""
        plugin = manager.load_target_domain("test_domain")
        assert plugin.id == "test_domain"
        assert plugin.name == "测试领域"
        assert len(plugin.constraints) == 1
        assert plugin.constraints[0].name == "test_constraint"
        assert plugin.constraints[0].severity == "critical"
        assert len(plugin.evaluation_criteria) == 2
        assert len(plugin.terminology) == 2
        assert len(plugin.source_domains) == 1

    def test_load_target_domain_not_found(self, manager):
        """Test loading a nonexistent domain raises error."""
        with pytest.raises(FileNotFoundError):
            manager.load_target_domain("nonexistent")

    def test_get_stats(self, manager, sample_yaml):
        """Test getting stats for a target domain."""
        stats = manager.get_stats("test_domain")
        assert stats["target_id"] == "test_domain"
        assert stats["name"] == "测试领域"
        assert stats["domain_count"] == 1
        assert stats["structure_count"] == 1
        assert stats["constraint_count"] == 1
        assert stats["evaluation_criteria_count"] == 2
        assert stats["anti_pattern_count"] == 1
        assert stats["terminology_count"] == 2
        assert stats["stale_idea_count"] == 1

    def test_create_target_domain_minimal(self, manager, tmp_domains_dir):
        """Test creating a minimal target domain."""
        path = manager.create_target_domain(
            target_id="new_domain",
            name="新领域",
            description="新领域描述",
        )
        assert path.exists()
        assert path == tmp_domains_dir / "new_domain.yaml"

        # Verify we can load it back
        plugin = manager.load_target_domain("new_domain")
        assert plugin.id == "new_domain"
        assert plugin.name == "新领域"
        assert plugin.constraints == []

    def test_create_target_domain_full(self, manager):
        """Test creating a target domain with all fields."""
        path = manager.create_target_domain(
            target_id="full_domain",
            name="完整领域",
            description="完整描述",
            constraints=[{
                "name": "c1",
                "description": "约束1",
                "severity": "critical",
                "check_prompt": "检查",
            }],
            evaluation_criteria=["指标1", "指标2"],
            anti_patterns=["反模式1"],
            terminology={"术语": "定义"},
            stale_ideas=["陈旧想法"],
        )
        assert path.exists()

        plugin = manager.load_target_domain("full_domain")
        assert len(plugin.constraints) == 1
        assert len(plugin.evaluation_criteria) == 2
        assert len(plugin.anti_patterns) == 1
        assert plugin.terminology == {"术语": "定义"}
        assert len(plugin.stale_ideas) == 1

    def test_create_target_domain_already_exists(self, manager, sample_yaml):
        """Test creating a domain that already exists raises error."""
        with pytest.raises(FileExistsError):
            manager.create_target_domain(
                target_id="test_domain",
                name="重复",
                description="重复",
            )

    def test_update_section(self, manager, sample_yaml):
        """Test updating a section."""
        manager.update_section(
            "test_domain",
            "evaluation_criteria",
            ["新指标一", "新指标二", "新指标三"],
        )

        plugin = manager.load_target_domain("test_domain")
        assert len(plugin.evaluation_criteria) == 3
        assert "新指标一" in plugin.evaluation_criteria

    def test_update_section_creates_backup(self, manager, sample_yaml, tmp_domains_dir):
        """Test that updating creates a backup file."""
        manager.update_section(
            "test_domain",
            "stale_ideas",
            ["新陈旧想法"],
        )

        backup_path = tmp_domains_dir / "test_domain.yaml.bak"
        assert backup_path.exists()

    def test_update_section_invalid(self, manager, sample_yaml):
        """Test updating an invalid section raises error."""
        with pytest.raises(ValueError, match="Invalid section"):
            manager.update_section("test_domain", "invalid_field", [])

    def test_add_source_domains(self, manager, sample_yaml):
        """Test adding source domains."""
        from x_creative.core.types import Domain, DomainStructure, TargetMapping

        domain = Domain(
            id="chemistry",
            name="化学",
            name_en="Chemistry",
            description="化学基础",
            structures=[
                DomainStructure(
                    id="reaction_rate",
                    name="反应速率",
                    description="化学反应速率",
                    key_variables=["concentration", "temperature"],
                    dynamics="浓度越高速率越快",
                )
            ],
            target_mappings=[
                TargetMapping(
                    structure="reaction_rate",
                    target="测试目标",
                    observable="测试指标",
                )
            ],
        )

        manager.add_source_domains("test_domain", [domain])

        plugin = manager.load_target_domain("test_domain")
        assert len(plugin.source_domains) == 2
        assert plugin.source_domains[1]["id"] == "chemistry"

    def test_validate_target_id_valid(self, manager):
        """Test valid target IDs."""
        is_valid, msg = manager.validate_target_id("new_domain")
        assert is_valid is True

        is_valid, msg = manager.validate_target_id("a123_test")
        assert is_valid is True

    def test_validate_target_id_empty(self, manager):
        """Test empty ID."""
        is_valid, msg = manager.validate_target_id("")
        assert is_valid is False

    def test_validate_target_id_invalid_format(self, manager):
        """Test invalid ID formats."""
        is_valid, _ = manager.validate_target_id("123abc")
        assert is_valid is False

        is_valid, _ = manager.validate_target_id("CamelCase")
        assert is_valid is False

        is_valid, _ = manager.validate_target_id("with-dash")
        assert is_valid is False

    def test_validate_target_id_too_long(self, manager):
        """Test ID that's too long."""
        is_valid, _ = manager.validate_target_id("a" * 51)
        assert is_valid is False

    def test_validate_target_id_already_exists(self, manager, sample_yaml):
        """Test ID that already exists."""
        is_valid, msg = manager.validate_target_id("test_domain")
        assert is_valid is False
        assert "已存在" in msg
