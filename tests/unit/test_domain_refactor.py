"""Tests for Phase 0 domain architecture refactoring.

Tests the new target_mappings, source_domains, and from_target_domain() features.
"""

import pytest
import yaml
from pathlib import Path
from unittest.mock import patch

from x_creative.core.types import Domain, DomainStructure, TargetMapping


class TestTargetDomainPluginSourceDomains:
    """Tests for TargetDomainPlugin.source_domains field."""

    def test_plugin_without_source_domains(self) -> None:
        """Plugin without source_domains should have empty list."""
        from x_creative.core.plugin import TargetDomainPlugin

        plugin = TargetDomainPlugin(
            id="test",
            name="Test",
            description="test domain",
        )

        assert plugin.source_domains == []

    def test_plugin_with_source_domains(self) -> None:
        """Plugin with embedded source domains."""
        from x_creative.core.plugin import TargetDomainPlugin

        plugin = TargetDomainPlugin(
            id="open_source_development",
            name="开源软件开发选题",
            description="开源项目创新研究",
            source_domains=[
                {
                    "id": "thermodynamics",
                    "name": "热力学系统",
                    "description": "能量转换和熵增",
                    "structures": [
                        {
                            "id": "entropy_increase",
                            "name": "熵增定律",
                            "description": "孤立系统趋向最大熵状态",
                            "key_variables": ["entropy", "energy"],
                            "dynamics": "单向增加，不可逆",
                        }
                    ],
                    "target_mappings": [
                        {
                            "structure": "entropy_increase",
                            "target": "A股收益率分布熵",
                            "observable": "rolling_entropy",
                        }
                    ],
                }
            ],
        )

        assert len(plugin.source_domains) == 1
        assert plugin.source_domains[0]["id"] == "thermodynamics"

    def test_get_domain_library(self) -> None:
        """get_domain_library() should create DomainLibrary from source_domains."""
        from x_creative.core.plugin import TargetDomainPlugin
        from x_creative.core.domain_loader import DomainLibrary

        plugin = TargetDomainPlugin(
            id="open_source_development",
            name="开源软件开发选题",
            description="开源项目创新研究",
            source_domains=[
                {
                    "id": "thermodynamics",
                    "name": "热力学",
                    "description": "热力学系统",
                    "structures": [
                        {
                            "id": "entropy",
                            "name": "熵",
                            "description": "熵增",
                            "key_variables": ["S"],
                            "dynamics": "增加",
                        }
                    ],
                    "target_mappings": [
                        {
                            "structure": "entropy",
                            "target": "市场混乱度",
                            "observable": "收益率分布熵",
                        }
                    ],
                },
                {
                    "id": "control_theory",
                    "name": "控制论",
                    "description": "反馈系统",
                    "structures": [],
                    "target_mappings": [],
                },
            ],
        )

        library = plugin.get_domain_library()
        assert isinstance(library, DomainLibrary)
        assert len(library) == 2
        assert library.get("thermodynamics") is not None
        assert library.get("control_theory") is not None

        # Verify mappings are correctly loaded
        thermo = library.get("thermodynamics")
        assert len(thermo.target_mappings) == 1
        assert thermo.target_mappings[0].target == "市场混乱度"

    def test_get_domain_library_empty(self) -> None:
        """get_domain_library() with no source domains returns empty library."""
        from x_creative.core.plugin import TargetDomainPlugin

        plugin = TargetDomainPlugin(
            id="test",
            name="Test",
            description="test",
        )

        library = plugin.get_domain_library()
        assert len(library) == 0

    def test_plugin_loads_from_yaml_with_source_domains(self, tmp_path: Path) -> None:
        """Test loading a YAML file that includes source_domains."""
        from x_creative.core.plugin import _load_yaml_plugin

        yaml_content = {
            "id": "test_domain",
            "name": "Test Domain",
            "description": "A test target domain",
            "constraints": [],
            "evaluation_criteria": ["metric1"],
            "source_domains": [
                {
                    "id": "physics",
                    "name": "物理学",
                    "description": "物理系统",
                    "structures": [
                        {
                            "id": "wave",
                            "name": "波动",
                            "description": "波动现象",
                            "key_variables": ["amplitude", "frequency"],
                            "dynamics": "振荡",
                        }
                    ],
                    "target_mappings": [
                        {
                            "structure": "wave",
                            "target": "价格波动周期",
                            "observable": "FFT频谱分析",
                        }
                    ],
                }
            ],
        }

        yaml_file = tmp_path / "test_domain.yaml"
        with open(yaml_file, "w", encoding="utf-8") as f:
            yaml.dump(yaml_content, f, allow_unicode=True)

        plugin = _load_yaml_plugin(yaml_file)
        assert plugin is not None
        assert plugin.id == "test_domain"
        assert len(plugin.source_domains) == 1
        assert plugin.source_domains[0]["id"] == "physics"

        # Verify domain library creation works
        library = plugin.get_domain_library()
        assert len(library) == 1
        physics = library.get("physics")
        assert physics is not None
        assert len(physics.structures) == 1
        assert len(physics.target_mappings) == 1

    def test_plugin_loads_from_yaml_without_source_domains(self, tmp_path: Path) -> None:
        """Test loading existing YAML format (no source_domains) still works."""
        from x_creative.core.plugin import _load_yaml_plugin

        yaml_content = {
            "id": "legacy_domain",
            "name": "Legacy",
            "description": "Old format",
            "constraints": [],
        }

        yaml_file = tmp_path / "legacy_domain.yaml"
        with open(yaml_file, "w", encoding="utf-8") as f:
            yaml.dump(yaml_content, f, allow_unicode=True)

        plugin = _load_yaml_plugin(yaml_file)
        assert plugin is not None
        assert plugin.source_domains == []

class TestDomainLibraryFromTargetDomain:
    """Tests for DomainLibrary.from_target_domain()."""

    def test_from_target_domain_with_embedded_domains(self, tmp_path: Path) -> None:
        """from_target_domain() should load from embedded source domains."""
        from x_creative.core.domain_loader import DomainLibrary
        from x_creative.core.plugin import TargetDomainPlugin, _load_yaml_plugin

        yaml_content = {
            "id": "test_td",
            "name": "Test TD",
            "description": "test",
            "source_domains": [
                {
                    "id": "domain_a",
                    "name": "Domain A",
                    "description": "desc",
                    "structures": [
                        {
                            "id": "s1",
                            "name": "S1",
                            "description": "d",
                            "key_variables": ["x"],
                            "dynamics": "dyn",
                        }
                    ],
                    "target_mappings": [
                        {
                            "structure": "s1",
                            "target": "t1",
                            "observable": "o1",
                        }
                    ],
                }
            ],
        }

        yaml_file = tmp_path / "test_td.yaml"
        with open(yaml_file, "w", encoding="utf-8") as f:
            yaml.dump(yaml_content, f, allow_unicode=True)

        plugin = _load_yaml_plugin(yaml_file)

        with patch("x_creative.core.plugin.load_target_domain", return_value=plugin):
            library = DomainLibrary.from_target_domain("test_td")

        assert len(library) == 1
        assert library.get("domain_a") is not None
        assert len(library.get("domain_a").target_mappings) == 1

    def test_from_target_domain_no_source_domains_raises(self) -> None:
        """from_target_domain() should raise ValueError when plugin has no source_domains."""
        from x_creative.core.domain_loader import DomainLibrary
        from x_creative.core.plugin import TargetDomainPlugin

        # Plugin with no source_domains
        plugin = TargetDomainPlugin(
            id="open_source_development",
            name="开源软件开发选题",
            description="test",
        )

        with patch("x_creative.core.plugin.load_target_domain", return_value=plugin):
            with pytest.raises(ValueError, match="no embedded source domains"):
                DomainLibrary.from_target_domain("open_source_development")

    def test_from_target_domain_unknown_id_raises(self) -> None:
        """from_target_domain() with unknown ID should raise ValueError."""
        from x_creative.core.domain_loader import DomainLibrary

        with patch("x_creative.core.plugin.load_target_domain", return_value=None):
            with pytest.raises(ValueError, match="not found"):
                DomainLibrary.from_target_domain("nonexistent_domain")


class TestDomainModelDumpCompat:
    """Tests for Domain model_dump() behavior with target_mappings."""

    def test_model_dump_uses_target_mappings(self) -> None:
        """model_dump() should output target_mappings."""
        domain = Domain(
            id="test",
            name="Test",
            description="test",
            structures=[],
            target_mappings=[
                TargetMapping(structure="s1", target="t1", observable="o1"),
            ],
        )

        dumped = domain.model_dump()
        assert "target_mappings" in dumped
        assert len(dumped["target_mappings"]) == 1

    def test_model_dump_round_trip(self) -> None:
        """Domain should survive model_dump → Domain(**data) round trip."""
        original = Domain(
            id="thermo",
            name="热力学",
            description="热力学系统",
            structures=[
                DomainStructure(
                    id="entropy",
                    name="熵",
                    description="desc",
                    key_variables=["S"],
                    dynamics="增加",
                )
            ],
            target_mappings=[
                TargetMapping(
                    structure="entropy",
                    target="混乱度",
                    observable="分布熵",
                )
            ],
        )

        dumped = original.model_dump()
        restored = Domain(**dumped)

        assert restored.id == original.id
        assert len(restored.target_mappings) == 1
        assert restored.target_mappings[0].target == "混乱度"


class TestBISOTargetDomainLoading:
    """Tests for BISO module target-domain-aware domain loading."""

    def test_get_domains_with_open_source_development(self) -> None:
        """_get_domains('open_source_development') should load embedded domains."""
        from x_creative.creativity.biso import BISOModule
        from x_creative.core.plugin import load_target_domain

        load_target_domain.cache_clear()
        biso = BISOModule()
        library = biso._get_domains("open_source_development")

        assert len(library) > 0

    def test_get_domains_unknown_target_raises(self) -> None:
        """_get_domains with unknown target_domain should raise ValueError."""
        from x_creative.creativity.biso import BISOModule
        from x_creative.core.plugin import load_target_domain

        load_target_domain.cache_clear()
        biso = BISOModule()

        with pytest.raises(ValueError):
            biso._get_domains("nonexistent_domain")

    def test_get_domains_none_target_raises(self) -> None:
        """_get_domains(None) should raise ValueError."""
        from x_creative.creativity.biso import BISOModule

        biso = BISOModule()

        with pytest.raises(ValueError, match="target_domain is required"):
            biso._get_domains(None)

    def test_get_domains_caches_results(self) -> None:
        """_get_domains should cache and return same instance on second call."""
        from x_creative.creativity.biso import BISOModule
        from x_creative.core.plugin import load_target_domain

        load_target_domain.cache_clear()
        biso = BISOModule()
        lib1 = biso._get_domains("open_source_development")
        lib2 = biso._get_domains("open_source_development")

        assert lib1 is lib2  # same object (cached)

    def test_get_domains_explicit_override(self) -> None:
        """Explicit domain_library should bypass target-domain loading."""
        from x_creative.creativity.biso import BISOModule
        from x_creative.core.domain_loader import DomainLibrary

        explicit = DomainLibrary([
            Domain(
                id="custom",
                name="自定义",
                description="custom domain",
                structures=[],
                target_mappings=[],
            )
        ])

        biso = BISOModule(domain_library=explicit)
        library = biso._get_domains("open_source_development")

        # Should return explicit library, not the plugin's domains
        assert len(library) == 1
        assert library.get("custom") is not None
