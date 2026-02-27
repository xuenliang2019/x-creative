"""Tests for YAML manager service."""

import pytest
from pathlib import Path
from ruamel.yaml import YAML

from x_creative.domain_manager.services.yaml_manager import YAMLManager
from x_creative.core.types import Domain, DomainStructure, TargetMapping


@pytest.fixture
def temp_target_domain_file(tmp_path: Path) -> Path:
    """Create a temporary target domain YAML file."""
    yaml = YAML()
    yaml.preserve_quotes = True

    content = {
        "id": "test_target",
        "name": "测试目标域",
        "description": "测试用目标域",
        "constraints": ["约束1"],
        "evaluation_criteria": ["指标1"],
        "source_domains": [
            {
                "id": "test_domain",
                "name": "测试领域",
                "name_en": "Test Domain",
                "description": "测试用领域",
                "structures": [
                    {
                        "id": "test_structure",
                        "name": "测试结构",
                        "description": "测试结构描述",
                        "key_variables": ["var1", "var2"],
                        "dynamics": "测试动态",
                    }
                ],
                "target_mappings": [
                    {
                        "structure": "test_structure",
                        "target": "测试目标",
                        "observable": "测试可观测量",
                    }
                ],
            }
        ],
    }

    file_path = tmp_path / "test_target.yaml"
    with open(file_path, "w", encoding="utf-8") as f:
        yaml.dump(content, f)

    return file_path



class TestYAMLManager:
    """Tests for YAMLManager class."""

    def test_load_domains(self, temp_target_domain_file: Path) -> None:
        """Test loading source domains from target domain YAML."""
        manager = YAMLManager("test_target", path=temp_target_domain_file)
        domains = manager.load_domains()

        assert len(domains) == 1
        assert domains[0].id == "test_domain"
        assert domains[0].name == "测试领域"
        assert len(domains[0].structures) == 1
        assert len(domains[0].target_mappings) == 1

    def test_add_source_domain(self, temp_target_domain_file: Path) -> None:
        """Test adding a new source domain."""
        manager = YAMLManager("test_target", path=temp_target_domain_file)

        new_domain = Domain(
            id="new_domain",
            name="新领域",
            name_en="New Domain",
            description="新领域描述",
            structures=[
                DomainStructure(
                    id="new_structure",
                    name="新结构",
                    description="新结构描述",
                    key_variables=["a", "b"],
                    dynamics="新动态",
                )
            ],
            target_mappings=[
                TargetMapping(
                    structure="new_structure",
                    target="新目标",
                    observable="新可观测量",
                )
            ],
        )

        manager.add_source_domain(new_domain)
        manager.save()

        # Reload and verify
        manager2 = YAMLManager("test_target", path=temp_target_domain_file)
        domains = manager2.load_domains()
        assert len(domains) == 2
        assert domains[1].id == "new_domain"
        assert len(domains[1].target_mappings) == 1

    def test_add_structure_to_domain(self, temp_target_domain_file: Path) -> None:
        """Test adding a structure to an existing source domain."""
        manager = YAMLManager("test_target", path=temp_target_domain_file)
        manager.load_domains()

        new_structure = DomainStructure(
            id="added_structure",
            name="添加的结构",
            description="添加的结构描述",
            key_variables=["x", "y"],
            dynamics="添加的动态",
        )
        new_mapping = TargetMapping(
            structure="added_structure",
            target="添加的目标",
            observable="添加的可观测量",
        )

        manager.add_structure("test_domain", new_structure, new_mapping)
        manager.save()

        # Reload and verify
        manager2 = YAMLManager("test_target", path=temp_target_domain_file)
        domains = manager2.load_domains()
        domain = next(d for d in domains if d.id == "test_domain")
        assert len(domain.structures) == 2
        assert len(domain.target_mappings) == 2

    def test_backup_created(self, temp_target_domain_file: Path) -> None:
        """Test that backup is created before saving."""
        manager = YAMLManager("test_target", path=temp_target_domain_file)
        manager.load_domains()
        manager.save()

        backup_path = temp_target_domain_file.with_suffix(".yaml.bak")
        assert backup_path.exists()

    def test_get_stats(self, temp_target_domain_file: Path) -> None:
        """Test getting statistics."""
        manager = YAMLManager("test_target", path=temp_target_domain_file)
        stats = manager.get_stats()

        assert stats["target_domain_id"] == "test_target"
        assert stats["target_domain_name"] == "测试目标域"
        assert stats["domain_count"] == 1
        assert stats["structure_count"] == 1
        assert stats["mapping_count"] == 1

    def test_get_target_domain_info(self, temp_target_domain_file: Path) -> None:
        """Test getting target domain plugin configuration."""
        manager = YAMLManager("test_target", path=temp_target_domain_file)
        info = manager.get_target_domain_info()

        assert info["id"] == "test_target"
        assert info["name"] == "测试目标域"
        assert len(info["constraints"]) == 1
        assert len(info["evaluation_criteria"]) == 1

    def test_get_domain(self, temp_target_domain_file: Path) -> None:
        """Test getting a source domain by ID."""
        manager = YAMLManager("test_target", path=temp_target_domain_file)

        domain = manager.get_domain("test_domain")
        assert domain is not None
        assert domain.id == "test_domain"

        missing = manager.get_domain("nonexistent")
        assert missing is None

    def test_domains_property(self, temp_target_domain_file: Path) -> None:
        """Test domains property with lazy loading."""
        manager = YAMLManager("test_target", path=temp_target_domain_file)

        # Should lazy-load on first access
        domains = manager.domains
        assert len(domains) == 1

    def test_list_target_domains(self) -> None:
        """Test listing all available target domains."""
        targets = YAMLManager.list_target_domains()

        assert len(targets) >= 1
        ids = {t["id"] for t in targets}
        assert "open_source_development" in ids

        for t in targets:
            assert "id" in t
            assert "name" in t
            assert "domain_count" in t
            assert "structure_count" in t

    def test_resolve_path_existing(self) -> None:
        """Test resolving path for existing target domain."""
        path = YAMLManager._resolve_path("open_source_development")
        assert path.exists()
        assert path.name == "open_source_development.yaml"

    def test_resolve_path_nonexistent(self) -> None:
        """Test resolving path for nonexistent target domain raises error."""
        with pytest.raises(FileNotFoundError):
            YAMLManager._resolve_path("nonexistent_domain_xyz")

    def test_plugin_config_preserved_after_save(
        self, temp_target_domain_file: Path
    ) -> None:
        """Test that plugin config (constraints, etc.) is preserved after save."""
        manager = YAMLManager("test_target", path=temp_target_domain_file)

        new_domain = Domain(
            id="extra",
            name="额外",
            description="额外领域",
            structures=[],
            target_mappings=[],
        )
        manager.add_source_domain(new_domain)
        manager.save()

        # Reload and verify plugin config preserved
        manager2 = YAMLManager("test_target", path=temp_target_domain_file)
        info = manager2.get_target_domain_info()
        assert info["name"] == "测试目标域"
        assert len(info["constraints"]) == 1
        assert len(manager2.domains) == 2
