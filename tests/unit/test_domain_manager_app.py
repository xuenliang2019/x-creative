"""Tests for domain manager app."""

import pytest
from unittest.mock import MagicMock, patch


class TestDomainManagerApp:
    """Tests for DomainManagerApp."""

    def test_app_creation_with_default_target(self):
        """Test app creation with default target domain."""
        with patch(
            "x_creative.domain_manager.app.YAMLManager"
        ) as mock_yaml:
            mock_yaml.return_value.get_stats.return_value = {
                "target_domain_id": "open_source_development",
                "target_domain_name": "开源软件开发选题",
                "domain_count": 20,
                "structure_count": 83,
                "mapping_count": 83,
            }

            from x_creative.domain_manager.app import DomainManagerApp
            app = DomainManagerApp()

            assert app.target_domain_id == "open_source_development"
            assert app.target_domain_name == "开源软件开发选题"
            mock_yaml.assert_called_once_with("open_source_development")

    def test_app_creation_with_custom_target(self):
        """Test app creation with custom target domain."""
        with patch(
            "x_creative.domain_manager.app.YAMLManager"
        ) as mock_yaml:
            mock_yaml.return_value.get_stats.return_value = {
                "target_domain_id": "open_source_development",
                "target_domain_name": "开源软件开发选题",
                "domain_count": 17,
                "structure_count": 75,
                "mapping_count": 75,
            }

            from x_creative.domain_manager.app import DomainManagerApp
            app = DomainManagerApp(target_domain_id="open_source_development")

            assert app.target_domain_id == "open_source_development"
            assert app.target_domain_name == "开源软件开发选题"
            mock_yaml.assert_called_once_with("open_source_development")

    def test_refresh_stats(self):
        """Test that refresh_stats updates the stats."""
        with patch(
            "x_creative.domain_manager.app.YAMLManager"
        ) as mock_yaml:
            mock_yaml.return_value.get_stats.return_value = {
                "target_domain_id": "open_source_development",
                "target_domain_name": "开源软件开发选题",
                "domain_count": 20,
                "structure_count": 83,
                "mapping_count": 83,
            }

            from x_creative.domain_manager.app import DomainManagerApp
            app = DomainManagerApp()

            # Update stats
            mock_yaml.return_value.get_stats.return_value = {
                "target_domain_id": "open_source_development",
                "target_domain_name": "开源软件开发选题",
                "domain_count": 21,
                "structure_count": 87,
                "mapping_count": 87,
            }
            app.refresh_stats()
            assert mock_yaml.return_value.get_stats.call_count == 2
