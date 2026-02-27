"""Tests for target manager app."""

from unittest.mock import MagicMock, patch


class TestTargetManagerApp:
    """Tests for TargetManagerApp."""

    def test_app_creation(self):
        """Test app creation."""
        with patch(
            "x_creative.target_manager.app.TargetYAMLManager"
        ) as mock_yaml:
            mock_yaml.return_value.list_target_domains.return_value = [
                {
                    "id": "open_source_development",
                    "name": "开源软件开发选题",
                    "description": "开源软件开发研究",
                    "domain_count": 20,
                    "constraint_count": 5,
                }
            ]

            from x_creative.target_manager.app import TargetManagerApp
            app = TargetManagerApp()

            assert app.yaml_manager is not None
            mock_yaml.assert_called_once()

    def test_app_yaml_manager_property(self):
        """Test yaml_manager property."""
        with patch(
            "x_creative.target_manager.app.TargetYAMLManager"
        ) as mock_yaml:
            from x_creative.target_manager.app import TargetManagerApp
            app = TargetManagerApp()

            assert app.yaml_manager == mock_yaml.return_value

    def test_app_refresh_targets(self):
        """Test refresh_targets returns fresh data."""
        with patch(
            "x_creative.target_manager.app.TargetYAMLManager"
        ) as mock_yaml:
            mock_yaml.return_value.list_target_domains.return_value = [
                {"id": "test", "name": "Test"}
            ]

            from x_creative.target_manager.app import TargetManagerApp
            app = TargetManagerApp()

            result = app.refresh_targets()
            assert len(result) == 1
            assert result[0]["id"] == "test"

    def test_app_title(self):
        """Test app title."""
        with patch(
            "x_creative.target_manager.app.TargetYAMLManager"
        ):
            from x_creative.target_manager.app import TargetManagerApp
            app = TargetManagerApp()
            assert app.TITLE == "xc-target"
