"""Tests for domain search service."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from x_creative.domain_manager.services.search import DomainSearchService


class TestDomainSearchService:
    """Tests for DomainSearchService."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        settings = MagicMock()
        settings.brave_search.api_key.get_secret_value.return_value = "test-key"
        settings.brave_search.base_url = "https://api.search.brave.com/res/v1"
        return settings

    @pytest.fixture
    def service(self, mock_settings):
        """Create service with mock settings."""
        with patch(
            "x_creative.domain_manager.services.search.get_settings",
            return_value=mock_settings,
        ):
            return DomainSearchService()

    async def test_search_domain_concepts(self, service):
        """Test searching for domain concepts."""
        mock_response = {
            "web": {
                "results": [
                    {
                        "title": "Thermodynamics",
                        "url": "https://example.com/thermo",
                        "description": "Study of heat and energy",
                    }
                ]
            }
        }

        with patch.object(
            service._client,
            "get",
            new_callable=AsyncMock,
            return_value=MagicMock(
                json=lambda: mock_response,
                raise_for_status=lambda: None,
            ),
        ):
            results = await service.search_domain_concepts("热力学")
            assert len(results) >= 1

    async def test_build_bilingual_queries(self, service):
        """Test bilingual query generation."""
        queries = service._build_bilingual_queries("热力学", "domain concepts")
        assert len(queries) == 2  # Chinese and English
        assert any("热力学" in q for q in queries)

    async def test_close(self, service):
        """Test closing the client."""
        await service.close()
        # Should not raise
