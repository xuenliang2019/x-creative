"""Tests for CKCoordinator integration in SAGACoordinator."""

import pytest
from unittest.mock import AsyncMock

from x_creative.config.settings import Settings


class TestCKSettings:
    def test_ck_defaults(self) -> None:
        s = Settings()
        assert s.ck_enabled is False
        assert s.ck_min_phase_duration_s == 10.0
        assert s.ck_max_k_expansion_per_session == 5

    def test_ck_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("X_CREATIVE_CK_ENABLED", "true")
        s = Settings()
        assert s.ck_enabled is True
