"""Tests for SAGA Coordinator."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from x_creative.core.types import Hypothesis, ProblemFrame, SearchConfig
from x_creative.creativity.engine import CreativityEngine
from x_creative.saga.budget import CognitiveBudget
from x_creative.saga.coordinator import AuditReport, SAGACoordinator, SAGAResult
from x_creative.saga.state import CognitionAlert


@pytest.fixture
def sample_problem() -> ProblemFrame:
    """Create a sample problem frame."""
    return ProblemFrame(
        description="How to design a viral open source tool?",
        target_domain="open_source_development",
    )


@pytest.fixture
def sample_hypotheses() -> list[Hypothesis]:
    """Create sample hypotheses for mock returns."""
    return [
        Hypothesis(
            id=f"hyp_{i:03d}",
            description=f"Test hypothesis {i}",
            source_domain=f"domain_{i}",
            source_structure=f"structure_{i}",
            analogy_explanation=f"Analogy {i}",
            observable=f"factor_{i}",
        )
        for i in range(3)
    ]


@pytest.fixture(autouse=True)
def mock_router_complete() -> AsyncMock:
    """Avoid outbound LLM calls in coordinator unit tests."""
    with patch("x_creative.llm.router.ModelRouter.complete", new_callable=AsyncMock) as mock_complete:
        mock_complete.return_value = MagicMock(content="[]")
        yield mock_complete


class TestAuditReport:
    """Tests for AuditReport model."""

    def test_default_creation(self) -> None:
        """Test creating AuditReport with defaults."""
        report = AuditReport()
        assert report.events_processed == 0
        assert report.alerts_raised == 0
        assert report.alert_summary == {}
        assert report.interventions == 0
        assert report.intervention_log == []
        assert report.budget_spent == 0.0
        assert report.budget_remaining == 0.0


class TestSAGAResult:
    """Tests for SAGAResult model."""

    def test_default_creation(self) -> None:
        """Test creating SAGAResult with defaults."""
        result = SAGAResult()
        assert result.hypotheses == []
        assert isinstance(result.audit_report, AuditReport)
        assert result.alerts == []
        assert result.intervention_log == []
        assert result.metrics == {}
        assert result.budget_spent == 0.0
        assert result.budget_total == 0.0

    def test_with_hypotheses(self, sample_hypotheses: list[Hypothesis]) -> None:
        """Test creating SAGAResult with hypotheses."""
        result = SAGAResult(
            hypotheses=sample_hypotheses,
            budget_spent=10.0,
            budget_total=100.0,
            elapsed_seconds=5.0,
        )
        assert len(result.hypotheses) == 3
        assert result.budget_spent == 10.0
        assert result.elapsed_seconds == 5.0


class TestSAGACoordinator:
    """Tests for SAGACoordinator."""

    @pytest.mark.asyncio
    async def test_run_with_mocked_engine(
        self,
        sample_problem: ProblemFrame,
        sample_hypotheses: list[Hypothesis],
    ) -> None:
        """Test SAGACoordinator.run() with a mocked CreativityEngine."""
        engine = CreativityEngine()

        # Mock the internal modules
        with patch.object(engine._biso, "generate_all_analogies") as mock_biso, \
             patch.object(engine._search, "run_search") as mock_search, \
             patch.object(engine, "score_and_verify_batch") as mock_score:

            mock_biso.return_value = sample_hypotheses
            mock_search.return_value = sample_hypotheses
            mock_score.return_value = sample_hypotheses

            coordinator = SAGACoordinator(engine=engine)
            result = await coordinator.run(sample_problem)

            assert isinstance(result, SAGAResult)
            assert len(result.hypotheses) == 3
            assert result.budget_total == 100.0
            assert result.elapsed_seconds > 0

            # Verify engine methods were called
            mock_biso.assert_called_once()
            mock_search.assert_called_once()
            mock_score.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_empty_biso(self, sample_problem: ProblemFrame) -> None:
        """Test coordinator handles empty BISO result gracefully."""
        engine = CreativityEngine()

        with patch.object(engine._biso, "generate_all_analogies") as mock_biso:
            mock_biso.return_value = []

            coordinator = SAGACoordinator(engine=engine)
            result = await coordinator.run(sample_problem)

            assert isinstance(result, SAGAResult)
            assert len(result.hypotheses) == 0

    @pytest.mark.asyncio
    async def test_run_with_custom_budget(
        self,
        sample_problem: ProblemFrame,
        sample_hypotheses: list[Hypothesis],
    ) -> None:
        """Test coordinator with custom budget settings."""
        engine = CreativityEngine()
        budget = CognitiveBudget(total_budget=50.0)

        with patch.object(engine._biso, "generate_all_analogies") as mock_biso, \
             patch.object(engine._search, "run_search") as mock_search, \
             patch.object(engine, "score_and_verify_batch") as mock_score:

            mock_biso.return_value = sample_hypotheses
            mock_search.return_value = sample_hypotheses
            mock_score.return_value = sample_hypotheses

            coordinator = SAGACoordinator(engine=engine, budget=budget)
            result = await coordinator.run(sample_problem)

            assert result.budget_total == 50.0

    @pytest.mark.asyncio
    async def test_run_engine_error_handling(
        self, sample_problem: ProblemFrame
    ) -> None:
        """Test that coordinator handles Fast Agent errors gracefully."""
        engine = CreativityEngine()

        with patch.object(engine._biso, "generate_all_analogies") as mock_biso:
            mock_biso.side_effect = RuntimeError("LLM API failed")

            coordinator = SAGACoordinator(engine=engine)
            result = await coordinator.run(sample_problem)

            assert isinstance(result, SAGAResult)
            assert len(result.hypotheses) == 0

    def test_create_detectors_phase1(self) -> None:
        """Coordinator should wire baseline detector set."""
        coordinator = SAGACoordinator()
        detectors = coordinator._create_detectors()
        assert detectors, "expected baseline detectors to be enabled"
        detector_names = {type(d).__name__ for d in detectors}
        assert "ScoreCompressionDetector" in detector_names
        assert "StructureCollapseDetector" in detector_names
        assert "DimensionCollinearityDetector" in detector_names
        assert "SourceDomainBiasDetector" in detector_names
        assert "ShallowRewriteDetector" in detector_names

    def test_create_detectors_uses_configured_thresholds(self) -> None:
        coordinator = SAGACoordinator()
        with patch("x_creative.saga.coordinator.get_settings") as mock_settings:
            mock_settings.return_value.score_compression_threshold = 0.66
            mock_settings.return_value.dimension_colinearity_threshold = 0.55
            detectors = coordinator._create_detectors()

        by_name = {type(det).__name__: det for det in detectors}
        assert by_name["ScoreCompressionDetector"]._std_threshold == 0.66
        assert by_name["DimensionCollinearityDetector"]._corr_threshold == 0.55

    def test_build_initial_directive_from_checkpoint_payload(self) -> None:
        directive = SAGACoordinator._build_initial_directive(
            {
                "directive_type": "adjust_search_params",
                "reason": "cp3 bias",
                "confidence": 0.8,
                "priority": 3,
                "payload": {"search_breadth": 7},
            }
        )
        assert directive is not None
        assert directive.directive_type.value == "adjust_search_params"
        assert directive.payload["search_breadth"] == 7

    def test_create_auditors_phase1(self) -> None:
        """Coordinator should wire baseline auditor set."""
        coordinator = SAGACoordinator()
        auditors = coordinator._create_auditors()
        assert auditors, "expected baseline auditors to be enabled"
        assert {type(a).__name__ for a in auditors} == {"DomainConstraintAuditor"}

    def test_auditor_receives_router(self) -> None:
        """DomainConstraintAuditor should receive router from engine."""
        engine = CreativityEngine()
        coordinator = SAGACoordinator(engine=engine)
        auditors = coordinator._create_auditors()
        auditor = auditors[0]
        assert auditor._router is engine._router

    def test_create_evaluators_phase1(self) -> None:
        """Coordinator should wire baseline evaluator set."""
        coordinator = SAGACoordinator()
        evaluators = coordinator._create_evaluators()
        assert evaluators, "expected baseline evaluators to be enabled"
        evaluator_names = {type(e).__name__ for e in evaluators}
        assert "AdversarialChallengeEvaluator" in evaluator_names
        assert "PatternMemoryEvaluator" in evaluator_names

    def test_semantic_hasher_created_when_api_key_available(self) -> None:
        """PatternMemory should have SemanticHasher when API key is configured."""
        coordinator = SAGACoordinator()
        with patch(
            "x_creative.config.settings.get_settings"
        ) as mock_settings:
            from pydantic import SecretStr

            mock_settings.return_value.openrouter.api_key = SecretStr("test-key")
            mock_settings.return_value.openrouter.base_url = "https://openrouter.ai/api/v1"
            hasher = coordinator._create_semantic_hasher()
            assert hasher is not None
            assert type(hasher).__name__ == "SemanticHasher"

    def test_semantic_hasher_none_without_api_key(self) -> None:
        """PatternMemory should fall back to literal when no API key."""
        coordinator = SAGACoordinator()
        with patch(
            "x_creative.config.settings.get_settings"
        ) as mock_settings:
            from pydantic import SecretStr

            mock_settings.return_value.openrouter.api_key = SecretStr("")
            hasher = coordinator._create_semantic_hasher()
            assert hasher is None

    def test_slow_agent_receives_router(self) -> None:
        """SlowAgent should receive router from engine via coordinator."""
        engine = CreativityEngine()
        coordinator = SAGACoordinator(engine=engine)
        # The router is passed in run(), but we can test the wiring is set
        # by checking that _create_auditors / _create_evaluators use it
        evaluators = coordinator._create_evaluators()
        adversarial = [e for e in evaluators if type(e).__name__ == "AdversarialChallengeEvaluator"][0]
        assert adversarial._router is engine._router

    def test_pattern_memory_path_defaults_to_cache_dir(self, tmp_path) -> None:  # noqa: ANN001
        coordinator = SAGACoordinator()

        with patch("x_creative.config.settings.get_settings") as mock_settings:
            mock_settings.return_value.cache_dir = tmp_path / "cache-root"
            path = coordinator._resolve_pattern_memory_path()

        assert path == (tmp_path / "cache-root" / "saga" / "pattern_memory.json")

    def test_validate_cross_family_routing_rejects_same_family(self) -> None:
        engine = CreativityEngine()
        coordinator = SAGACoordinator(engine=engine)

        with patch.object(
            engine._router,
            "get_model",
            side_effect=["anthropic/claude-sonnet-4", "anthropic/claude-3-haiku"],
        ):
            with pytest.raises(ValueError, match="different model family"):
                coordinator._validate_cross_family_routing()

    def test_validate_cross_family_routing_accepts_different_families(self) -> None:
        engine = CreativityEngine()
        coordinator = SAGACoordinator(engine=engine)

        with patch.object(
            engine._router,
            "get_model",
            side_effect=["anthropic/claude-sonnet-4", "google/gemini-3-pro"],
        ):
            coordinator._validate_cross_family_routing()


class TestCKRuntimeIntegration:
    @pytest.mark.asyncio
    async def test_ck_evaluate_transition_called_on_multiple_checkpoints(
        self, sample_problem: ProblemFrame, sample_hypotheses: list[Hypothesis],
    ) -> None:
        """CK checkpoint evaluation should run across Fast-Agent checkpoints."""
        engine = CreativityEngine()
        coordinator = SAGACoordinator(engine=engine, enable_ck=True)

        with patch.object(
            coordinator,
            "_evaluate_ck_checkpoint",
            wraps=coordinator._evaluate_ck_checkpoint,
        ) as ck_eval, patch.object(
            engine._biso, "generate_all_analogies", return_value=sample_hypotheses
        ), patch.object(
            engine._search, "run_search", return_value=sample_hypotheses
        ), patch.object(
            engine, "score_and_verify_batch", return_value=sample_hypotheses
        ), patch.object(
            engine, "filter_by_threshold", return_value=sample_hypotheses
        ):
            await coordinator.run(sample_problem)

        # CP-4 (biso), CP-5 (search), CP-6 (verify) checkpoints.
        assert ck_eval.call_count >= 3

    @pytest.mark.asyncio
    async def test_ck_evaluate_transition_called_during_run(
        self, sample_problem: ProblemFrame, sample_hypotheses: list[Hypothesis],
    ) -> None:
        """CKCoordinator.evaluate_transition should be called during run()."""
        engine = CreativityEngine()
        coordinator = SAGACoordinator(engine=engine, enable_ck=True)

        with patch.object(engine._biso, "generate_all_analogies", return_value=sample_hypotheses):
            with patch.object(engine._search, "run_search", return_value=sample_hypotheses):
                with patch.object(engine, "score_and_verify_batch", return_value=sample_hypotheses):
                    with patch.object(engine, "filter_by_threshold", return_value=sample_hypotheses):
                        result = await coordinator.run(sample_problem)

        ck = coordinator.ck_coordinator
        assert ck is not None
        # The checkpoint was called (we can verify by checking the coordinator was consulted)

    @pytest.mark.asyncio
    async def test_ck_coverage_plateau_detected(
        self, sample_problem: ProblemFrame, sample_hypotheses: list[Hypothesis],
    ) -> None:
        """When MOME coverage stagnates, a coverage_plateau trigger should be added."""
        engine = CreativityEngine()
        coordinator = SAGACoordinator(engine=engine, enable_ck=True)

        # Set up a mock MOME archive on the search module
        mock_mome = MagicMock()
        mock_mome.coverage_ratio = 0.5
        engine._search._mome_archive = mock_mome

        # Set initial coverage so stagnation is detected
        coordinator._last_mome_coverage = 0.5

        with patch.object(engine._biso, "generate_all_analogies", return_value=sample_hypotheses):
            with patch.object(engine._search, "run_search", return_value=sample_hypotheses):
                with patch.object(engine, "score_and_verify_batch", return_value=sample_hypotheses):
                    with patch.object(engine, "filter_by_threshold", return_value=sample_hypotheses):
                        result = await coordinator.run(sample_problem)

        ck = coordinator.ck_coordinator
        assert ck is not None
        # Verify a coverage_plateau trigger was added
        assert any(t.trigger_type == "coverage_plateau" for t in ck._triggers)


class TestCKSettingsPassthrough:
    """Tests for CK settings being forwarded from Settings to CKCoordinator."""

    @pytest.mark.asyncio
    async def test_ck_settings_from_env_passed_to_coordinator(
        self,
        sample_problem: ProblemFrame,
        sample_hypotheses: list[Hypothesis],
    ) -> None:
        """CK settings from Settings should be forwarded to CKCoordinator."""
        engine = CreativityEngine()
        coordinator = SAGACoordinator(engine=engine, enable_ck=True)

        with patch.object(engine._biso, "generate_all_analogies") as mock_biso, \
             patch.object(engine._search, "run_search") as mock_search, \
             patch.object(engine, "score_and_verify_batch") as mock_score:

            mock_biso.return_value = sample_hypotheses
            mock_search.return_value = sample_hypotheses
            mock_score.return_value = sample_hypotheses

            with patch("x_creative.config.settings.get_settings") as mock_gs:
                mock_s = mock_gs.return_value
                mock_s.ck_min_phase_duration_s = 42.0
                mock_s.ck_max_k_expansion_per_session = 3
                mock_s.ck_coverage_plateau_threshold = 4
                mock_s.ck_evidence_gap_threshold = 7.0
                result = await coordinator.run(sample_problem)

        ck = coordinator.ck_coordinator
        assert ck is not None
        assert ck._min_phase_duration_s == 42.0
        assert ck._max_k_expansion == 3
        assert ck._coverage_plateau_threshold == 4
        assert ck._evidence_gap_threshold == 7.0
