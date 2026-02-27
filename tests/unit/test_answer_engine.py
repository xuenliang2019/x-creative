"""Tests for AnswerEngine."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from x_creative.answer.types import AnswerConfig, FrameBuildResult
from x_creative.core.plugin import TargetDomainPlugin
from x_creative.core.types import Hypothesis, ProblemFrame
from x_creative.session.types import Session


def _mock_frame():
    return ProblemFrame(
        description="How to design a viral open source tool?",
        target_domain="open_source_development",
        objective="Design a viral open source tool",
        domain_hint={"domain_id": "open_source_development", "confidence": 0.9},
    )


def _mock_plugin():
    return TargetDomainPlugin(id="open_source_development", name="开源软件开发选题", description="Open source development")


def _mock_hypothesis(idx: int):
    return Hypothesis(
        id=f"h_{idx}", description=f"Hypothesis {idx}",
        source_domain=f"domain_{idx}", source_structure=f"structure_{idx}",
        analogy_explanation=f"Analogy {idx}", observable=f"measure_{idx}",
    )


class TestAnswerEngine:
    @pytest.mark.asyncio
    async def test_answer_exits_on_conflicting_user_constraints(self, tmp_path):
        """If user constraints contradict, AnswerEngine should raise and stop early."""
        from x_creative.answer.engine import AnswerEngine

        frame = ProblemFrame(
            description="test",
            constraints=[
                "must use MIT license",
                "do not use MIT license",
                "must support Python 3.12",
                "do not support Python 3.12",
            ],
        )
        session = Session(id="test-constraints", topic="conflicts")

        with (
            patch("x_creative.answer.engine.SessionManager") as MockSM,
            patch("x_creative.answer.engine.ProblemFrameBuilder") as MockPFB,
            patch("x_creative.answer.engine.TargetDomainResolver") as MockTDR,
        ):
            mock_sm = MockSM.return_value
            mock_sm.create_session.return_value = session
            mock_sm.save_stage_data = MagicMock()
            mock_sm.data_dir = tmp_path

            mock_pfb = MockPFB.return_value
            mock_pfb.build = AsyncMock(return_value=FrameBuildResult(frame=frame, confidence=0.9))

            mock_tdr = MockTDR.return_value
            mock_tdr.resolve = AsyncMock()

            engine = AnswerEngine(config=AnswerConfig(saga_enabled=False, auto_refine=False))

            from x_creative.answer.constraint_preflight import UserConstraintConflictError

            with pytest.raises(UserConstraintConflictError) as excinfo:
                await engine.answer("test")

            report = excinfo.value.report
            assert report["conflict_pairs"]
            # Must not proceed to later stages.
            assert mock_tdr.resolve.await_count == 0

    def test_cp1_audit_detects_conflicts_and_undecidable_constraints(self):
        from x_creative.answer.engine import AnswerEngine

        frame = ProblemFrame(
            description="test",
            constraints=[
                "must use MIT license",
                "do not use MIT license",
                "please optimize for better quality",
            ],
        )
        result = AnswerEngine._cp1_constraint_audit(frame)
        assert result["conflict_pairs"]
        assert result["undecidable_constraints"]
        assert result["passed"] is False

    def test_checkpoint_directives_apply_before_generation(self):
        from x_creative.answer.engine import AnswerEngine
        from x_creative.core.types import SearchConfig

        frame = ProblemFrame(description="test", constraints=["must be cross-platform"])
        search_cfg = SearchConfig(search_depth=2, search_breadth=3)
        directives = [
            {
                "directive_type": "add_constraint",
                "payload": {"constraint": "all constraints must be operationally testable"},
            },
            {
                "directive_type": "adjust_search_params",
                "payload": {"search_breadth": 7},
            },
        ]

        updated_frame, updated_cfg = AnswerEngine._apply_checkpoint_directives(
            frame,
            search_cfg,
            directives,
        )
        assert "all constraints must be operationally testable" in updated_frame.constraints
        assert updated_cfg.search_breadth == 7

    def test_cp2_cp3_findings_generate_slow_agent_style_directives(self):
        from x_creative.answer.engine import AnswerEngine

        directives = AnswerEngine._derive_checkpoint_directives(
            cp1_result={"conflict_pairs": [], "undecidable_constraints": []},
            cp2_result={"warnings": ["high_confidence_hint_mismatch"]},
            cp3_result={"bias_flags": ["dominant_domain_bias"]},
        )
        directive_types = {d["directive_type"] for d in directives}
        assert "adjust_search_params" in directive_types

    @pytest.mark.asyncio
    async def test_answer_full_pipeline(self, tmp_path):
        """End-to-end: question -> AnswerPack."""
        from x_creative.answer.engine import AnswerEngine

        frame = _mock_frame()
        plugin = _mock_plugin()
        hypotheses = [_mock_hypothesis(i) for i in range(3)]
        session = Session(id="test-123", topic="viral-tool")

        with (
            patch("x_creative.answer.engine.SessionManager") as MockSM,
            patch("x_creative.answer.engine.ProblemFrameBuilder") as MockPFB,
            patch("x_creative.answer.engine.TargetDomainResolver") as MockTDR,
            patch("x_creative.answer.engine.SourceDomainSelector") as MockSDS,
            patch("x_creative.answer.engine.CreativityEngine") as MockCE,
            patch("x_creative.answer.engine.TalkerReasonerSolver") as MockSolver,
        ):
            # Session
            mock_sm = MockSM.return_value
            mock_sm.create_session.return_value = session
            mock_sm.save_stage_data = MagicMock()
            mock_sm.data_dir = tmp_path

            # ProblemFrameBuilder
            mock_pfb = MockPFB.return_value
            mock_pfb.build = AsyncMock(return_value=FrameBuildResult(frame=frame, confidence=0.9))

            # TargetDomainResolver
            mock_tdr = MockTDR.return_value
            mock_tdr.resolve = AsyncMock(return_value=plugin)

            # SourceDomainSelector
            mock_sds = MockSDS.return_value
            # Real selector returns Domain, but engine only needs .id and list pass-through.
            source_domains = [MagicMock(id="fluid_dynamics"), MagicMock(id="epidemiology")]
            mock_sds.select = AsyncMock(return_value=source_domains)

            # CreativityEngine — generate returns hypotheses (already scored/verified)
            mock_ce = MockCE.return_value
            mock_ce.generate = AsyncMock(return_value=hypotheses)
            mock_ce.close = AsyncMock()

            # Solver (won't be called because auto_refine=False)
            mock_solver = MockSolver.return_value
            mock_solver.run = AsyncMock()
            mock_solver.close = AsyncMock()

            config = AnswerConfig(saga_enabled=False, auto_refine=False)
            engine = AnswerEngine(config=config)
            pack = await engine.answer("How to design a viral open source tool?")

        assert pack.needs_clarification is False
        assert pack.session_id == "test-123"
        assert "# How to design a viral open source tool?" in pack.answer_md
        assert pack.answer_json["version"] == "1.0"
        assert (tmp_path / session.id / "answer.md").exists()
        assert (tmp_path / session.id / "answer.json").exists()
        # SourceDomainSelector output must be forwarded into generation path.
        _, kwargs = mock_ce.generate.call_args
        assert kwargs["source_domains"] == source_domains

    @pytest.mark.asyncio
    async def test_answer_clarification_needed(self):
        """When ProblemFrameBuilder needs clarification, return immediately."""
        from x_creative.answer.engine import AnswerEngine

        partial = ProblemFrame(description="vague")
        session = Session(id="test-456", topic="vague")

        with (
            patch("x_creative.answer.engine.SessionManager") as MockSM,
            patch("x_creative.answer.engine.ProblemFrameBuilder") as MockPFB,
        ):
            mock_sm = MockSM.return_value
            mock_sm.create_session.return_value = session

            mock_pfb = MockPFB.return_value
            mock_pfb.build = AsyncMock(
                return_value=FrameBuildResult(
                    frame=None,
                    needs_clarification=True,
                    clarification_question="Which domain?",
                    partial_frame=partial,
                    confidence=0.1,
                )
            )

            config = AnswerConfig(saga_enabled=False)
            engine = AnswerEngine(config=config)
            pack = await engine.answer("Do the thing")

        assert pack.needs_clarification is True
        assert pack.clarification_question == "Which domain?"

    @pytest.mark.asyncio
    async def test_answer_uses_saga_when_enabled(self, tmp_path):
        """saga_enabled=True should route BISO→SEARCH→VERIFY through SAGACoordinator."""
        from x_creative.answer.engine import AnswerEngine

        frame = _mock_frame()
        plugin = _mock_plugin()
        hypotheses = [_mock_hypothesis(i) for i in range(2)]
        session = Session(id="test-789", topic="viral-tool")
        source_domains = [MagicMock(id="fluid_dynamics"), MagicMock(id="ecology")]

        with (
            patch("x_creative.answer.engine.SessionManager") as MockSM,
            patch("x_creative.answer.engine.ProblemFrameBuilder") as MockPFB,
            patch("x_creative.answer.engine.TargetDomainResolver") as MockTDR,
            patch("x_creative.answer.engine.SourceDomainSelector") as MockSDS,
            patch("x_creative.answer.engine.SAGACoordinator") as MockSAGA,
            patch("x_creative.answer.engine.CreativityEngine") as MockCE,
        ):
            mock_sm = MockSM.return_value
            mock_sm.create_session.return_value = session
            mock_sm.save_stage_data = MagicMock()
            mock_sm.data_dir = tmp_path

            mock_pfb = MockPFB.return_value
            mock_pfb.build = AsyncMock(return_value=FrameBuildResult(frame=frame, confidence=0.9))

            mock_tdr = MockTDR.return_value
            mock_tdr.resolve = AsyncMock(return_value=plugin)

            mock_sds = MockSDS.return_value
            mock_sds.select = AsyncMock(return_value=source_domains)

            saga_result = MagicMock()
            saga_result.hypotheses = hypotheses
            saga_result.intervention_log = [{"type": "audit"}]
            saga_result.budget_spent = 12.0
            saga_result.metrics = {"search_rounds": 3}
            MockSAGA.return_value.run = AsyncMock(return_value=saga_result)

            # Should not be used in saga path.
            MockCE.return_value.generate = AsyncMock(return_value=hypotheses)
            MockCE.return_value.close = AsyncMock()

            engine = AnswerEngine(config=AnswerConfig(saga_enabled=True, auto_refine=False))
            pack = await engine.answer("How to design a viral open source tool?")

        assert pack.needs_clarification is False
        assert MockSAGA.return_value.run.await_count == 1
        _, kwargs = MockSAGA.return_value.run.call_args
        assert kwargs["source_domains"] == source_domains
        assert "initial_directives" in kwargs

    @pytest.mark.asyncio
    async def test_answer_quick_mode_wires_runtime_and_operator_settings(self, tmp_path):
        """AnswerEngine should apply mode-based depth/breadth and settings-based runtime profile."""
        from x_creative.answer.engine import AnswerEngine

        frame = _mock_frame()
        plugin = _mock_plugin()
        hypotheses = [_mock_hypothesis(i) for i in range(2)]
        session = Session(id="test-mode", topic="viral-tool")

        fake_settings = MagicMock()
        fake_settings.runtime_profile = "interactive"
        fake_settings.enable_extreme = True
        fake_settings.enable_blending = True
        fake_settings.enable_transform_space = True
        fake_settings.max_blend_pairs = 4
        fake_settings.max_transform_hypotheses = 3
        fake_settings.blend_expand_budget_per_round = 6
        fake_settings.transform_space_budget_per_round = 5
        fake_settings.hyperpath_expand_topN = 9
        fake_settings.ck_enabled = False
        fake_settings.constraint_similarity_threshold = 0.6
        fake_settings.max_constraints = 15

        with (
            patch("x_creative.answer.engine.get_settings", return_value=fake_settings),
            patch("x_creative.answer.engine.SessionManager") as MockSM,
            patch("x_creative.answer.engine.ProblemFrameBuilder") as MockPFB,
            patch("x_creative.answer.engine.TargetDomainResolver") as MockTDR,
            patch("x_creative.answer.engine.SourceDomainSelector") as MockSDS,
            patch("x_creative.answer.engine.CreativityEngine") as MockCE,
        ):
            mock_sm = MockSM.return_value
            mock_sm.create_session.return_value = session
            mock_sm.save_stage_data = MagicMock()
            mock_sm.data_dir = tmp_path

            MockPFB.return_value.build = AsyncMock(
                return_value=FrameBuildResult(frame=frame, confidence=0.9)
            )
            MockTDR.return_value.resolve = AsyncMock(return_value=plugin)
            source_domains = [MagicMock(id="fluid_dynamics"), MagicMock(id="ecology")]
            MockSDS.return_value.select = AsyncMock(return_value=source_domains)

            mock_ce = MockCE.return_value
            mock_ce.generate = AsyncMock(return_value=hypotheses)
            mock_ce.close = AsyncMock()

            config = AnswerConfig(
                saga_enabled=False,
                auto_refine=False,
                mode="quick",
                search_depth=5,
                search_breadth=7,
            )
            engine = AnswerEngine(config=config)
            await engine.answer("How to design a viral open source tool?")

        _, kwargs = mock_ce.generate.call_args
        search_cfg = kwargs["config"]
        assert search_cfg.search_depth == 2
        assert search_cfg.search_breadth == 3
        assert search_cfg.runtime_profile == "interactive"
        assert search_cfg.enable_blending is True
        assert search_cfg.enable_transform_space is True
        assert search_cfg.blend_expand_budget_per_round == 6
        assert search_cfg.transform_space_budget_per_round == 5
        assert search_cfg.hyperpath_expand_topN == 9

    @pytest.mark.asyncio
    async def test_answer_hkg_flag_disables_hkg_on_engine(self, tmp_path):
        from x_creative.answer.engine import AnswerEngine

        frame = _mock_frame()
        plugin = _mock_plugin()
        hypotheses = [_mock_hypothesis(0)]
        session = Session(id="test-hkg", topic="viral-tool")

        with (
            patch("x_creative.answer.engine.SessionManager") as MockSM,
            patch("x_creative.answer.engine.ProblemFrameBuilder") as MockPFB,
            patch("x_creative.answer.engine.TargetDomainResolver") as MockTDR,
            patch("x_creative.answer.engine.SourceDomainSelector") as MockSDS,
            patch("x_creative.answer.engine.CreativityEngine") as MockCE,
        ):
            mock_sm = MockSM.return_value
            mock_sm.create_session.return_value = session
            mock_sm.save_stage_data = MagicMock()
            mock_sm.data_dir = tmp_path

            MockPFB.return_value.build = AsyncMock(
                return_value=FrameBuildResult(frame=frame, confidence=0.9)
            )
            MockTDR.return_value.resolve = AsyncMock(return_value=plugin)
            MockSDS.return_value.select = AsyncMock(return_value=[MagicMock(id="queueing_theory")])

            mock_ce = MockCE.return_value
            mock_ce.generate = AsyncMock(return_value=hypotheses)
            mock_ce.close = AsyncMock()
            mock_ce.set_hkg_enabled = MagicMock()

            config = AnswerConfig(saga_enabled=False, auto_refine=False, hkg_enabled=False)
            engine = AnswerEngine(config=config)
            await engine.answer("How to design a viral open source tool?")

        mock_ce.set_hkg_enabled.assert_called_once_with(False)

    @pytest.mark.asyncio
    async def test_answer_saga_budget_uses_cli_budget(self, tmp_path):
        from x_creative.answer.engine import AnswerEngine

        frame = _mock_frame()
        plugin = _mock_plugin()
        hypotheses = [_mock_hypothesis(i) for i in range(2)]
        session = Session(id="test-budget", topic="viral-tool")
        source_domains = [MagicMock(id="fluid_dynamics"), MagicMock(id="ecology")]

        allocation = MagicMock()
        allocation.emergency_reserve = 10.0
        allocation.domain_audit = 9.0
        allocation.biso_monitor = 13.5
        allocation.search_monitor = 13.5
        allocation.verify_monitor = 18.0
        allocation.adversarial = 22.5
        allocation.global_review = 13.5

        fake_settings = MagicMock()
        fake_settings.runtime_profile = "interactive"
        fake_settings.enable_extreme = True
        fake_settings.enable_blending = False
        fake_settings.enable_transform_space = False
        fake_settings.max_blend_pairs = 3
        fake_settings.max_transform_hypotheses = 2
        fake_settings.blend_expand_budget_per_round = 3
        fake_settings.transform_space_budget_per_round = 2
        fake_settings.hyperpath_expand_topN = 5
        fake_settings.ck_enabled = False
        fake_settings.constraint_similarity_threshold = 0.6
        fake_settings.max_constraints = 15
        fake_settings.saga_cognitive_budget_allocation = allocation

        with (
            patch("x_creative.answer.engine.get_settings", return_value=fake_settings),
            patch("x_creative.answer.engine.SessionManager") as MockSM,
            patch("x_creative.answer.engine.ProblemFrameBuilder") as MockPFB,
            patch("x_creative.answer.engine.TargetDomainResolver") as MockTDR,
            patch("x_creative.answer.engine.SourceDomainSelector") as MockSDS,
            patch("x_creative.answer.engine.SAGACoordinator") as MockSAGA,
            patch("x_creative.answer.engine.CreativityEngine") as MockCE,
        ):
            mock_sm = MockSM.return_value
            mock_sm.create_session.return_value = session
            mock_sm.save_stage_data = MagicMock()
            mock_sm.data_dir = tmp_path

            MockPFB.return_value.build = AsyncMock(
                return_value=FrameBuildResult(frame=frame, confidence=0.9)
            )
            MockTDR.return_value.resolve = AsyncMock(return_value=plugin)
            MockSDS.return_value.select = AsyncMock(return_value=source_domains)

            saga_result = MagicMock()
            saga_result.hypotheses = hypotheses
            saga_result.intervention_log = []
            saga_result.budget_spent = 12.0
            saga_result.metrics = {"search_rounds": 3}
            MockSAGA.return_value.run = AsyncMock(return_value=saga_result)

            MockCE.return_value.close = AsyncMock()
            MockCE.return_value.set_hkg_enabled = MagicMock()

            cfg = AnswerConfig(saga_enabled=True, auto_refine=False, budget=321)
            engine = AnswerEngine(config=cfg)
            await engine.answer("How to design a viral open source tool?")

        _, kwargs = MockSAGA.call_args
        budget = kwargs["budget"]
        assert budget.total_budget == pytest.approx(321.0)
        assert budget.reserve_ratio == pytest.approx(0.1)
