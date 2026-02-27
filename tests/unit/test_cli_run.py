"""Tests for run CLI commands."""

import json

import pytest
from typer.testing import CliRunner


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def temp_data_dir(tmp_path, monkeypatch):
    """Set up temporary data directory."""
    data_dir = tmp_path / "local_data"
    data_dir.mkdir()
    monkeypatch.setenv("X_CREATIVE_DATA_DIR", str(data_dir))
    return data_dir


class TestRunProblem:
    def test_run_problem_no_session(self, runner, temp_data_dir):
        from x_creative.cli.main import app

        result = runner.invoke(app, ["run", "problem", "--description", "Test"])
        # Should fail because no session
        assert result.exit_code != 0 or "No current session" in result.stdout

    def test_run_problem_with_session(self, runner, temp_data_dir):
        from x_creative.cli.main import app

        runner.invoke(app, ["session", "new", "Test Session", "--id", "test"])
        result = runner.invoke(app, [
            "run", "problem",
            "--description", "Test problem description",
            "--target-domain", "general",
            "--constraint", "Constraint 1",
            "--constraint", "Constraint 2",
        ])
        assert result.exit_code == 0
        assert "problem" in result.stdout.lower() or "completed" in result.stdout.lower()

        # Verify files created
        assert (temp_data_dir / "test" / "problem.json").exists()
        assert (temp_data_dir / "test" / "problem.md").exists()

    def test_run_problem_with_context(self, runner, temp_data_dir):
        from x_creative.cli.main import app

        runner.invoke(app, ["session", "new", "Test Session", "--id", "ctx-test"])
        result = runner.invoke(app, [
            "run", "problem",
            "--description", "Test with context",
            "--target-domain", "open_source_development",
            "--context", '{"language": "python", "license": "MIT"}',
        ])
        assert result.exit_code == 0

        # Verify context is saved
        import json
        with open(temp_data_dir / "ctx-test" / "problem.json") as f:
            data = json.load(f)
        assert data["context"]["language"] == "python"
        assert data["target_domain"] == "open_source_development"

    def test_run_problem_invalid_context(self, runner, temp_data_dir):
        from x_creative.cli.main import app

        runner.invoke(app, ["session", "new", "Test", "--id", "invalid-ctx"])
        result = runner.invoke(app, [
            "run", "problem",
            "--description", "Test",
            "--context", "not valid json",
        ])
        assert result.exit_code != 0
        assert "Invalid JSON" in result.stdout


class TestRunBiso:
    def test_run_biso_no_problem(self, runner, temp_data_dir):
        from x_creative.cli.main import app

        runner.invoke(app, ["session", "new", "Test", "--id", "test"])
        result = runner.invoke(app, ["run", "biso"])
        # Should fail because problem not completed
        assert result.exit_code != 0 or "problem" in result.stdout.lower()

    def test_run_biso_applies_stage_b_feasibility_filter(
        self, runner, temp_data_dir, monkeypatch
    ):
        from x_creative.cli.main import app
        from x_creative.core.plugin import TargetDomainPlugin

        runner.invoke(app, ["session", "new", "BISO Session", "--id", "biso-stageb"])
        runner.invoke(
            app,
            [
                "run",
                "problem",
                "--session",
                "biso-stageb",
                "--description",
                "Test BISO Stage-B filter wiring",
                "--target-domain",
                "general",
            ],
        )

        plugin = TargetDomainPlugin(
            id="general",
            name="General",
            description="General domain",
            source_domains=[
                {"id": "d1", "name": "D1", "description": "Domain 1"},
                {"id": "d2", "name": "D2", "description": "Domain 2"},
            ],
        )
        monkeypatch.setattr("x_creative.cli.run.load_target_domain", lambda _domain_id: plugin)

        filtered = [plugin.get_domain_library().get("d1")]
        filtered = [d for d in filtered if d is not None]

        async def fake_filter(self, frame, target):  # noqa: ANN001, ANN202
            return filtered

        monkeypatch.setattr(
            "x_creative.answer.source_selector.SourceDomainSelector.filter_by_mapping_feasibility",
            fake_filter,
        )

        captured: dict[str, object] = {}

        async def fake_generate(
            self,
            problem,
            num_per_domain=3,
            max_domains=None,
            max_concurrency=None,
            source_domains=None,
        ):  # noqa: ANN001, ANN201
            captured["source_domains"] = source_domains
            return []

        monkeypatch.setattr(
            "x_creative.creativity.biso.BISOModule.generate_all_analogies",
            fake_generate,
        )

        result = runner.invoke(app, ["run", "biso", "--session", "biso-stageb"])
        assert result.exit_code == 0
        assert isinstance(captured.get("source_domains"), list)
        assert len(captured["source_domains"]) == 1
        assert captured["source_domains"][0].id == "d1"


class TestRunWithSessionOption:
    def test_run_with_session_override(self, runner, temp_data_dir):
        from x_creative.cli.main import app

        runner.invoke(app, ["session", "new", "Session 1", "--id", "s1"])
        runner.invoke(app, ["session", "new", "Session 2", "--id", "s2"])

        # Run problem on s1 even though s2 is current
        result = runner.invoke(app, [
            "run", "problem",
            "--session", "s1",
            "--description", "Test",
        ])
        assert result.exit_code == 0

        # Verify s1 has the problem file
        assert (temp_data_dir / "s1" / "problem.json").exists()
        assert not (temp_data_dir / "s2" / "problem.json").exists()


class TestRunSearchWiring:
    def test_run_search_wires_runtime_profile_and_mapping_gate(
        self, runner, temp_data_dir, monkeypatch
    ):
        from x_creative.cli.main import app
        from x_creative.session.manager import SessionManager
        from x_creative.session.types import StageStatus

        runner.invoke(app, ["session", "new", "Search Session", "--id", "search-wire"])
        manager = SessionManager(data_dir=temp_data_dir)
        manager.save_stage_data(
            "search-wire",
            "problem",
            {
                "description": "Test search wiring",
                "target_domain": "general",
                "constraints": [],
                "context": {},
            },
        )
        manager.save_stage_data(
            "search-wire",
            "biso",
            {
                "hypotheses": [
                    {
                        "id": "h1",
                        "description": "Hypothesis",
                        "source_domain": "d",
                        "source_structure": "s",
                        "analogy_explanation": "a",
                        "observable": "obs",
                        "mapping_quality": 8.0,
                    }
                ]
            },
        )
        manager.update_stage_status("search-wire", "problem", StageStatus.COMPLETED)
        manager.update_stage_status("search-wire", "biso", StageStatus.COMPLETED)

        class DummySettings:
            enable_extreme = True
            enable_blending = False
            enable_transform_space = False
            max_blend_pairs = 3
            max_transform_hypotheses = 2
            runtime_profile = "interactive"
            blend_expand_budget_per_round = 7
            transform_space_budget_per_round = 6
            hyperpath_expand_topN = 4
            mapping_quality_gate_enabled = True
            mapping_quality_gate_threshold = 6.5

        monkeypatch.setattr("x_creative.cli.run.get_settings", lambda: DummySettings())

        class DummyRouter:
            async def close(self):  # noqa: ANN201
                return None

        monkeypatch.setattr("x_creative.llm.router.ModelRouter", DummyRouter)

        captured: dict[str, object] = {}

        class DummySearchModule:
            def __init__(self, router=None, mapping_quality_gate=None):  # noqa: ANN001
                captured["mapping_quality_gate"] = mapping_quality_gate

            async def run_search(self, initial_hypotheses, config):  # noqa: ANN001, ANN201
                captured["runtime_profile"] = config.runtime_profile
                captured["blend_budget"] = config.blend_expand_budget_per_round
                captured["transform_budget"] = config.transform_space_budget_per_round
                captured["hyperpath_topn"] = config.hyperpath_expand_topN
                return initial_hypotheses

        monkeypatch.setattr("x_creative.creativity.search.SearchModule", DummySearchModule)

        result = runner.invoke(app, ["run", "search", "--session", "search-wire"])
        assert result.exit_code == 0
        assert captured["mapping_quality_gate"] == 6.5
        assert captured["runtime_profile"] == "interactive"
        assert captured["blend_budget"] == 7
        assert captured["transform_budget"] == 6
        assert captured["hyperpath_topn"] == 4


class TestRunSolve:
    def test_run_solve_requires_verify_completed(self, runner, temp_data_dir):
        from x_creative.cli.main import app

        runner.invoke(app, ["session", "new", "Solve Session", "--id", "solve-req"])
        result = runner.invoke(app, ["run", "solve"])

        assert result.exit_code != 0
        assert "verify" in result.stdout.lower()

    def test_run_solve_generates_artifacts(self, runner, temp_data_dir, monkeypatch):
        from x_creative.cli.main import app
        from x_creative.core.types import ProblemFrame
        from x_creative.session.manager import SessionManager
        from x_creative.session.types import StageStatus

        runner.invoke(app, ["session", "new", "Solve Session", "--id", "solve-ok"])
        manager = SessionManager(data_dir=temp_data_dir)

        problem = ProblemFrame(
            description="Build a practical execution plan for reducing churn",
            target_domain="general",
        )
        manager.save_stage_data("solve-ok", "problem", problem.model_dump())

        verify_hypothesis = {
            "id": "hyp_001",
            "description": "Use queueing-theory inspired prioritization for support tickets",
            "source_domain": "queueing_theory",
            "source_structure": "priority_queue",
            "analogy_explanation": "Map service queues to ticket triage",
            "observable": "ticket_age_weighted_resolution_time",
            "scores": {
                "divergence": 6.0,
                "testability": 7.0,
                "rationale": 7.0,
                "robustness": 6.5,
                "feasibility": 7.5,
            },
        }
        manager.save_stage_data("solve-ok", "verify", {"hypotheses": [verify_hypothesis]})

        verify_md = temp_data_dir / "solve-ok" / "verify.md"
        verify_md.write_text("# Verification Results\n\nIdea: Prioritize high-risk tickets first.\n", encoding="utf-8")

        manager.update_stage_status("solve-ok", "problem", StageStatus.COMPLETED)
        manager.update_stage_status("solve-ok", "biso", StageStatus.COMPLETED)
        manager.update_stage_status("solve-ok", "search", StageStatus.COMPLETED)
        manager.update_stage_status("solve-ok", "verify", StageStatus.COMPLETED)

        class DummySolver:
            def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
                pass

            async def run(self, problem, verify_markdown, hypotheses, max_ideas, max_web_results):
                return {
                    "solution_markdown": "# Final Solution\n\nUse staged rollout.\n",
                    "applied_constraints": ["all claims must cite evidence"],
                    "evidence": [],
                    "metrics": {"idea_count": 1},
                    "audit_report": {"events_processed": 1},
                }

        monkeypatch.setattr("x_creative.cli.run.SAGASolver", DummySolver)

        result = runner.invoke(app, ["run", "solve", "--session", "solve-ok"])
        assert result.exit_code == 0

        solve_md = temp_data_dir / "solve-ok" / "solve.md"
        solve_json = temp_data_dir / "solve-ok" / "solve.json"

        assert solve_md.exists()
        assert solve_json.exists()
        assert "Final Solution" in solve_md.read_text(encoding="utf-8")

        data = json.loads(solve_json.read_text(encoding="utf-8"))
        assert data["metrics"]["idea_count"] == 1

    def test_run_solve_auto_refine_enabled_by_default(self, runner, temp_data_dir, monkeypatch):
        from x_creative.cli.main import app
        from x_creative.core.types import ProblemFrame
        from x_creative.session.manager import SessionManager
        from x_creative.session.types import StageStatus

        runner.invoke(app, ["session", "new", "Solve Session", "--id", "solve-default-auto"])
        manager = SessionManager(data_dir=temp_data_dir)

        problem = ProblemFrame(
            description="Build a practical execution plan for reducing churn",
            target_domain="general",
        )
        manager.save_stage_data("solve-default-auto", "problem", problem.model_dump())
        manager.save_stage_data(
            "solve-default-auto",
            "verify",
            {
                "hypotheses": [
                    {
                        "id": "hyp_001",
                        "description": "test",
                        "source_domain": "queueing_theory",
                        "source_structure": "priority_queue",
                        "analogy_explanation": "Map queue",
                        "observable": "ticket_age_weighted_resolution_time",
                    }
                ]
            },
        )
        (temp_data_dir / "solve-default-auto" / "verify.md").write_text(
            "# Verification Results\n\nIdea.\n",
            encoding="utf-8",
        )

        manager.update_stage_status("solve-default-auto", "problem", StageStatus.COMPLETED)
        manager.update_stage_status("solve-default-auto", "biso", StageStatus.COMPLETED)
        manager.update_stage_status("solve-default-auto", "search", StageStatus.COMPLETED)
        manager.update_stage_status("solve-default-auto", "verify", StageStatus.COMPLETED)

        captured: dict[str, bool] = {}

        class DummySolver:
            def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
                pass

            async def run(
                self,
                problem,
                verify_markdown,
                hypotheses,
                max_ideas,
                max_web_results,
                auto_refine=False,
            ):
                captured["auto_refine"] = bool(auto_refine)
                return {
                    "solution_markdown": "# Final Solution\n\nUse staged rollout.\n",
                    "applied_constraints": [],
                    "evidence": [],
                    "metrics": {"idea_count": 1},
                    "audit_report": {"events_processed": 1},
                }

        monkeypatch.setattr("x_creative.cli.run.SAGASolver", DummySolver)

        result = runner.invoke(app, ["run", "solve", "--session", "solve-default-auto"])
        assert result.exit_code == 0
        assert captured.get("auto_refine") is True


class TestRunVerify:
    def test_run_verify_uses_dual_verify_path(self, runner, temp_data_dir, monkeypatch):
        from x_creative.cli.main import app
        from x_creative.core.types import ProblemFrame
        from x_creative.session.manager import SessionManager
        from x_creative.session.types import StageStatus

        runner.invoke(app, ["session", "new", "Verify Session", "--id", "verify-dual"])
        manager = SessionManager(data_dir=temp_data_dir)

        problem = ProblemFrame(
            description="Find robust open source project hypotheses",
            target_domain="general",
        )
        manager.save_stage_data("verify-dual", "problem", problem.model_dump())
        manager.save_stage_data(
            "verify-dual",
            "search",
            {
                "hypotheses": [
                    {
                        "id": "h_low_final",
                        "description": "low final",
                        "source_domain": "d",
                        "source_structure": "s",
                        "analogy_explanation": "a",
                        "observable": "o",
                        "scores": {
                            "divergence": 9.0,
                            "testability": 9.0,
                            "rationale": 9.0,
                            "robustness": 9.0,
                            "feasibility": 9.0,
                        },
                    },
                    {
                        "id": "h_high_final",
                        "description": "high final",
                        "source_domain": "d",
                        "source_structure": "s",
                        "analogy_explanation": "a",
                        "observable": "o",
                        "scores": {
                            "divergence": 6.0,
                            "testability": 6.0,
                            "rationale": 6.0,
                            "robustness": 6.0,
                            "feasibility": 6.0,
                        },
                    },
                ]
            },
        )

        manager.update_stage_status("verify-dual", "problem", StageStatus.COMPLETED)
        manager.update_stage_status("verify-dual", "biso", StageStatus.COMPLETED)
        manager.update_stage_status("verify-dual", "search", StageStatus.COMPLETED)

        class DummyEngine:
            seen_problem: ProblemFrame | None = None

            async def score_and_verify_batch(self, hypotheses, problem_frame=None):  # noqa: ANN001
                DummyEngine.seen_problem = problem_frame
                return [
                    hypotheses[0].model_copy(update={"final_score": 6.1}),
                    hypotheses[1].model_copy(update={"final_score": 8.7}),
                ]

            def filter_by_threshold(self, hypotheses, threshold=5.0):  # noqa: ANN001
                return hypotheses

            def sort_by_score(self, hypotheses, descending=True):  # noqa: ANN001
                return sorted(
                    hypotheses,
                    key=lambda h: float(h.final_score or 0.0),
                    reverse=descending,
                )

            async def close(self) -> None:
                return None

        monkeypatch.setattr("x_creative.cli.run.CreativityEngine", DummyEngine)

        result = runner.invoke(
            app,
            ["run", "verify", "--session", "verify-dual", "--threshold", "0", "--top", "2"],
        )
        assert result.exit_code == 0

        verify_data = manager.load_stage_data("verify-dual", "verify")
        assert verify_data is not None
        ids = [h["id"] for h in verify_data["hypotheses"]]
        assert ids == ["h_high_final", "h_low_final"]
        assert DummyEngine.seen_problem is not None
        assert DummyEngine.seen_problem.description == "Find robust open source project hypotheses"
