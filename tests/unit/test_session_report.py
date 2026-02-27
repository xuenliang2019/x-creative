"""Tests for report generation."""

import pytest


class TestReportGenerator:
    """Tests for ReportGenerator class."""

    def test_generate_problem_report(self):
        from x_creative.session.report import ReportGenerator
        from x_creative.core.types import ProblemFrame

        problem = ProblemFrame(
            description="设计一个能实现病毒式传播的开源命令行工具",
            target_domain="open_source_development",
            constraints=["必须跨平台", "安装步骤不超过2步"],
        )

        report = ReportGenerator.problem_report(problem)

        assert "# Problem Definition" in report
        assert "设计一个能实现病毒式传播的开源命令行工具" in report
        assert "Target Domain" in report
        assert "open_source_development" in report
        assert "必须跨平台" in report

    def test_generate_problem_report_with_context(self):
        from x_creative.session.report import ReportGenerator
        from x_creative.core.types import ProblemFrame

        problem = ProblemFrame(
            description="测试问题",
            target_domain="general",
            context={"key": "value"},
        )

        report = ReportGenerator.problem_report(problem)

        assert "Context:" in report
        assert '"key": "value"' in report

    def test_generate_problem_report_with_constraints(self):
        from x_creative.session.report import ReportGenerator
        from x_creative.core.types import ProblemFrame

        problem = ProblemFrame(
            description="How to design a viral open source tool?",
            target_domain="open_source_development",
            constraints=["Must be cross-platform", "Easy to install"],
        )

        report = ReportGenerator.problem_report(problem)

        assert "How to design a viral open source tool?" in report
        assert "Must be cross-platform" in report
        assert "Easy to install" in report

    def test_generate_biso_report(self):
        from x_creative.session.report import ReportGenerator
        from x_creative.core.types import Hypothesis

        hypotheses = [
            Hypothesis(
                id="h1",
                description="Test hypothesis",
                source_domain="thermodynamics",
                source_structure="entropy",
                analogy_explanation="Heat flow analogy",
                observable="some_metric",
            )
        ]

        report = ReportGenerator.biso_report(hypotheses)

        assert "# BISO Results" in report
        assert "Test hypothesis" in report
        assert "thermodynamics" in report

    def test_generate_verify_report(self):
        from x_creative.session.report import ReportGenerator
        from x_creative.core.types import Hypothesis, HypothesisScores

        hypotheses = [
            Hypothesis(
                id="h1",
                description="High score hypothesis",
                source_domain="thermodynamics",
                source_structure="entropy",
                analogy_explanation="Analogy",
                observable="metric",
                scores=HypothesisScores(
                    divergence=8.0,
                    testability=9.0,
                    rationale=7.5,
                    robustness=8.0,
                    feasibility=7.0,
                ),
            )
        ]

        report = ReportGenerator.verify_report(hypotheses)

        assert "# Verification Results" in report
        assert "High score hypothesis" in report
        assert "8.0" in report or "8.1" in report  # Score appears
        assert "Feasibility" in report  # New dimension appears
