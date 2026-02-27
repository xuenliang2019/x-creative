"""Markdown report generation for session stages."""

import json
from datetime import datetime

from x_creative.core.types import Hypothesis, ProblemFrame


class ReportGenerator:
    """Generate Markdown reports for each pipeline stage."""

    @staticmethod
    def problem_report(problem: ProblemFrame) -> str:
        """Generate a report for the problem definition stage."""
        lines = [
            "# Problem Definition",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Description",
            "",
            problem.description,
            "",
            "## Parameters",
            "",
            f"- **Target Domain:** {problem.target_domain}",
        ]

        # Show context if present
        if problem.context:
            lines.append(f"- **Context:** `{json.dumps(problem.context, ensure_ascii=False)}`")

        lines.append("")

        # Show constraints if present
        if problem.constraints:
            lines.extend([
                "## Constraints",
                "",
                *[f"- {c}" for c in problem.constraints],
                "",
            ])

        return "\n".join(lines)

    @staticmethod
    def biso_report(hypotheses: list[Hypothesis]) -> str:
        """Generate a report for the BISO stage."""
        lines = [
            "# BISO Results",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Total Hypotheses:** {len(hypotheses)}",
            "",
        ]

        # Group by domain
        by_domain: dict[str, list[Hypothesis]] = {}
        for h in hypotheses:
            by_domain.setdefault(h.source_domain, []).append(h)

        for domain, domain_hypotheses in by_domain.items():
            lines.extend([
                f"## {domain}",
                "",
            ])
            for i, h in enumerate(domain_hypotheses, 1):
                lines.extend([
                    f"### {i}. {h.description[:80]}{'...' if len(h.description) > 80 else ''}",
                    "",
                    f"**ID:** `{h.id}`",
                    f"**Structure:** {h.source_structure}",
                    "",
                    "**Analogy:**",
                    "",
                    h.analogy_explanation,
                    "",
                    "**Observable:**",
                    "",
                    "```",
                    h.observable,
                    "```",
                    "",
                ])

        return "\n".join(lines)

    @staticmethod
    def search_report(hypotheses: list[Hypothesis]) -> str:
        """Generate a report for the SEARCH stage."""
        lines = [
            "# Search Results",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Total Hypotheses:** {len(hypotheses)}",
            "",
        ]

        # Group by generation
        by_gen: dict[int, list[Hypothesis]] = {}
        for h in hypotheses:
            by_gen.setdefault(h.generation, []).append(h)

        for gen in sorted(by_gen.keys()):
            gen_hypotheses = by_gen[gen]
            lines.extend([
                f"## Generation {gen}",
                "",
                f"*{len(gen_hypotheses)} hypotheses*",
                "",
            ])
            for h in gen_hypotheses[:10]:  # Limit display
                exp_type = f" ({h.expansion_type})" if h.expansion_type else ""
                lines.extend([
                    f"- **{h.description[:60]}**{exp_type}",
                    f"  - Domain: {h.source_domain}/{h.source_structure}",
                    "",
                ])

            if len(gen_hypotheses) > 10:
                lines.append(f"*... and {len(gen_hypotheses) - 10} more*\n")

        return "\n".join(lines)

    @staticmethod
    def verify_report(hypotheses: list[Hypothesis], top_n: int = 50) -> str:
        """Generate a report for the VERIFY stage."""
        lines = [
            "# Verification Results",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Total Verified:** {len(hypotheses)}",
            "",
            "## Summary Statistics",
            "",
        ]

        if hypotheses:
            scores = [h.composite_score() for h in hypotheses]
            avg_score = sum(scores) / len(scores)
            lines.extend([
                f"- **Average Score:** {avg_score:.2f}",
                f"- **Highest Score:** {max(scores):.2f}",
                f"- **Lowest Score:** {min(scores):.2f}",
                "",
            ])

        lines.extend([
            f"## Top {min(top_n, len(hypotheses))} Hypotheses",
            "",
        ])

        for i, h in enumerate(hypotheses[:top_n], 1):
            score = h.composite_score()
            lines.extend([
                f"### {i}. {h.description[:80]}",
                "",
                f"**Score:** {score:.1f}",
                f"**Source:** {h.source_domain}/{h.source_structure}",
                "",
            ])

            if h.scores:
                lines.extend([
                    "| Dimension | Score |",
                    "|-----------|-------|",
                    f"| Divergence | {h.scores.divergence:.1f} |",
                    f"| Testability | {h.scores.testability:.1f} |",
                    f"| Rationale | {h.scores.rationale:.1f} |",
                    f"| Robustness | {h.scores.robustness:.1f} |",
                    f"| Feasibility | {h.scores.feasibility:.1f} |",
                    "",
                ])

            lines.extend([
                "**Analogy:**",
                "",
                h.analogy_explanation[:300] + ("..." if len(h.analogy_explanation) > 300 else ""),
                "",
                "**Observable:**",
                "",
                "```",
                h.observable,
                "```",
                "",
                "---",
                "",
            ])

        return "\n".join(lines)
