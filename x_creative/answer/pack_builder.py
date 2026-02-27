"""AnswerPackBuilder - assembles the final AnswerPack from session artifacts."""

from __future__ import annotations

from typing import Any

from x_creative.answer.types import AnswerPack
from x_creative.core.plugin import TargetDomainPlugin
from x_creative.core.types import Domain, Hypothesis, ProblemFrame
from x_creative.session.types import Session


class AnswerPackBuilder:
    @classmethod
    def build(
        cls,
        session: Session,
        problem_frame: ProblemFrame,
        target_plugin: TargetDomainPlugin,
        source_domains: list[Domain],
        verified_hypotheses: list[Hypothesis],
        solve_result: dict[str, Any] | None,
        question: str = "",
        budget_used: int = 0,
        budget_total: int = 120,
        saga_directives: list[dict] | None = None,
        search_rounds: int = 0,
        cp1_result: dict[str, Any] | None = None,
        cp2_result: dict[str, Any] | None = None,
        cp3_result: dict[str, Any] | None = None,
        duration_seconds: float = 0.0,
    ) -> AnswerPack:
        q = question or problem_frame.description

        answer_md = cls._build_markdown(
            question=q,
            problem_frame=problem_frame,
            target_plugin=target_plugin,
            source_domains=source_domains,
            hypotheses=verified_hypotheses,
            solve_result=solve_result,
            budget_used=budget_used,
            budget_total=budget_total,
            saga_directives=saga_directives or [],
        )
        answer_json = cls._build_json(
            question=q,
            session=session,
            problem_frame=problem_frame,
            target_plugin=target_plugin,
            source_domains=source_domains,
            hypotheses=verified_hypotheses,
            solve_result=solve_result,
            budget_used=budget_used,
            budget_total=budget_total,
            saga_directives=saga_directives or [],
            search_rounds=search_rounds,
            cp1_result=cp1_result,
            cp2_result=cp2_result,
            cp3_result=cp3_result,
            duration_seconds=duration_seconds,
        )
        return AnswerPack(
            question=q,
            answer_md=answer_md,
            answer_json=answer_json,
            session_id=session.id,
        )

    @classmethod
    def _build_markdown(
        cls,
        question: str,
        problem_frame: ProblemFrame,
        target_plugin: TargetDomainPlugin,
        source_domains: list[Domain],
        hypotheses: list[Hypothesis],
        solve_result: dict[str, Any] | None,
        budget_used: int,
        budget_total: int,
        saga_directives: list[dict],
    ) -> str:
        lines = [f"# {question}", ""]

        lines.append("## Direct Answer")
        if solve_result and solve_result.get("solution_markdown"):
            lines.append(solve_result["solution_markdown"])
        else:
            lines.append("*No solution generated yet.*")
        lines.append("")

        lines.append("## Key Evidence")
        if hypotheses:
            for i, h in enumerate(hypotheses[:5], 1):
                score = getattr(h, "final_score", None) or ""
                score_str = (
                    f" — confidence: {score:.2f}"
                    if isinstance(score, (int, float))
                    else ""
                )
                lines.append(
                    f"{i}. **{h.description[:80]}** — Source: {h.source_domain}{score_str}"
                )
        else:
            lines.append("*No verified hypotheses.*")
        lines.append("")

        lines.append("## Risks & Boundaries")
        if problem_frame.open_questions:
            for oq in problem_frame.open_questions:
                lines.append(f"- **OPEN**: {oq}")
        else:
            lines.append("*No identified risks.*")
        lines.append("")

        if hypotheses:
            lines.append("## Hypothesis Ranking (Top 5)")
            lines.append("| # | Hypothesis | Source Domain | Score |")
            lines.append("|---|-----------|--------------|-------|")
            for i, h in enumerate(hypotheses[:5], 1):
                score = getattr(h, "final_score", None)
                score_str = (
                    f"{score:.2f}" if isinstance(score, (int, float)) else "N/A"
                )
                lines.append(
                    f"| {i} | {h.description[:50]} | {h.source_domain} | {score_str} |"
                )
            lines.append("")

        lines.append("## Methodology Appendix")
        lines.append(
            f"- Problem frame: {problem_frame.objective or problem_frame.description[:60]}"
        )
        lines.append(f"- Target domain: {target_plugin.id}")
        lines.append(f"- Source domains: {len(source_domains)}")
        lines.append(f"- Total hypotheses: {len(hypotheses)}")
        lines.append(f"- Budget used: {budget_used}/{budget_total} units")
        lines.append(f"- SAGA directives: {len(saga_directives)}")
        lines.append("")

        return "\n".join(lines)

    @classmethod
    def _build_json(
        cls,
        question: str,
        session: Session,
        problem_frame: ProblemFrame,
        target_plugin: TargetDomainPlugin,
        source_domains: list[Domain],
        hypotheses: list[Hypothesis],
        solve_result: dict[str, Any] | None,
        budget_used: int,
        budget_total: int,
        saga_directives: list[dict],
        search_rounds: int,
        cp1_result: dict[str, Any] | None,
        cp2_result: dict[str, Any] | None,
        cp3_result: dict[str, Any] | None,
        duration_seconds: float,
    ) -> dict[str, Any]:
        summary = ""
        if solve_result and solve_result.get("solution_markdown"):
            summary = solve_result["solution_markdown"][:500]

        hyp_data = []
        for h in hypotheses:
            entry: dict[str, Any] = {
                "id": h.id,
                "description": h.description,
                "source_domain": h.source_domain,
                "observable": h.observable,
            }
            if hasattr(h, "final_score") and h.final_score is not None:
                entry["final_score"] = h.final_score
            hyp_data.append(entry)

        domain_confidence = 1.0
        if problem_frame.domain_hint:
            domain_confidence = problem_frame.domain_hint.get("confidence", 1.0)

        return {
            "version": "1.0",
            "question": question,
            "answer": {
                "summary": summary,
                "confidence": domain_confidence,
                "evidence": [],
            },
            "risks": [
                {"severity": "medium", "description": oq, "mitigation": ""}
                for oq in (problem_frame.open_questions or [])
            ],
            "hypotheses": hyp_data,
            "metadata": {
                "session_id": session.id,
                "target_domain": target_plugin.id,
                "target_confidence": domain_confidence,
                "source_domains_count": len(source_domains),
                "total_hypotheses_generated": len(hypotheses),
                "search_rounds": search_rounds,
                "budget_used": budget_used,
                "budget_total": budget_total,
                "saga_directives": saga_directives,
                "checkpoints": {
                    "cp1": cp1_result or {},
                    "cp2": cp2_result or {},
                    "cp3": cp3_result or {},
                },
                "duration_seconds": duration_seconds,
            },
        }
