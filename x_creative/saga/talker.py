"""Talker (System 1) — belief-conditioned output generation for Talker-Reasoner architecture.

Generates a detailed, executable solution plan (4000-8000 tokens) based on
the structured BeliefState built by the Reasoner.
"""

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any

import structlog

from x_creative.core.types import ProblemFrame
from x_creative.llm.router import ModelRouter
from x_creative.saga.belief import BeliefState, EvidenceItem

logger = structlog.get_logger()

ProgressCallback = Callable[[str, dict[str, Any]], Awaitable[None] | None]


class Talker:
    """Belief-conditioned output generator (System 1).

    Takes the Reasoner's structured BeliefState and generates a detailed,
    evidence-grounded, executable solution in markdown.
    """

    def __init__(
        self,
        router: ModelRouter,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        self._router = router
        self._progress_callback = progress_callback

    async def generate(
        self,
        belief: BeliefState,
        problem: ProblemFrame,
    ) -> str:
        """Generate a detailed solution based on the belief state.

        Returns:
            Markdown solution text (4000-8000 tokens target).
        """
        await self._report_progress("talker_generating", {"phase": "start"})

        belief_payload = self._build_belief_payload(belief)
        user_constraints_block = self._format_user_constraints(problem)

        messages = [
            {
                "role": "system",
                "content": (
                    "你是一个务实的研究策略顾问。基于 Reasoner 提供的结构化信念状态，"
                    "生成详细的可执行方案。\n\n"
                    "输出要求:\n"
                    "- 使用中文\n"
                    "- 结构: 问题重述, 核心洞察, 可执行方案(分阶段详细步骤+具体指标+工具推荐), "
                    "风险矩阵, 验证里程碑, References\n"
                    "- 所有主张必须引用 [E#] 证据编号\n"
                    "- References 段列出每个 [E#] 对应的 URL\n"
                    "- 避免捏造事实；证据不足时明确标注 '证据不足'\n"
                    "- 目标长度: 4000-8000 tokens，要详细、具体、可操作\n"
                    "- 必须满足用户硬约束（见下方 C# 列表）；如无法满足必须明确标注并解释原因\n"
                    "- 如果信念状态中包含 user_clarifications（特别是带有 context 的详细风险分析），"
                    "必须在方案中针对这些风险逐一提出具体缓解措施，并在风险矩阵中标注\n"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"问题描述:\n{problem.description}\n\n"
                    f"目标领域: {problem.target_domain}\n\n"
                    f"{user_constraints_block}\n\n"
                    f"Reasoner 信念状态:\n{belief_payload}"
                ),
            },
        ]

        try:
            result = await self._router.complete(
                task="talker_output",
                messages=messages,
                temperature=0.2,
                max_tokens=8192,
            )
            belief.total_llm_calls += 1
            belief.total_tokens_used += result.prompt_tokens + result.completion_tokens

            text = (result.content or "").strip()
            if text:
                await self._report_progress(
                    "talker_generating",
                    {
                        "phase": "done",
                        "tokens": result.completion_tokens,
                        "citations": text.count("[E"),
                    },
                )
                return text
        except Exception as exc:
            logger.warning("Talker generation failed, using fallback", error=str(exc))

        return self._fallback_solution(problem, belief)

    @staticmethod
    def _format_user_constraints(problem: ProblemFrame) -> str:
        """Format user constraints into a stable C# list for prompts."""
        texts: list[str] = []
        seen: set[str] = set()

        for spec in getattr(problem, "structured_constraints", []) or []:
            if getattr(spec, "origin", "user") != "user":
                continue
            text = str(getattr(spec, "text", "")).strip()
            if not text:
                continue
            key = " ".join(text.split()).lower()
            if key in seen:
                continue
            seen.add(key)
            texts.append(" ".join(text.split()))

        if not texts:
            for raw in getattr(problem, "constraints", []) or []:
                text = str(raw).strip()
                if not text:
                    continue
                key = " ".join(text.split()).lower()
                if key in seen:
                    continue
                seen.add(key)
                texts.append(" ".join(text.split()))

        if not texts:
            return "用户硬约束 (C#，必须满足):\n- none"

        lines = ["用户硬约束 (C#，必须满足):"]
        for i, text in enumerate(texts, start=1):
            lines.append(f"- C{i}: {text}")
        return "\n".join(lines)

    def _build_belief_payload(self, belief: BeliefState) -> str:
        """Build a compact JSON representation of the belief state for the prompt."""
        payload: dict[str, Any] = {}

        # Problem analysis
        pa = belief.problem_analysis
        payload["problem_analysis"] = {
            "core_challenge": pa.core_challenge,
            "sub_problems": pa.sub_problems,
            "success_criteria": pa.success_criteria,
            "implicit_constraints": pa.implicit_constraints,
            "domain_context": pa.domain_context,
        }

        # Hypothesis verdicts (top ones)
        payload["hypothesis_verdicts"] = [
            {
                "hypothesis_id": v.hypothesis_id,
                "description": v.description,
                "source_domain": v.source_domain,
                "relevance": v.relevance,
                "strength": v.strength,
                "weakness": v.weakness,
                "actionability": v.actionability,
                "priority": v.priority,
            }
            for v in belief.hypothesis_verdicts[:8]
        ]

        # Evidence
        payload["evidence"] = [
            {
                "evidence_id": e.evidence_id,
                "hypothesis_description": e.hypothesis_description,
                "source_domain": e.source_domain,
                "novelty_score": e.novelty_score,
                "novelty_analysis": e.novelty_analysis[:300],
                "reasoner_assessment": e.reasoner_assessment,
                "references": e.references[:5],
            }
            for e in belief.evidence
        ]

        # Cross validation
        payload["cross_validation"] = belief.cross_validation.model_dump()

        # Solution blueprint
        payload["solution_blueprint"] = belief.solution_blueprint.model_dump()

        # Quality assessment
        payload["quality_assessment"] = belief.quality_assessment.model_dump()

        # User clarifications (include context for risk detail analysis)
        if belief.user_clarifications:
            payload["user_clarifications"] = [
                {
                    "question": c.question,
                    "response": c.response,
                    **({"context": c.context} if c.context else {}),
                }
                for c in belief.user_clarifications
            ]

        return json.dumps(payload, ensure_ascii=False, indent=2)

    def _fallback_solution(self, problem: ProblemFrame, belief: BeliefState) -> str:
        """Generate a template solution when LLM fails."""
        lines = [
            "# 可执行方案",
            "",
            "## 问题重述",
            "",
            problem.description,
            "",
            "## 核心洞察",
            "",
        ]

        for insight in belief.solution_blueprint.key_insights[:5]:
            lines.append(f"- {insight}")
        if not belief.solution_blueprint.key_insights:
            lines.append("- （Reasoner 未能生成关键洞察）")

        lines.extend(["", "## 可执行方案", ""])

        for e in belief.evidence:
            lines.append(
                f"- [{e.evidence_id}] {e.hypothesis_description} "
                f"(novelty={e.novelty_score:.2f})"
            )

        lines.extend([
            "",
            "## 风险矩阵",
            "",
        ])
        for risk in belief.quality_assessment.risks[:5]:
            lines.append(f"- [{risk.get('severity', 'N/A')}] {risk.get('risk', 'N/A')}")
        if not belief.quality_assessment.risks:
            lines.append("- 审计未发现显著风险")

        lines.extend([
            "",
            "## 验证里程碑",
            "",
            "- 第 1 周：完成基线与试点分组",
            "- 第 2-4 周：跟踪核心指标并复盘",
            "",
            "## References",
            "",
        ])
        for e in belief.evidence:
            for ref in e.references[:3]:
                url = ref.get("url", "")
                title = ref.get("title", "reference")
                if url:
                    lines.append(f"- [{e.evidence_id}] {title}: {url}")

        return "\n".join(lines)

    async def _report_progress(self, event: str, payload: dict[str, Any]) -> None:
        callback = self._progress_callback
        if callback is None:
            return
        try:
            maybe_result = callback(event, payload)
            if asyncio.iscoroutine(maybe_result):
                await maybe_result
        except Exception as exc:
            logger.warning("Progress callback failed", event=event, error=str(exc))
