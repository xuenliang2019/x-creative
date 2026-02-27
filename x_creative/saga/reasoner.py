"""Reasoner (System 2) — multi-step reasoning agent for Talker-Reasoner architecture.

Executes a 7-step Think→Act→Observe→Update cycle to build a structured
BeliefState that the Talker uses to generate the final solution.
"""

import asyncio
import json
import re
import time
from collections.abc import Awaitable, Callable
from typing import Any

import structlog

from x_creative.config.settings import (
    SearchConfig as NoveltySearchConfig,
    SearchRoundConfig,
    get_settings,
)
from x_creative.core.types import ConstraintSpec, Hypothesis, ProblemFrame
from x_creative.llm.router import ModelRouter
from x_creative.saga.belief import (
    BeliefState,
    CrossValidation,
    EvidenceItem,
    HypothesisVerdict,
    ProblemAnalysis,
    QualityAssessment,
    ReasoningPhase,
    ReasoningStep,
    RefinementRound,
    SolutionBlueprint,
    UserClarification,
    UserQuestion,
)
from x_creative.verify.search import SearchValidator

logger = structlog.get_logger()

ProgressCallback = Callable[[str, dict[str, Any]], Awaitable[None] | None]
UserQuestionCallback = Callable[[UserQuestion], Awaitable[str]]


class ReasonerFatalError(Exception):
    """Fatal error in Reasoner that should not be silently swallowed."""


class QualityAuditRejected(ReasonerFatalError):
    """Raised when the user rejects high-risk items during quality audit."""


def _extract_json_object(text: str) -> dict[str, Any]:
    """Extract the first JSON object from text."""
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        return {}
    return {}


def _extract_json_array(text: str) -> list[Any]:
    """Extract the first JSON array from text."""
    match = re.search(r"\[[\s\S]*\]", text)
    if not match:
        return []
    try:
        parsed = json.loads(match.group(0))
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        return []
    return []


class Reasoner:
    """Multi-step reasoning agent (System 2).

    Builds a structured BeliefState through 7 reasoning phases:
    1. Problem Analysis
    2. Hypothesis Evaluation
    3. Evidence Gathering (web search)
    4. Cross Validation
    5. Solution Planning
    6. Quality Audit (cross-model adversarial)
    7. Belief Synthesis (programmatic)
    """

    def __init__(
        self,
        router: ModelRouter,
        progress_callback: ProgressCallback | None = None,
        user_callback: UserQuestionCallback | None = None,
        max_web_results: int = 8,
    ) -> None:
        self._router = router
        self._progress_callback = progress_callback
        self._user_callback = user_callback
        self._max_web_results = max_web_results
        try:
            self._max_constraints = max(1, int(get_settings().max_constraints))
        except Exception:
            self._max_constraints = 15

    @staticmethod
    def _hypothesis_rank_score(hypothesis: Hypothesis) -> float:
        """Rank score preferring verified final_score over composite score."""
        if hypothesis.final_score is not None:
            return float(hypothesis.final_score)
        return float(hypothesis.composite_score())

    @staticmethod
    def _normalize_constraint_text(text: str) -> str:
        """Normalize constraints for lightweight semantic deduplication."""
        lowered = text.lower()
        lowered = lowered.replace("-", "").replace("_", "").replace(" ", "")
        lowered = lowered.replace("（", "(").replace("）", ")")
        lowered = lowered.replace("，", ",").replace("。", ".")
        # Collapse common format variants.
        lowered = lowered.replace("ticklevel", "ticklevel")
        return lowered

    async def reason(
        self,
        problem: ProblemFrame,
        verify_markdown: str,
        hypotheses: list[Hypothesis],
        max_ideas: int = 8,
        inner_max: int = 1,
    ) -> BeliefState:
        """Execute the full reasoning chain and return the built belief state.

        Args:
            inner_max: Max inner-loop iterations for Step 5↔6 refinement.
                       1 = no refinement (original behavior).
        """
        belief = BeliefState()

        # Select top hypotheses
        sorted_hyps = sorted(
            hypotheses,
            key=self._hypothesis_rank_score,
            reverse=True,
        )
        selected = sorted_hyps[: max(1, max_ideas)]

        # Steps 1-4: execute once
        fixed_steps = [
            (ReasoningPhase.PROBLEM_ANALYSIS, self._step_problem_analysis),
            (ReasoningPhase.HYPOTHESIS_EVALUATION, self._step_hypothesis_evaluation),
            (ReasoningPhase.EVIDENCE_GATHERING, self._step_evidence_gathering),
            (ReasoningPhase.CROSS_VALIDATION, self._step_cross_validation),
        ]

        step_number = 0
        for phase, step_fn in fixed_steps:
            step_number += 1
            await self._execute_and_record_step(
                belief, phase, step_fn, step_number, 7,
                problem=problem, verify_markdown=verify_markdown, selected=selected,
            )

        # Steps 5-6: inner refinement loop
        prior_risks: list[dict[str, str]] = []
        for inner_round in range(1, inner_max + 1):
            high_risks_before = len(prior_risks)

            if inner_max > 1:
                await self._report_progress(
                    "refine_inner_round",
                    {
                        "round": inner_round,
                        "max_rounds": inner_max,
                        "high_risks_before": high_risks_before,
                    },
                )

            # Step 5: Solution Planning (with prior risks injected)
            step_number += 1
            await self._execute_and_record_step(
                belief, ReasoningPhase.SOLUTION_PLANNING,
                self._step_solution_planning, step_number, 7,
                problem=problem, verify_markdown=verify_markdown,
                selected=selected, prior_risks=prior_risks,
            )

            # Step 6: Quality Audit (non-interactive during refinement)
            step_number += 1
            saved_callback = self._user_callback
            if inner_max > 1:
                self._user_callback = None  # disable interactive during refinement
            try:
                await self._execute_and_record_step(
                    belief, ReasoningPhase.QUALITY_AUDIT,
                    self._step_quality_audit, step_number, 7,
                    problem=problem, verify_markdown=verify_markdown,
                    selected=selected,
                )
            finally:
                self._user_callback = saved_callback

            # Check convergence
            high_risks = [
                r for r in belief.quality_assessment.risks
                if r.get("severity") == "high"
            ]
            high_risks_after = len(high_risks)

            # Record refinement round
            if inner_max > 1:
                belief.refinement_trace.inner_rounds.append(
                    RefinementRound(
                        round_number=inner_round,
                        high_risks_before=high_risks_before,
                        high_risks_after=high_risks_after,
                        risks_addressed=prior_risks,
                    )
                )

            if not high_risks:
                if inner_max > 1:
                    belief.refinement_trace.converged = True
                break

            # Feed risks into next round
            prior_risks = high_risks

        belief.refinement_trace.final_high_risk_count = len(
            [r for r in belief.quality_assessment.risks if r.get("severity") == "high"]
        )

        # Step 7: Belief Synthesis
        step_number += 1
        await self._execute_and_record_step(
            belief, ReasoningPhase.BELIEF_SYNTHESIS,
            self._step_belief_synthesis, step_number, 7,
            problem=problem, verify_markdown=verify_markdown, selected=selected,
        )

        return belief

    async def _execute_and_record_step(
        self,
        belief: BeliefState,
        phase: ReasoningPhase,
        step_fn: Any,
        step_number: int,
        total_steps: int,
        **kwargs: Any,
    ) -> None:
        """Execute a reasoning step, record timing/usage, and report progress."""
        step_started = time.perf_counter()
        step_llm_calls = belief.total_llm_calls
        step_tokens = belief.total_tokens_used

        try:
            await step_fn(belief=belief, **kwargs)
        except ReasonerFatalError:
            raise
        except Exception as exc:
            logger.warning(
                "Reasoner step failed, continuing",
                step=step_number,
                phase=phase.value,
                error=str(exc),
            )

        elapsed = round(time.perf_counter() - step_started, 3)
        reasoning_step = ReasoningStep(
            step_number=step_number,
            phase=phase,
            thought=f"Step {step_number}: {phase.value}",
            action=f"Execute {phase.value}",
            observation=self._summarize_step(belief, phase),
            belief_update=f"Updated belief.{phase.value}",
            llm_calls=belief.total_llm_calls - step_llm_calls,
            tokens_used=belief.total_tokens_used - step_tokens,
            elapsed_seconds=elapsed,
        )
        belief.reasoning_steps.append(reasoning_step)

        await self._report_progress(
            "reasoner_step",
            {
                "step": step_number,
                "total_steps": total_steps,
                "phase": phase.value,
                "elapsed_seconds": elapsed,
                "llm_calls": reasoning_step.llm_calls,
                "tokens_used": reasoning_step.tokens_used,
                "summary": reasoning_step.observation,
            },
        )

    # ------------------------------------------------------------------
    # Step 1: Problem Analysis
    # ------------------------------------------------------------------

    async def _step_problem_analysis(
        self,
        belief: BeliefState,
        problem: ProblemFrame,
        verify_markdown: str,
        selected: list[Hypothesis],  # noqa: ARG002
    ) -> None:
        messages = [
            {
                "role": "system",
                "content": (
                    "你是一个深度分析师。分析给定问题，输出严格 JSON（无其他文字）。\n"
                    "JSON 格式:\n"
                    "{\n"
                    '  "core_challenge": "核心挑战的一句话描述",\n'
                    '  "sub_problems": ["子问题1", "子问题2", ...],\n'
                    '  "success_criteria": ["成功标准1", ...],\n'
                    '  "implicit_constraints": ["隐含约束1", ...],\n'
                    '  "domain_context": "目标领域的关键背景"\n'
                    "}"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"问题描述:\n{problem.description}\n\n"
                    f"目标领域: {problem.target_domain}\n"
                    f"约束条件: {', '.join(problem.constraints) if problem.constraints else '无'}\n\n"
                    f"verify.md 摘要 (前3000字):\n{verify_markdown[:3000]}"
                ),
            },
        ]

        result = await self._router.complete(
            task="reasoner_step",
            messages=messages,
            temperature=0.3,
            max_tokens=4096,
        )
        belief.total_llm_calls += 1
        belief.total_tokens_used += result.prompt_tokens + result.completion_tokens

        data = _extract_json_object(result.content)
        belief.problem_analysis = ProblemAnalysis(
            core_challenge=str(data.get("core_challenge", "")),
            sub_problems=[str(s) for s in data.get("sub_problems", [])],
            success_criteria=[str(s) for s in data.get("success_criteria", [])],
            implicit_constraints=[str(s) for s in data.get("implicit_constraints", [])],
            domain_context=str(data.get("domain_context", "")),
        )

    # ------------------------------------------------------------------
    # Step 2: Hypothesis Evaluation
    # ------------------------------------------------------------------

    async def _step_hypothesis_evaluation(
        self,
        belief: BeliefState,
        problem: ProblemFrame,  # noqa: ARG002
        verify_markdown: str,  # noqa: ARG002
        selected: list[Hypothesis],
    ) -> None:
        hyp_summaries = []
        for h in selected:
            hyp_summaries.append(
                {
                    "id": h.id,
                    "description": h.description,
                    "source_domain": h.source_domain,
                    "source_structure": h.source_structure,
                    "score": round(h.composite_score(), 3),
                }
            )

        messages = [
            {
                "role": "system",
                "content": (
                    "你是一个研究假说评估专家。逐个评估以下假说，输出严格 JSON 数组（无其他文字）。\n"
                    "每个元素格式:\n"
                    "{\n"
                    '  "hypothesis_id": "H-xxx",\n'
                    '  "description": "假说描述",\n'
                    '  "source_domain": "来源领域",\n'
                    '  "relevance": "与问题的相关性分析",\n'
                    '  "strength": "核心优势",\n'
                    '  "weakness": "主要弱点",\n'
                    '  "actionability": "可执行性评估",\n'
                    '  "priority": 1\n'
                    "}\n"
                    "priority: 1=最高优先级, 数字越大优先级越低。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"问题核心挑战: {belief.problem_analysis.core_challenge}\n"
                    f"子问题: {json.dumps(belief.problem_analysis.sub_problems, ensure_ascii=False)}\n\n"
                    f"候选假说:\n{json.dumps(hyp_summaries, ensure_ascii=False, indent=2)}"
                ),
            },
        ]

        result = await self._router.complete(
            task="reasoner_step",
            messages=messages,
            temperature=0.3,
            max_tokens=4096,
        )
        belief.total_llm_calls += 1
        belief.total_tokens_used += result.prompt_tokens + result.completion_tokens

        verdicts_data = _extract_json_array(result.content)
        verdicts = []
        for item in verdicts_data:
            verdicts.append(
                HypothesisVerdict(
                    hypothesis_id=str(item.get("hypothesis_id", "")),
                    description=str(item.get("description", "")),
                    source_domain=str(item.get("source_domain", "")),
                    relevance=str(item.get("relevance", "")),
                    strength=str(item.get("strength", "")),
                    weakness=str(item.get("weakness", "")),
                    actionability=str(item.get("actionability", "")),
                    priority=int(item.get("priority", 99)),
                )
            )
        belief.hypothesis_verdicts = sorted(verdicts, key=lambda v: v.priority)

        # Interactive: ask user to confirm top priorities
        if self._user_callback and belief.hypothesis_verdicts:
            top3 = belief.hypothesis_verdicts[:3]
            options = [f"[{v.hypothesis_id}] {v.description[:60]}" for v in top3]
            question = UserQuestion(
                question="以下是 Reasoner 评估的 Top 3 高优假说，是否调整优先级？",
                context="输入编号重排（如 '2,1,3'），或直接回车接受当前排序。",
                options=options,
                default="",
            )
            response = await self._user_callback(question)
            if response.strip():
                belief.user_clarifications.append(
                    UserClarification(
                        question=question.question,
                        context=question.context,
                        response=response,
                        phase=ReasoningPhase.HYPOTHESIS_EVALUATION,
                    )
                )

    # ------------------------------------------------------------------
    # Step 3: Evidence Gathering (web search)
    # ------------------------------------------------------------------

    async def _step_evidence_gathering(
        self,
        belief: BeliefState,
        problem: ProblemFrame,  # noqa: ARG002
        verify_markdown: str,  # noqa: ARG002
        selected: list[Hypothesis],
    ) -> None:
        # Determine which hypotheses to search for — use high priority verdicts
        high_priority_ids = {
            v.hypothesis_id
            for v in belief.hypothesis_verdicts
            if v.priority <= 3
        }
        # If no priorities set, use all selected
        if not high_priority_ids:
            high_priority_ids = {h.id for h in selected}

        search_hypotheses = [h for h in selected if h.id in high_priority_ids]
        if not search_hypotheses:
            search_hypotheses = selected[:3]

        search_config = NoveltySearchConfig(
            rounds=[
                SearchRoundConfig(name="concept", weight=0.3, max_results=self._max_web_results),
                SearchRoundConfig(name="implementation", weight=0.5, max_results=self._max_web_results),
                SearchRoundConfig(name="cross_domain", weight=0.2, max_results=self._max_web_results),
            ],
            search_threshold=6.0,
        )

        total = len(search_hypotheses)
        semaphore = asyncio.Semaphore(3)

        async def _gather_one(
            i: int,
            hypothesis: Hypothesis,
            validator: SearchValidator,
        ) -> tuple[EvidenceItem, int, int]:
            """Gather evidence for one hypothesis, returning (item, llm_calls, tokens)."""
            async with semaphore:
                preliminary = hypothesis.composite_score()
                verdict = await validator.validate(hypothesis, preliminary_score=preliminary)

                refs = [
                    {
                        "title": work.title,
                        "url": work.url,
                        "similarity": work.similarity,
                        "difference_summary": work.difference_summary,
                    }
                    for work in verdict.similar_works[: self._max_web_results]
                ]

                # LLM assessment of the evidence (returns text + usage deltas)
                assessment, llm_calls, tokens = await self._assess_evidence(
                    hypothesis, verdict.novelty_analysis, refs
                )

                item = EvidenceItem(
                    evidence_id=f"E{i}",
                    hypothesis_id=hypothesis.id,
                    hypothesis_description=hypothesis.description,
                    source_domain=hypothesis.source_domain,
                    source_structure=hypothesis.source_structure,
                    preliminary_score=round(preliminary, 4),
                    novelty_score=round(verdict.score, 4),
                    novelty_analysis=verdict.novelty_analysis,
                    references=refs,
                    reasoner_assessment=assessment,
                )

                await self._report_progress(
                    "reasoner_evidence",
                    {
                        "hypothesis_id": hypothesis.id,
                        "evidence_id": item.evidence_id,
                        "references": len(refs),
                        "idea_index": i,
                        "idea_total": total,
                    },
                )

                return item, llm_calls, tokens

        async with SearchValidator(search_config=search_config) as validator:
            results = await asyncio.gather(
                *[
                    _gather_one(i, h, validator)
                    for i, h in enumerate(search_hypotheses, 1)
                ]
            )

        # Collect evidence items (preserve original order) and aggregate usage
        evidence_items: list[EvidenceItem] = []
        for item, llm_calls, tokens in results:
            evidence_items.append(item)
            belief.total_llm_calls += llm_calls
            belief.total_tokens_used += tokens

        belief.evidence = evidence_items

    async def _assess_evidence(
        self,
        hypothesis: Hypothesis,
        novelty_analysis: str,
        refs: list[dict[str, Any]],
    ) -> tuple[str, int, int]:
        """LLM-based qualitative assessment of evidence for a hypothesis.

        Returns:
            Tuple of (assessment_text, llm_calls_delta, tokens_delta).
        """
        ref_summary = "\n".join(
            f"- {r.get('title', 'N/A')} (similarity={r.get('similarity', 'N/A')})"
            for r in refs[:5]
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "你是一个证据评估专家。简要评估此假说的证据质量（2-3句话），"
                    "重点关注证据充分性、潜在偏见和实用价值。只输出评估文字，无需JSON。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"假说: {hypothesis.description}\n"
                    f"新颖性分析: {novelty_analysis[:500]}\n"
                    f"相关文献:\n{ref_summary}"
                ),
            },
        ]
        try:
            result = await self._router.complete(
                task="reasoner_step",
                messages=messages,
                temperature=0.3,
                max_tokens=512,
            )
            tokens = result.prompt_tokens + result.completion_tokens
            return (result.content or "").strip(), 1, tokens
        except Exception as exc:
            logger.warning("Evidence assessment failed", error=str(exc))
            return "", 0, 0

    # ------------------------------------------------------------------
    # Step 4: Cross Validation
    # ------------------------------------------------------------------

    async def _step_cross_validation(
        self,
        belief: BeliefState,
        problem: ProblemFrame,  # noqa: ARG002
        verify_markdown: str,  # noqa: ARG002
        selected: list[Hypothesis],  # noqa: ARG002
    ) -> None:
        evidence_summary = [
            {
                "evidence_id": e.evidence_id,
                "hypothesis_id": e.hypothesis_id,
                "description": e.hypothesis_description[:80],
                "source_domain": e.source_domain,
                "novelty_score": e.novelty_score,
            }
            for e in belief.evidence
        ]

        messages = [
            {
                "role": "system",
                "content": (
                    "你是一个交叉验证分析师。分析假说间的关系，输出严格 JSON（无其他文字）。\n"
                    "JSON 格式:\n"
                    "{\n"
                    '  "complementary_pairs": [{"a": "E1", "b": "E2", "reason": "互补原因"}],\n'
                    '  "contradictions": [{"a": "E1", "b": "E3", "reason": "矛盾点"}],\n'
                    '  "dependencies": [{"from": "E1", "to": "E2", "reason": "依赖关系"}],\n'
                    '  "synthesis": "综合分析总结"\n'
                    "}"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"问题核心: {belief.problem_analysis.core_challenge}\n\n"
                    f"证据项:\n{json.dumps(evidence_summary, ensure_ascii=False, indent=2)}"
                ),
            },
        ]

        result = await self._router.complete(
            task="reasoner_step",
            messages=messages,
            temperature=0.3,
            max_tokens=4096,
        )
        belief.total_llm_calls += 1
        belief.total_tokens_used += result.prompt_tokens + result.completion_tokens

        data = _extract_json_object(result.content)
        belief.cross_validation = CrossValidation(
            complementary_pairs=data.get("complementary_pairs", []),
            contradictions=data.get("contradictions", []),
            dependencies=data.get("dependencies", []),
            synthesis=str(data.get("synthesis", "")),
        )

    # ------------------------------------------------------------------
    # Step 5: Solution Planning
    # ------------------------------------------------------------------

    async def _step_solution_planning(
        self,
        belief: BeliefState,
        problem: ProblemFrame,
        verify_markdown: str,  # noqa: ARG002
        selected: list[Hypothesis],  # noqa: ARG002
        prior_risks: list[dict[str, str]] | None = None,
    ) -> None:
        # Build compact belief summary for the prompt
        evidence_brief = [
            {
                "id": e.evidence_id,
                "idea": e.hypothesis_description[:60],
                "domain": e.source_domain,
                "novelty": e.novelty_score,
                "assessment": e.reasoner_assessment[:100],
            }
            for e in belief.evidence
        ]

        compiled_constraints: list[ConstraintSpec] = []
        if problem.structured_constraints:
            compiled_constraints = [
                spec.model_copy() for spec in problem.structured_constraints
            ]
        elif problem.constraints:
            compiled_constraints = [
                ConstraintSpec(text=str(text), origin="user")
                for text in problem.constraints
                if str(text).strip()
            ]

        from x_creative.saga.constraint_checker import (
            compile_constraint_activation,
            format_constraint_prompt_block,
        )

        reference_texts = [
            problem.description,
            belief.problem_analysis.core_challenge,
            *[item["idea"] for item in evidence_brief if item.get("idea")],
            *[risk.get("risk", "") for risk in (prior_risks or [])],
        ]
        compiled_constraint_plan = compile_constraint_activation(
            constraints=compiled_constraints,
            reference_texts=reference_texts,
            max_constraints=self._max_constraints,
            active_soft_min=3,
            active_soft_max=6,
        )
        constraint_block = format_constraint_prompt_block(compiled_constraint_plan)
        if not constraint_block:
            constraint_block = "HardCore constraints (must satisfy):\n- none"

        # Build risk injection block for refinement rounds
        risk_injection = ""
        if prior_risks:
            risks_json = json.dumps(prior_risks, ensure_ascii=False, indent=2)
            risk_injection = (
                "\n\n⚠ 上一轮质量审查发现以下高风险项，你的修改方案必须明确解决它们：\n"
                f"{risks_json}\n"
                "请针对每个高风险项在方案中加入具体的缓解措施。保留上一轮方案中没有问题的部分。"
            )

        messages = [
            {
                "role": "system",
                "content": (
                    "你是一个战略规划师。基于已有分析设计分阶段可执行方案，输出严格 JSON（无其他文字）。\n"
                    "要求简洁：phases 不超过 3 个，每个 phase 的 actions 不超过 5 条，每条 action 不超过 40 字。\n"
                    "JSON 格式:\n"
                    "{\n"
                    '  "executive_summary": "方案概要(2-3句话)",\n'
                    '  "key_insights": ["关键洞察1", "关键洞察2", ...],\n'
                    '  "phases": [\n'
                    '    {"name": "阶段1", "objective": "目标", "actions": ["行动1", ...], '
                    '"evidence_refs": ["E1"], "duration": "预估周期", "success_metric": "指标"}\n'
                    "  ],\n"
                    '  "dependencies": ["依赖1", ...],\n'
                    '  "tools_and_resources": ["工具1", ...]\n'
                    "}"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"问题: {problem.description[:500]}\n\n"
                    f"核心挑战: {belief.problem_analysis.core_challenge}\n"
                    f"子问题: {json.dumps(belief.problem_analysis.sub_problems, ensure_ascii=False)}\n\n"
                    f"约束编译结果:\n{constraint_block}\n\n"
                    f"证据:\n{json.dumps(evidence_brief, ensure_ascii=False, indent=2)}\n\n"
                    f"交叉验证综合: {belief.cross_validation.synthesis}"
                    f"{risk_injection}"
                ),
            },
        ]

        result = await self._router.complete(
            task="reasoner_step",
            messages=messages,
            temperature=0.3,
            max_tokens=16384,
        )
        belief.total_llm_calls += 1
        belief.total_tokens_used += result.prompt_tokens + result.completion_tokens

        # Detect truncation: finish_reason == "length" means max_tokens was hit
        if result.finish_reason == "length":
            logger.warning(
                "Step 5 output truncated (hit max_tokens)",
                completion_tokens=result.completion_tokens,
                max_tokens=16384,
            )

        raw_content = result.content
        data = _extract_json_object(raw_content)
        belief.solution_blueprint = SolutionBlueprint(
            executive_summary=str(data.get("executive_summary", "")),
            key_insights=[str(s) for s in data.get("key_insights", [])],
            phases=data.get("phases", []),
            dependencies=[str(s) for s in data.get("dependencies", [])],
            tools_and_resources=[str(s) for s in data.get("tools_and_resources", [])],
        )

        if not belief.solution_blueprint.phases:
            # Show what the LLM actually returned so the user can diagnose
            preview = raw_content[:2000]
            parsed_keys = list(data.keys()) if data else []
            truncation_hint = ""
            if result.finish_reason == "length":
                truncation_hint = (
                    "\n  ⚠ 输出被截断 (finish_reason=length)，"
                    f"completion_tokens={result.completion_tokens}。"
                )
            raise ReasonerFatalError(
                "Solution Planning（Step 5）未生成有效的执行阶段。\n"
                f"  JSON 解析结果: keys={parsed_keys}{truncation_hint}\n"
                f"  LLM 原始输出（前 2000 字符）:\n{preview}"
            )

    # ------------------------------------------------------------------
    # Step 6: Quality Audit (cross-model adversarial)
    # ------------------------------------------------------------------

    async def _step_quality_audit(
        self,
        belief: BeliefState,
        problem: ProblemFrame,  # noqa: ARG002
        verify_markdown: str,  # noqa: ARG002
        selected: list[Hypothesis],  # noqa: ARG002
    ) -> None:
        # Abort if blueprint is empty — Step 5 must have failed silently
        if not belief.solution_blueprint.phases:
            raise ReasonerFatalError(
                "Solution blueprint 为空，无法进行质量审计（Step 5 可能被静默跳过）"
            )

        blueprint_json = belief.solution_blueprint.model_dump_json(indent=2)
        evidence_brief = json.dumps(
            [
                {
                    "id": e.evidence_id,
                    "description": e.hypothesis_description[:60],
                    "novelty": e.novelty_score,
                    "refs": len(e.references),
                }
                for e in belief.evidence
            ],
            ensure_ascii=False,
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "你是一个对抗性审查员。你的任务是找出方案中的问题，输出严格 JSON（无其他文字）。\n"
                    "重点检查: 幻觉风险、证据缺口、逻辑漏洞、执行风险。\n"
                    "JSON 格式:\n"
                    "{\n"
                    '  "verdict": "approve 或 revise",\n'
                    '  "risks": [{"risk": "风险描述", "severity": "high/medium/low", "mitigation": "缓解措施"}],\n'
                    '  "evidence_gaps": ["缺口1", ...],\n'
                    '  "hallucination_flags": ["可疑声明1", ...],\n'
                    '  "improvement_suggestions": ["建议1", ...]\n'
                    "}"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"方案蓝图:\n{blueprint_json[:4000]}\n\n"
                    f"证据概要:\n{evidence_brief}"
                ),
            },
        ]

        result = await self._router.complete(
            task="saga_adversarial",
            messages=messages,
            temperature=0.4,
            max_tokens=4096,
        )
        belief.total_llm_calls += 1
        belief.total_tokens_used += result.prompt_tokens + result.completion_tokens

        data = _extract_json_object(result.content)
        belief.quality_assessment = QualityAssessment(
            verdict=str(data.get("verdict", "")),
            risks=data.get("risks", []),
            evidence_gaps=[str(s) for s in data.get("evidence_gaps", [])],
            hallucination_flags=[str(s) for s in data.get("hallucination_flags", [])],
            improvement_suggestions=[str(s) for s in data.get("improvement_suggestions", [])],
        )

        # Interactive: if major risks found, ask user about risk tolerance
        high_risks = [r for r in belief.quality_assessment.risks if r.get("severity") == "high"]
        if self._user_callback and high_risks:
            risk_desc = "; ".join(r.get("risk", "")[:60] for r in high_risks[:3])
            question = UserQuestion(
                question=f"审查发现 {len(high_risks)} 个高风险项，是否接受并继续？",
                context=f"高风险: {risk_desc}",
                options=["接受风险，继续生成方案", "需要更多信息", "拒绝，终止生成"],
                default="接受风险，继续生成方案",
            )
            response = await self._user_callback(question)
            belief.user_clarifications.append(
                UserClarification(
                    question=question.question,
                    context=question.context,
                    response=response,
                    phase=ReasoningPhase.QUALITY_AUDIT,
                )
            )

            # Handle user's choice
            choice = response.strip()
            rejected = False

            if choice in ("3", "拒绝，终止生成"):
                # First prompt: option 3 = reject immediately
                rejected = True
            elif choice in ("2", "需要更多信息"):
                # Generate detailed risk analysis via LLM
                detail = await self._generate_risk_detail(
                    belief, high_risks
                )
                # Present detailed analysis and ask again
                followup = UserQuestion(
                    question="以下是详细风险分析，是否接受风险继续？",
                    context=detail,
                    options=["接受风险，继续生成方案", "拒绝，终止生成"],
                    default="接受风险，继续生成方案",
                )
                followup_response = await self._user_callback(followup)
                belief.user_clarifications.append(
                    UserClarification(
                        question=followup.question,
                        context=followup.context,
                        response=followup_response,
                        phase=ReasoningPhase.QUALITY_AUDIT,
                    )
                )
                followup_choice = followup_response.strip()
                if followup_choice in ("2", "拒绝，终止生成"):
                    rejected = True

            if rejected:
                raise QualityAuditRejected(
                    f"用户拒绝了 {len(high_risks)} 个高风险项，终止方案生成"
                )

    # ------------------------------------------------------------------
    # Step 7: Belief Synthesis (programmatic, no LLM)
    # ------------------------------------------------------------------

    async def _step_belief_synthesis(
        self,
        belief: BeliefState,
        problem: ProblemFrame,  # noqa: ARG002
        verify_markdown: str,  # noqa: ARG002
        selected: list[Hypothesis],  # noqa: ARG002
    ) -> None:
        # Compute confidence score based on available data
        scores = []

        # Evidence coverage
        if belief.evidence:
            avg_novelty = sum(e.novelty_score for e in belief.evidence) / len(belief.evidence)
            scores.append(min(avg_novelty / 10.0, 1.0))

        # Quality audit verdict
        if belief.quality_assessment.verdict == "approve":
            scores.append(0.9)
        elif belief.quality_assessment.verdict == "revise":
            scores.append(0.5)

        # Blueprint completeness
        if belief.solution_blueprint.phases:
            scores.append(min(len(belief.solution_blueprint.phases) / 3.0, 1.0))

        # Problem analysis completeness
        if belief.problem_analysis.core_challenge:
            scores.append(0.8)

        belief.confidence = round(sum(scores) / max(len(scores), 1), 3)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _generate_risk_detail(
        self,
        belief: BeliefState,
        high_risks: list[dict[str, str]],
    ) -> str:
        """Call LLM to generate a detailed analysis of high-risk items."""
        risks_json = json.dumps(high_risks, ensure_ascii=False, indent=2)
        blueprint_summary = ""
        if belief.solution_blueprint.phases:
            phase_names = [p.name for p in belief.solution_blueprint.phases]
            blueprint_summary = f"方案阶段: {', '.join(phase_names)}"

        messages = [
            {
                "role": "system",
                "content": (
                    "你是一位跨领域创新审查专家。请对以下高风险项逐一进行详细分析。\n"
                    "对每个风险项，请说明:\n"
                    "1. **风险根因**: 为什么这是一个风险\n"
                    "2. **潜在影响**: 如果不处理，会导致什么后果\n"
                    "3. **缓解建议**: 具体可操作的缓解措施\n"
                    "4. **严重程度评估**: 在当前方案上下文中的实际严重性\n\n"
                    "请用清晰的中文回答，每个风险项用分隔线隔开。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"高风险项:\n{risks_json}\n\n"
                    f"{blueprint_summary}\n\n"
                    f"方案核心挑战: {belief.problem_analysis.core_challenge}"
                ),
            },
        ]

        result = await self._router.complete(
            task="saga_adversarial",
            messages=messages,
            temperature=0.3,
            max_tokens=4096,
        )
        belief.total_llm_calls += 1
        belief.total_tokens_used += result.prompt_tokens + result.completion_tokens
        return result.content

    async def risks_to_constraints(
        self,
        risks: list[dict[str, str]],
        existing_constraints: list[str | ConstraintSpec],
    ) -> list[ConstraintSpec]:
        """Convert unresolved high-risk items into concrete, actionable constraints.

        Used by the outer refinement loop to feed risk insights back into
        the ProblemFrame before re-running verify + solve.

        Returns:
            List of new structured constraints (deduplicated against existing).
        """
        if not risks:
            return []

        risks_json = json.dumps(risks, ensure_ascii=False, indent=2)
        existing_constraint_texts = [
            item.text if isinstance(item, ConstraintSpec) else str(item)
            for item in existing_constraints
        ]
        existing_json = json.dumps(existing_constraint_texts, ensure_ascii=False)

        messages = [
            {
                "role": "system",
                "content": (
                    "你是一个约束条件设计专家。将未解决的风险项转化为具体、可验证的约束条件。\n"
                    "要求:\n"
                    "- 每条约束应该是一个明确的要求，而非风险描述\n"
                    "- 约束要具体可执行，例如'方案必须包含带外样本验证'而非'注意过拟合'\n"
                    "- 不要与已有约束重复（语义去重）\n"
                    "- 输出严格 JSON 数组（无其他文字）:\n"
                    '["约束1", "约束2", ...]'
                ),
            },
            {
                "role": "user",
                "content": (
                    f"未解决的高风险项:\n{risks_json}\n\n"
                    f"已有约束条件:\n{existing_json}"
                ),
            },
        ]

        result = await self._router.complete(
            task="reasoner_step",
            messages=messages,
            temperature=0.3,
            max_tokens=2048,
        )

        raw = _extract_json_array(result.content)
        existing_norm = {
            self._normalize_constraint_text(item)
            for item in existing_constraint_texts
        }
        new_constraints: list[ConstraintSpec] = []
        seen_norm: set[str] = set()
        for item in raw:
            if not isinstance(item, str):
                continue
            candidate = item.strip()
            if not candidate:
                continue
            normalized = self._normalize_constraint_text(candidate)
            if normalized in existing_norm or normalized in seen_norm:
                continue
            seen_norm.add(normalized)
            new_constraints.append(
                ConstraintSpec(
                    text=candidate,
                    priority="high",
                    type="soft",
                    origin="risk_refinement",
                    weight=0.7,
                )
            )

        return new_constraints

    def _summarize_step(self, belief: BeliefState, phase: ReasoningPhase) -> str:
        """Generate a short summary of what a step produced."""
        if phase == ReasoningPhase.PROBLEM_ANALYSIS:
            pa = belief.problem_analysis
            return (
                f"识别 {len(pa.sub_problems)} 个子问题, "
                f"{len(pa.implicit_constraints)} 个隐含约束"
            )
        elif phase == ReasoningPhase.HYPOTHESIS_EVALUATION:
            high = sum(1 for v in belief.hypothesis_verdicts if v.priority <= 1)
            mid = sum(1 for v in belief.hypothesis_verdicts if 1 < v.priority <= 3)
            low = sum(1 for v in belief.hypothesis_verdicts if v.priority > 3)
            return f"高优: {high}, 中优: {mid}, 低优: {low}"
        elif phase == ReasoningPhase.EVIDENCE_GATHERING:
            total_refs = sum(len(e.references) for e in belief.evidence)
            return f"{len(belief.evidence)} 条证据, {total_refs} 篇参考文献"
        elif phase == ReasoningPhase.CROSS_VALIDATION:
            cv = belief.cross_validation
            return (
                f"互补: {len(cv.complementary_pairs)}, "
                f"矛盾: {len(cv.contradictions)}, "
                f"依赖: {len(cv.dependencies)}"
            )
        elif phase == ReasoningPhase.SOLUTION_PLANNING:
            bp = belief.solution_blueprint
            return f"{len(bp.phases)} 个阶段, {len(bp.key_insights)} 个关键洞察"
        elif phase == ReasoningPhase.QUALITY_AUDIT:
            qa = belief.quality_assessment
            return f"verdict={qa.verdict}, risks={len(qa.risks)}"
        elif phase == ReasoningPhase.BELIEF_SYNTHESIS:
            return f"confidence={belief.confidence}"
        return ""

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
