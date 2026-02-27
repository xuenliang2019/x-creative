"""MappingScorer - rule + LLM hybrid scoring for structural mapping quality.

Rule layer (zero LLM cost):
- effective_rows >= 6 (anti-padding)
- duplicate_ratio <= 0.3 (anti-padding)
- row_usage_coverage >= 0.7 (anti-padding)
- mapping_type covers >= 2 types
- at least 1 systematicity_group with >= 3 items

LLM layer (called only if rules pass):
- evaluates systematicity and mapping depth
- runs mapping unpack audit on sampled rows
"""

from __future__ import annotations

import inspect
import json
import random
import re
from collections import Counter
from collections.abc import Awaitable, Callable
from typing import Any

import structlog
from pydantic import BaseModel, Field

from x_creative.core.types import Hypothesis, MappingItem
from x_creative.llm.router import ModelRouter

logger = structlog.get_logger()

MIN_EFFECTIVE_ROWS = 6
MIN_MAPPING_TYPES = 2
MIN_GROUP_SIZE = 3
MAX_DUPLICATE_RATIO = 0.3
MIN_ROW_USAGE_COVERAGE = 0.7
DUPLICATE_SIMILARITY_THRESHOLD = 0.85
RULE_FAIL_MAX_SCORE = 4.0
UNPACK_SAMPLE_SIZE = 2
UNPACK_FAIL_SCORE_CAP = 6.5
MAPPING_PADDING_SUSPECTED = "mapping_padding_suspected"


class RuleScoreResult(BaseModel):
    """Result of rule-based mapping quality check."""

    rules_passed: bool = Field(..., description="Whether all hard rules passed")
    score: float = Field(..., ge=0.0, le=10.0, description="Rule-based score")
    violations: list[str] = Field(default_factory=list, description="Hard-rule violations")
    weak_warnings: list[str] = Field(default_factory=list, description="Non-blocking warnings")
    raw_rows: int = Field(default=0, ge=0, description="Raw mapping_table row count")
    effective_rows: int = Field(default=0, ge=0, description="Effective row count")
    duplicate_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    row_usage_coverage: float = Field(default=0.0, ge=0.0, le=1.0)


class MappingUnpackAuditResult(BaseModel):
    """Result of unpack audit on sampled mapping rows."""

    sampled_row_indices: list[int] = Field(default_factory=list)
    supported_count: int = Field(default=0, ge=0)
    passed: bool = Field(default=True)
    reasoning: str = Field(default="")
    row_verdicts: list[dict[str, Any]] = Field(default_factory=list)


class MappingScoreResult(BaseModel):
    """Full mapping quality score result."""

    score: float = Field(..., ge=0.0, le=10.0, description="Final mapping quality score")
    rule_result: RuleScoreResult = Field(..., description="Rule layer result")
    llm_score: float | None = Field(default=None, description="LLM layer score if evaluated")
    llm_reasoning: str | None = Field(default=None, description="LLM reasoning")
    unpack_audit: MappingUnpackAuditResult | None = Field(
        default=None,
        description="Audit result from sampled row unpack checks",
    )
    score_cap_applied: bool = Field(default=False)
    events: list[str] = Field(default_factory=list)


class MappingScorer:
    """Hybrid rule + LLM scorer for structural mapping quality."""

    def __init__(
        self,
        router: ModelRouter | None = None,
        event_callback: Callable[[str, dict[str, Any]], Awaitable[None] | None] | None = None,
        random_seed: int | None = None,
    ) -> None:
        self._router = router
        self._event_callback = event_callback
        self._random_seed = random_seed

    def rule_score(self, hypothesis: Hypothesis) -> RuleScoreResult:
        """Evaluate mapping quality using anti-padding rule checks."""
        violations: list[str] = []
        weak_warnings: list[str] = []
        table = hypothesis.mapping_table

        raw_rows = len(table)
        effective_row_indices = self._effective_row_indices(table)
        effective_rows = len(effective_row_indices)

        if raw_rows < MIN_EFFECTIVE_ROWS:
            weak_warnings.append(
                f"mapping_table raw rows={raw_rows} is below recommended {MIN_EFFECTIVE_ROWS}"
            )

        # Hard rule 1: effective rows
        if effective_rows < MIN_EFFECTIVE_ROWS:
            violations.append(
                f"effective_rows={effective_rows}, minimum is {MIN_EFFECTIVE_ROWS}"
            )

        # Hard rule 2: relation completeness
        empty_relations = sum(
            1
            for item in table
            if not item.source_relation.strip() or not item.target_relation.strip()
        )
        if empty_relations > 0:
            violations.append(
                f"{empty_relations} rows missing source_relation or target_relation"
            )

        # Hard rule 3: duplicate ratio
        duplicate_ratio = self._duplicate_ratio(table)
        if duplicate_ratio > MAX_DUPLICATE_RATIO:
            violations.append(
                f"duplicate_ratio={duplicate_ratio:.2f} exceeds {MAX_DUPLICATE_RATIO:.2f}"
            )

        # Hard rule 4: mapping_type coverage (on effective rows)
        types_used = {
            table[idx].mapping_type
            for idx in effective_row_indices
            if idx < len(table)
        }
        if len(types_used) < MIN_MAPPING_TYPES:
            violations.append(
                f"Only {len(types_used)} effective mapping types used, minimum is {MIN_MAPPING_TYPES}"
            )

        # Hard rule 5: systematicity group size (on effective rows)
        effective_items = [table[idx] for idx in effective_row_indices if idx < len(table)]
        group_counts = Counter(item.systematicity_group_id for item in effective_items)
        max_group_size = max(group_counts.values()) if group_counts else 0
        if max_group_size < MIN_GROUP_SIZE:
            violations.append(
                f"Largest systematicity group has {max_group_size} effective rows, "
                f"minimum is {MIN_GROUP_SIZE}"
            )

        # Hard rule 6: observable_link must be present on effective rows
        missing_links = [
            idx
            for idx in effective_row_indices
            if not table[idx].observable_link or not table[idx].observable_link.strip()
        ]
        if missing_links:
            violations.append(
                f"{len(missing_links)} effective rows missing observable_link"
            )

        # Hard rule 7: row usage coverage
        row_usage_coverage = self._row_usage_coverage(hypothesis, effective_row_indices)
        if effective_rows > 0 and row_usage_coverage < MIN_ROW_USAGE_COVERAGE:
            violations.append(
                f"row_usage_coverage={row_usage_coverage:.2f} below {MIN_ROW_USAGE_COVERAGE:.2f}"
            )

        checks = [
            effective_rows >= MIN_EFFECTIVE_ROWS,
            empty_relations == 0,
            duplicate_ratio <= MAX_DUPLICATE_RATIO,
            len(types_used) >= MIN_MAPPING_TYPES,
            max_group_size >= MIN_GROUP_SIZE,
            not missing_links,
            effective_rows == 0 or row_usage_coverage >= MIN_ROW_USAGE_COVERAGE,
        ]

        rules_passed = len(violations) == 0
        if rules_passed:
            score = 5.0
        else:
            passed_rules = sum(1 for ok in checks if ok)
            score = max(0.0, (passed_rules / len(checks)) * RULE_FAIL_MAX_SCORE)

        return RuleScoreResult(
            rules_passed=rules_passed,
            score=round(score, 1),
            violations=violations,
            weak_warnings=weak_warnings,
            raw_rows=raw_rows,
            effective_rows=effective_rows,
            duplicate_ratio=round(duplicate_ratio, 3),
            row_usage_coverage=round(row_usage_coverage, 3),
        )

    async def score(self, hypothesis: Hypothesis) -> MappingScoreResult:
        """Full mapping quality scoring: rules first, then LLM if rules pass."""
        rule_result = self.rule_score(hypothesis)

        if not rule_result.rules_passed:
            return MappingScoreResult(
                score=rule_result.score,
                rule_result=rule_result,
            )

        llm_score, llm_reasoning = await self._llm_evaluate(hypothesis)
        final_score = llm_score if llm_score is not None else rule_result.score

        unpack_audit: MappingUnpackAuditResult | None = None
        score_cap_applied = False
        events: list[str] = []

        unpack_audit = await self._mapping_unpack_audit(hypothesis)
        if unpack_audit is not None and not unpack_audit.passed:
            final_score = min(final_score, UNPACK_FAIL_SCORE_CAP)
            score_cap_applied = True
            events.append(MAPPING_PADDING_SUSPECTED)
            await self._emit_event(
                MAPPING_PADDING_SUSPECTED,
                {
                    "hypothesis_id": hypothesis.id,
                    "sampled_row_indices": unpack_audit.sampled_row_indices,
                    "supported_count": unpack_audit.supported_count,
                    "sample_size": len(unpack_audit.sampled_row_indices),
                },
            )

        return MappingScoreResult(
            score=round(final_score, 1),
            rule_result=rule_result,
            llm_score=llm_score,
            llm_reasoning=llm_reasoning,
            unpack_audit=unpack_audit,
            score_cap_applied=score_cap_applied,
            events=events,
        )

    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())

    @classmethod
    def _tokenize(cls, text: str) -> set[str]:
        normalized = cls._normalize_text(text)
        return set(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]{1,}", normalized))

    @classmethod
    def _row_tokens(cls, item: MappingItem) -> set[str]:
        parts = [
            item.source_concept,
            item.target_concept,
            item.source_relation,
            item.target_relation,
            item.mapping_type,
            item.systematicity_group_id,
            item.observable_link or "",
        ]
        return cls._tokenize(" ".join(parts))

    @staticmethod
    def _jaccard_similarity(left: set[str], right: set[str]) -> float:
        if not left and not right:
            return 1.0
        if not left or not right:
            return 0.0
        return len(left & right) / len(left | right)

    @classmethod
    def _duplicate_ratio(cls, table: list[MappingItem]) -> float:
        if not table:
            return 0.0
        token_sets = [cls._row_tokens(item) for item in table]
        duplicate_count = 0
        for i, tokens in enumerate(token_sets):
            if any(
                cls._jaccard_similarity(tokens, token_sets[j]) >= DUPLICATE_SIMILARITY_THRESHOLD
                for j in range(i)
            ):
                duplicate_count += 1
        return duplicate_count / len(table)

    @classmethod
    def _effective_row_indices(cls, table: list[MappingItem]) -> list[int]:
        indices: list[int] = []
        seen_target_relations: set[str] = set()
        seen_constraint_nodes: set[str] = set()
        seen_process_nodes: set[str] = set()

        for idx, item in enumerate(table):
            source_relation = cls._normalize_text(item.source_relation)
            target_relation = cls._normalize_text(item.target_relation)
            if not source_relation or not target_relation:
                continue

            target_concept = cls._normalize_text(item.target_concept)
            adds_target_relation = target_relation not in seen_target_relations
            adds_constraint = (
                item.mapping_type == "constraint"
                and bool(target_concept)
                and target_concept not in seen_constraint_nodes
            )
            adds_process = (
                item.mapping_type == "process"
                and bool(target_concept)
                and target_concept not in seen_process_nodes
            )

            if adds_target_relation or adds_constraint or adds_process:
                indices.append(idx)

            seen_target_relations.add(target_relation)
            if item.mapping_type == "constraint" and target_concept:
                seen_constraint_nodes.add(target_concept)
            if item.mapping_type == "process" and target_concept:
                seen_process_nodes.add(target_concept)

        return indices

    @classmethod
    def _row_usage_coverage(
        cls,
        hypothesis: Hypothesis,
        effective_row_indices: list[int],
    ) -> float:
        if not effective_row_indices:
            return 0.0

        table = hypothesis.mapping_table

        downstream_texts: list[str] = [
            cls._normalize_text(hypothesis.observable),
            cls._normalize_text(hypothesis.description),
        ]

        failure_text = " ".join(
            [
                f"{fm.scenario} {fm.why_breaks} {fm.detectable_signal}"
                for fm in hypothesis.failure_modes
            ]
        )
        downstream_texts.append(cls._normalize_text(failure_text))

        coverage = hypothesis.hkg_evidence.coverage if hypothesis.hkg_evidence else {}
        if isinstance(coverage, dict):
            for key, value in coverage.items():
                if "mechanism" in key or "bridge" in key:
                    downstream_texts.append(cls._normalize_text(str(value)))

        used_rows = 0
        for idx in effective_row_indices:
            if idx >= len(table):
                continue
            row = table[idx]
            link = cls._normalize_text(row.observable_link or "")
            target_concept = cls._normalize_text(row.target_concept)
            target_relation = cls._normalize_text(row.target_relation)

            is_used = False
            if link:
                is_used = any(link and link in text for text in downstream_texts)

            if not is_used and target_concept:
                is_used = any(target_concept in text for text in downstream_texts)

            if not is_used and target_relation:
                is_used = any(target_relation in text for text in downstream_texts)

            if is_used:
                used_rows += 1

        return used_rows / len(effective_row_indices)

    async def _llm_evaluate(
        self, hypothesis: Hypothesis
    ) -> tuple[float | None, str | None]:
        """LLM evaluation of mapping systematicity and depth."""
        if self._router is None:
            return None, None

        table_text = "\n".join(
            f"- [{item.mapping_type}] {item.source_concept} → {item.target_concept} | "
            f"Source relation: {item.source_relation} | "
            f"Target relation: {item.target_relation} | "
            f"Group: {item.systematicity_group_id} | "
            f"Observable link: {item.observable_link or ''}"
            for item in hypothesis.mapping_table
        )

        prompt = f"""Evaluate the quality of this structural mapping table for a cross-domain analogy hypothesis.

## Hypothesis
{hypothesis.description}

## Observable
{hypothesis.observable}

## Mapping Table
{table_text}

## Evaluation Criteria
1. **Systematicity** (0-10): Do the mappings form coherent relation systems?
2. **Depth** (0-10): Are the mappings substantive, not superficial?

Return JSON:
{{"systematicity": <0-10>, "depth": <0-10>, "reasoning": "<explanation>"}}"""

        try:
            result = await self._router.complete(
                task="logic_verification",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            content = str(result.content).strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1] if "\n" in content else content[3:]
                if content.endswith("```"):
                    content = content[:-3]
            data = json.loads(content)
            systematicity = max(0.0, min(10.0, float(data.get("systematicity", 5.0))))
            depth = max(0.0, min(10.0, float(data.get("depth", 5.0))))
            score = (systematicity + depth) / 2.0
            reasoning = str(data.get("reasoning", ""))
            return round(score, 1), reasoning
        except Exception as e:
            logger.warning("LLM mapping evaluation failed", error=str(e))
            return None, None

    async def _mapping_unpack_audit(
        self,
        hypothesis: Hypothesis,
    ) -> MappingUnpackAuditResult | None:
        """Sample mapping rows and audit whether each row supports downstream artifacts."""
        if self._router is None:
            return None

        effective_indices = self._effective_row_indices(hypothesis.mapping_table)
        if len(effective_indices) < UNPACK_SAMPLE_SIZE:
            return None

        rng_seed = f"{hypothesis.id}:{self._random_seed}" if self._random_seed is not None else hypothesis.id
        rng = random.Random(rng_seed)
        sampled = sorted(rng.sample(effective_indices, k=UNPACK_SAMPLE_SIZE))

        sampled_rows: list[str] = []
        for idx in sampled:
            item = hypothesis.mapping_table[idx]
            sampled_rows.append(
                f"- row_index={idx}, mapping_type={item.mapping_type}, "
                f"target_relation={item.target_relation}, "
                f"observable_link={item.observable_link or ''}, "
                f"target_concept={item.target_concept}"
            )

        failure_modes_text = "\n".join(
            f"- scenario: {fm.scenario}; why_breaks: {fm.why_breaks}; signal: {fm.detectable_signal}"
            for fm in hypothesis.failure_modes
        ) or "- none"

        prompt = f"""You are auditing potential mapping-table padding.

Hypothesis:
{hypothesis.description}

Observable:
{hypothesis.observable}

Failure modes:
{failure_modes_text}

Sampled mapping rows:
{chr(10).join(sampled_rows)}

For EACH sampled row, decide whether it is concretely used by observable/mechanism/failure-mode evidence.
Return strict JSON:
{{
  "rows": [
    {{"row_index": <int>, "supported": <true|false>, "reason": "<short reason>"}}
  ],
  "overall_reasoning": "<summary>"
}}"""

        try:
            result = await self._router.complete(
                task="logic_verification",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=1024,
            )
            content = str(result.content).strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1] if "\n" in content else content[3:]
                if content.endswith("```"):
                    content = content[:-3]
            data = json.loads(content)
            rows = data.get("rows", []) if isinstance(data, dict) else []
            if not isinstance(rows, list):
                rows = []

            row_verdicts: list[dict[str, Any]] = []
            supported_count = 0
            sampled_set = set(sampled)
            for row in rows:
                if not isinstance(row, dict):
                    continue
                row_index = row.get("row_index")
                try:
                    row_index_int = int(row_index)
                except Exception:
                    continue
                if row_index_int not in sampled_set:
                    continue
                supported = bool(row.get("supported", False))
                if supported:
                    supported_count += 1
                row_verdicts.append(
                    {
                        "row_index": row_index_int,
                        "supported": supported,
                        "reason": str(row.get("reason", "")).strip(),
                    }
                )

            verdict_by_index = {v["row_index"]: v for v in row_verdicts}
            for row_index in sampled:
                if row_index not in verdict_by_index:
                    row_verdicts.append(
                        {
                            "row_index": row_index,
                            "supported": False,
                            "reason": "missing verdict",
                        }
                    )

            row_verdicts = sorted(row_verdicts, key=lambda v: int(v.get("row_index", -1)))
            supported_count = sum(1 for v in row_verdicts if v.get("supported") is True)
            passed = supported_count >= UNPACK_SAMPLE_SIZE
            reasoning = str(data.get("overall_reasoning", "")).strip() if isinstance(data, dict) else ""

            return MappingUnpackAuditResult(
                sampled_row_indices=sampled,
                supported_count=supported_count,
                passed=passed,
                reasoning=reasoning,
                row_verdicts=row_verdicts,
            )
        except Exception as exc:
            logger.warning("Mapping unpack audit failed", error=str(exc))
            return MappingUnpackAuditResult(
                sampled_row_indices=sampled,
                supported_count=0,
                passed=False,
                reasoning="audit parse failed",
                row_verdicts=[],
            )

    async def _emit_event(self, event_type: str, payload: dict[str, Any]) -> None:
        if self._event_callback is None:
            return
        try:
            maybe_awaitable = self._event_callback(event_type, payload)
            if inspect.isawaitable(maybe_awaitable):
                await maybe_awaitable
        except Exception as exc:
            logger.debug(
                "Mapping scorer event callback failed",
                event_type=event_type,
                error=str(exc),
            )
