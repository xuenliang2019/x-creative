"""Domain-specific constraint auditor for SAGA."""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from math import sqrt
import re
from typing import Any

from x_creative.core.plugin import load_target_domain
from x_creative.llm.router import ModelRouter
from x_creative.saga.events import DirectiveType, FastAgentEvent, SlowAgentDirective
from x_creative.saga.slow_agent import BaseAuditor
from x_creative.saga.state import CognitionAlert, SharedCognitionState


class DomainConstraintAuditor(BaseAuditor):
    """Audit hypotheses against critical domain constraints.

    Proactively checks ALL critical constraints from the target domain plugin,
    using LLM judgment when available, with regex fast pre-check fallback.
    """

    _LOOKAHEAD_PATTERNS = (
        # NOTE: Do NOT treat "t+1"/"T+1" as a standalone look-ahead indicator.
        # In many domains it is frequently used to describe temporal references
        # (e.g., predict next-period value), which is not a data leak by itself.
        # Over-aggressive matching here can wrongly hard-reject most hypotheses.
        # Match natural-language "future" and also snake_case observables like "future_ret".
        # Treat underscore as a boundary (unlike \b which treats '_' as a word char).
        re.compile(r"(?<![a-z0-9])future(?![a-z0-9])", re.IGNORECASE),
        re.compile(r"(?<![a-z0-9])look[\s_-]?ahead(?![a-z0-9])", re.IGNORECASE),
        re.compile(r"未来"),
    )
    _STALE_SIMILARITY_THRESHOLD = 0.28
    _CONSTRAINT_STOPWORDS = {
        "the",
        "and",
        "with",
        "for",
        "must",
        "cannot",
        "cannot",
        "using",
        "use",
        "data",
        "model",
        "constraint",
        "domain",
        "information",
        "strict",
    }

    def __init__(self, router: Any | None = None) -> None:
        self._router = router
        self._critical_cache: dict[str, list[str]] = {}

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize free text for lightweight substring checks."""
        return re.sub(r"\s+", " ", text).strip().lower()

    @staticmethod
    def _semantic_units(text: str) -> Counter[str]:
        """Build a lightweight embedding-like token vector for semantic similarity."""
        normalized = DomainConstraintAuditor._normalize_text(text)
        units: list[str] = []

        units.extend(re.findall(r"[a-z0-9_]+", normalized))

        for chunk in re.findall(r"[\u4e00-\u9fff]+", normalized):
            units.extend(list(chunk))
            units.extend(chunk[i:i + 2] for i in range(len(chunk) - 1))

        return Counter(units)

    @staticmethod
    def _cosine_similarity(counter_a: Counter[str], counter_b: Counter[str]) -> float:
        if not counter_a or not counter_b:
            return 0.0
        intersection_keys = set(counter_a) & set(counter_b)
        dot = sum(counter_a[key] * counter_b[key] for key in intersection_keys)
        mag_a = sqrt(sum(value * value for value in counter_a.values()))
        mag_b = sqrt(sum(value * value for value in counter_b.values()))
        if mag_a == 0.0 or mag_b == 0.0:
            return 0.0
        return dot / (mag_a * mag_b)

    @classmethod
    def _semantic_similarity(cls, left: str, right: str) -> float:
        left_norm = cls._normalize_text(left)
        right_norm = cls._normalize_text(right)
        if not left_norm or not right_norm:
            return 0.0
        if right_norm in left_norm or left_norm in right_norm:
            return 1.0

        left_units = cls._semantic_units(left_norm)
        right_units = cls._semantic_units(right_norm)
        cosine = cls._cosine_similarity(left_units, right_units)

        left_vocab = set(left_units)
        right_vocab = set(right_units)
        union = left_vocab | right_vocab
        jaccard = (len(left_vocab & right_vocab) / len(union)) if union else 0.0
        return max(cosine, jaccard)

    def _collect_plugin_violations(
        self,
        hypothesis_text: str,
        target_domain_id: str | None,
        target_plugin: Any = None,
    ) -> list[str]:
        """Collect anti-pattern / stale-idea violations from target plugin."""
        plugin = target_plugin
        if plugin is None:
            if not target_domain_id:
                return []
            plugin = load_target_domain(target_domain_id)
            if plugin is None:
                return []

        normalized_text = self._normalize_text(hypothesis_text)
        violations: list[str] = []

        for anti_pattern in plugin.anti_patterns:
            similarity = self._semantic_similarity(normalized_text, anti_pattern)
            if similarity >= self._STALE_SIMILARITY_THRESHOLD:
                violations.append(f"anti_pattern:{anti_pattern}")

        for stale_idea in plugin.stale_ideas:
            similarity = self._semantic_similarity(normalized_text, stale_idea)
            if similarity >= self._STALE_SIMILARITY_THRESHOLD:
                violations.append(f"stale_idea:{stale_idea}")

        return violations

    def _collect_heuristic_critical_violations(
        self,
        hypothesis_text: str,
        constraints: list,
    ) -> list[str]:
        """Heuristic critical-constraint checks used when LLM audit is unavailable."""
        normalized_text = self._normalize_text(hypothesis_text)
        violations: list[str] = []
        for constraint in constraints:
            name = str(getattr(constraint, "name", "")).strip()
            if not name:
                continue
            description = str(getattr(constraint, "description", ""))
            token_pool = f"{name.replace('_', ' ')} {description}"
            tokens = {
                tok
                for tok in re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]{2,}", token_pool.lower())
                if tok not in self._CONSTRAINT_STOPWORDS and len(tok) > 1
            }
            if not tokens:
                continue
            hits = sum(1 for token in tokens if token in normalized_text)
            if hits >= 2 or (hits >= 1 and len(tokens) <= 2):
                violations.append(f"critical:{name}")
        return violations

    async def _check_critical_constraints(
        self,
        hypothesis_text: str,
        constraints: list,
    ) -> list[str]:
        """Check hypothesis against critical constraints using LLM."""
        if not constraints:
            return []

        violations = set(
            self._collect_heuristic_critical_violations(hypothesis_text, constraints)
        )

        if not self._router:
            return sorted(violations)

        can_run_llm = True
        if isinstance(self._router, ModelRouter):
            settings = getattr(self._router, "_settings", None)
            if settings is not None:
                has_openrouter = bool(settings.openrouter.api_key.get_secret_value())
                has_yunwu = bool(settings.yunwu.api_key.get_secret_value())
                if not has_openrouter and not has_yunwu:
                    can_run_llm = False

        if can_run_llm:
            constraint_descriptions = "\n".join(
                f"- {c.name}: {c.description}" for c in constraints
            )
            prompt = (
                "Given the following hypothesis:\n"
                f"{hypothesis_text}\n\n"
                "Check if it violates ANY of these critical domain constraints:\n"
                f"{constraint_descriptions}\n\n"
                "Return a JSON array of violated constraint names (strings). "
                "Return [] if no violations detected."
            )
            try:
                completion = await asyncio.wait_for(
                    self._router.complete(
                        task="saga_deep_audit",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                        max_tokens=200,
                    ),
                    timeout=1.5,
                )
                raw = str(getattr(completion, "content", "[]"))
                raw = raw.strip()
                if raw.startswith("```"):
                    lines = raw.splitlines()
                    if lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines and lines[-1].strip() == "```":
                        lines = lines[:-1]
                    raw = "\n".join(lines).strip()
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    for item in parsed:
                        if item:
                            violations.add(f"critical:{str(item)}")
            except Exception:
                pass
        return sorted(violations)

    async def audit(
        self,
        event: FastAgentEvent,  # noqa: ARG002
        alerts: list[CognitionAlert],
        state: SharedCognitionState,
    ) -> list[SlowAgentDirective]:
        """Issue directives for hypotheses violating hard constraints.

        Runs proactively (no longer requires existing critical alerts).
        """
        if not state.hypotheses_pool:
            return []

        # Load plugin critical constraints
        critical_constraints = []
        if state.target_plugin is not None:
            critical_constraints = state.target_plugin.get_critical_constraints()
        elif state.target_domain_id:
            plugin = load_target_domain(state.target_domain_id)
            if plugin is not None:
                critical_constraints = plugin.get_critical_constraints()

        directives: list[SlowAgentDirective] = []
        for item in state.hypotheses_pool:
            hypothesis = item if isinstance(item, dict) else {}
            hypothesis_id = str(hypothesis.get("id", "")).strip()
            if not hypothesis_id:
                continue

            text = " ".join(
                str(hypothesis.get(field, ""))
                for field in ("description", "observable", "analogy_explanation")
            )
            violations: list[str] = []

            # Fast pre-check: regex patterns (zero cost)
            #
            # Only scan observable/formula-like fields for look-ahead cues.
            # Free-text description often mentions "T+1" (settlement rule) or
            # "predict next period" and should not be treated as a hard data-leak signal.
            observable_text = str(hypothesis.get("observable", ""))
            if observable_text and any(
                pattern.search(observable_text) for pattern in self._LOOKAHEAD_PATTERNS
            ):
                violations.append("critical:no_lookahead_bias")

            # Full check: all critical constraints via LLM
            if critical_constraints:
                cache_key = (
                    f"{state.target_domain_id}|{hypothesis_id}|"
                    f"{text}|{','.join(c.name for c in critical_constraints)}"
                )
                if cache_key in self._critical_cache:
                    llm_violations = self._critical_cache[cache_key]
                else:
                    llm_violations = await self._check_critical_constraints(
                        text, critical_constraints
                    )
                    self._critical_cache[cache_key] = llm_violations
                violations.extend(llm_violations)

            # Anti-pattern + stale-ideas (preserved)
            violations.extend(
                self._collect_plugin_violations(
                    hypothesis_text=text,
                    target_domain_id=state.target_domain_id,
                    target_plugin=state.target_plugin,
                )
            )

            if violations:
                violations = list(dict.fromkeys(violations))
                directives.append(
                    SlowAgentDirective(
                        directive_type=DirectiveType.FLAG_HYPOTHESIS,
                        reason="Domain constraint violation detected by DomainConstraintAuditor",
                        confidence=0.9,
                        payload={
                            "hypothesis_id": hypothesis_id,
                            "violations": violations,
                        },
                        priority=2,
                    )
                )

        return directives
