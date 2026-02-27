"""Baseline evaluators for SAGA checkpoint interventions."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from x_creative.llm.router import ModelRouter
from x_creative.saga.budget import CognitiveBudget
from x_creative.saga.events import DirectiveType, FastAgentEvent, SlowAgentDirective
from x_creative.saga.memory import PatternMemory
from x_creative.saga.slow_agent import BaseEvaluator, hypothesis_rank_score
from x_creative.saga.state import SharedCognitionState


def _strip_code_fence(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return text


class AdversarialChallengeEvaluator(BaseEvaluator):
    """Generate adversarial challenges for top-ranked hypotheses."""

    _CHALLENGE_TEMPLATES: dict[str, str] = {
        "counterexample": "What realistic scenario would make this hypothesis fail?",
        "simplification_test": "Could this be just a rephrased simple baseline method?",
        "causal_reversal": "Could the causal direction be reversed under plausible conditions?",
        "operationalization": "Can you define a concrete, falsifiable experiment for this claim?",
    }

    def __init__(self, top_n: int = 3, router: Any | None = None) -> None:
        self._top_n = top_n
        self._router = router
        self._router_llm_enabled: bool | None = None
        self._challenge_cache: dict[str, list[dict[str, Any]]] = {}

    @classmethod
    def _normalize_challenge_type(cls, value: str | None) -> str | None:
        if not value:
            return None
        normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "simplification": "simplification_test",
            "simplify_test": "simplification_test",
            "causal_reverse": "causal_reversal",
            "causal_reversal_test": "causal_reversal",
            "operational": "operationalization",
            "operationalization_test": "operationalization",
        }
        normalized = aliases.get(normalized, normalized)
        if normalized not in cls._CHALLENGE_TEMPLATES:
            return None
        return normalized

    def _default_challenges(self, hypothesis_id: str | None) -> list[dict[str, Any]]:
        return [
            {
                "hypothesis_id": hypothesis_id,
                "challenge_type": challenge_type,
                "severity": "medium",
                "attack_question": prompt,
                "risk_summary": "Potential reward-hacking pattern to verify",
            }
            for challenge_type, prompt in self._CHALLENGE_TEMPLATES.items()
        ]

    def _parse_router_challenges(
        self,
        raw_content: str,
        hypothesis_id: str | None,
    ) -> list[dict[str, Any]]:
        """Parse router output and ensure all four theory-required challenge types exist."""
        challenges_by_type: dict[str, dict[str, Any]] = {}

        try:
            parsed = json.loads(_strip_code_fence(raw_content))
        except Exception:
            parsed = None

        items: list[dict[str, Any]]
        if isinstance(parsed, list):
            items = [item for item in parsed if isinstance(item, dict)]
        elif isinstance(parsed, dict):
            items = [parsed]
        else:
            items = []

        for item in items:
            challenge_type = self._normalize_challenge_type(item.get("challenge_type"))
            if challenge_type is None:
                continue
            payload = {
                "hypothesis_id": hypothesis_id,
                "challenge_type": challenge_type,
                "severity": str(item.get("severity", "medium")),
                "attack_question": str(
                    item.get("attack_question")
                    or self._CHALLENGE_TEMPLATES[challenge_type]
                ),
                "risk_summary": str(
                    item.get("risk_summary")
                    or "Potential unsupported assumption"
                ),
            }
            challenges_by_type[challenge_type] = payload

        for challenge_type, default_question in self._CHALLENGE_TEMPLATES.items():
            if challenge_type in challenges_by_type:
                continue
            challenges_by_type[challenge_type] = {
                "hypothesis_id": hypothesis_id,
                "challenge_type": challenge_type,
                "severity": "medium",
                "attack_question": default_question,
                "risk_summary": "Potential reward-hacking pattern to verify",
            }

        return [challenges_by_type[key] for key in self._CHALLENGE_TEMPLATES]

    async def evaluate(
        self,
        event: FastAgentEvent,  # noqa: ARG002
        state: SharedCognitionState,
        budget: CognitiveBudget,  # noqa: ARG002
    ) -> list[SlowAgentDirective]:
        if not state.hypotheses_pool:
            return []

        candidates = sorted(
            [item for item in state.hypotheses_pool if isinstance(item, dict)],
            key=hypothesis_rank_score,
            reverse=True,
        )[: self._top_n]
        if not candidates:
            return []

        directives: list[SlowAgentDirective] = []

        for target in candidates:
            hypothesis_id = target.get("id")
            cache_key = (
                f"{hypothesis_id}|{target.get('description', '')}|"
                f"{target.get('observable', '')}"
            )
            if cache_key in self._challenge_cache:
                challenge_payloads = self._challenge_cache[cache_key]
            else:
                challenge_payloads = await self._generate_challenges(target, hypothesis_id)
                self._challenge_cache[cache_key] = challenge_payloads
            directives.extend(
                SlowAgentDirective(
                    directive_type=DirectiveType.INJECT_CHALLENGE,
                    reason="Adversarial challenge generated at checkpoint",
                    confidence=0.8,
                    payload=payload,
                    priority=3,
                )
                for payload in challenge_payloads
            )

        directives.append(
            SlowAgentDirective(
                directive_type=DirectiveType.RESCORE_BATCH,
                reason="Re-score batch after adversarial challenge injection",
                confidence=0.75,
                payload={"trigger": "adversarial_challenge"},
                priority=4,
            )
        )
        return directives

    async def _generate_challenges(
        self,
        target: dict[str, Any],
        hypothesis_id: str | None,
    ) -> list[dict[str, Any]]:
        """Generate adversarial challenges for a single hypothesis."""
        if not self._router_is_available():
            return self._default_challenges(hypothesis_id)

        description = str(target.get("description", ""))[:500]
        observable = str(target.get("observable", ""))[:200]
        prompt = (
            f"Hypothesis: {description}\n"
            f"Observable: {observable}\n\n"
            "Generate exactly four adversarial challenges as JSON array. "
            "Required challenge_type values: counterexample, simplification_test, "
            "causal_reversal, operationalization. "
            "Each item must include: challenge_type, severity, attack_question, risk_summary."
        )
        try:
            completion = await asyncio.wait_for(
                self._router.complete(
                    task="saga_adversarial",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=500,
                ),
                timeout=1.5,
            )
        except Exception:
            return self._default_challenges(hypothesis_id)

        return self._parse_router_challenges(
            raw_content=str(getattr(completion, "content", "")),
            hypothesis_id=hypothesis_id,
        )

    def _router_is_available(self) -> bool:
        """Check whether router-backed LLM calls are likely usable."""
        if self._router is None:
            return False
        if self._router_llm_enabled is not None:
            return self._router_llm_enabled

        can_run = True
        if isinstance(self._router, ModelRouter):
            settings = getattr(self._router, "_settings", None)
            if settings is not None:
                has_openrouter = bool(settings.openrouter.api_key.get_secret_value())
                has_yunwu = bool(settings.yunwu.api_key.get_secret_value())
                can_run = has_openrouter or has_yunwu

        self._router_llm_enabled = can_run
        return can_run


class PatternMemoryEvaluator(BaseEvaluator):
    """Flag repeated hypothesis patterns using cross-session memory."""

    def __init__(self, memory: PatternMemory | None = None) -> None:
        self._memory = memory or PatternMemory()

    async def evaluate(
        self,
        event: FastAgentEvent,  # noqa: ARG002
        state: SharedCognitionState,
        budget: CognitiveBudget,  # noqa: ARG002
    ) -> list[SlowAgentDirective]:
        directives: list[SlowAgentDirective] = []
        to_record: list[dict] = []

        for item in state.hypotheses_pool:
            if not isinstance(item, dict):
                continue
            description = str(item.get("description", ""))
            observable = str(item.get("observable", ""))
            fingerprint = await self._memory.fingerprint(description, observable)
            if self._memory.get_count(fingerprint) > 0:
                directives.append(
                    SlowAgentDirective(
                        directive_type=DirectiveType.FLAG_HYPOTHESIS,
                        reason="Repeated hypothesis pattern detected",
                        confidence=0.85,
                        payload={
                            "hypothesis_id": item.get("id"),
                            "fingerprint": fingerprint,
                            "violation": "pattern_repeat",
                        },
                        priority=5,
                    )
                )
            to_record.append(item)

        if to_record:
            await self._memory.record_batch(to_record)

        return directives
