"""LLM-based target domain metadata generation service."""

import asyncio
import json
import re
from collections.abc import Callable
from typing import Any

import structlog

from x_creative.core.plugin import DomainConstraint
from x_creative.core.types import Domain, TargetMapping
from x_creative.creativity.utils import safe_json_loads
from x_creative.llm.router import ModelRouter

logger = structlog.get_logger()


class TargetGeneratorService:
    """Service for generating target domain metadata using LLM."""

    def __init__(self) -> None:
        """Initialize the generator service."""
        self._router = ModelRouter()

    async def close(self) -> None:
        """Close the router."""
        await self._router.close()

    async def __aenter__(self) -> "TargetGeneratorService":
        """Async context manager entry."""
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        await self.close()

    async def generate_constraints(
        self,
        target_name: str,
        target_description: str,
    ) -> list[DomainConstraint]:
        """Generate domain constraints for a target domain.

        Args:
            target_name: Target domain display name.
            target_description: Target domain description.

        Returns:
            List of DomainConstraint objects.
        """
        prompt = f"""你是一个领域分析专家。为给定的目标领域生成关键约束条件。

目标领域：{target_name}
描述：{target_description}

请生成 5-8 个约束条件。每个约束应包含：
1. name: 唯一标识符 (snake_case)
2. description: 人类可读的描述
3. severity: "critical"（违反会导致完全失败）、"important"（显著影响质量）或 "advisory"（建议遵循）
4. check_prompt: 用于 LLM 验证的提示语

返回 JSON 格式：
```json
{{
    "constraints": [
        {{
            "name": "constraint_id",
            "description": "约束描述",
            "severity": "critical",
            "check_prompt": "检查..."
        }}
    ]
}}
```"""

        result = await self._router.complete(
            task="knowledge_extraction",
            messages=[{"role": "user", "content": prompt}],
        )

        try:
            data = self._extract_json(result.content)
            return [
                DomainConstraint(**c)
                for c in data.get("constraints", [])
            ]
        except Exception as e:
            logger.error("Failed to parse constraints", error=str(e))
            return []

    async def generate_evaluation_criteria(
        self,
        target_name: str,
        target_description: str,
    ) -> list[str]:
        """Generate evaluation criteria for a target domain.

        Args:
            target_name: Target domain display name.
            target_description: Target domain description.

        Returns:
            List of evaluation criteria strings.
        """
        prompt = f"""你是一个领域评估专家。为给定的目标领域生成评估指标。

目标领域：{target_name}
描述：{target_description}

请生成 5-8 个可量化或可定性评估的指标。每个指标应该：
1. 具体且可衡量
2. 与领域核心目标相关
3. 包含中英文名称（如有通用英文名）

返回 JSON 格式：
```json
{{
    "evaluation_criteria": [
        "指标名称 (English Name)",
        "指标名称"
    ]
}}
```"""

        result = await self._router.complete(
            task="knowledge_extraction",
            messages=[{"role": "user", "content": prompt}],
        )

        try:
            data = self._extract_json(result.content)
            return data.get("evaluation_criteria", [])
        except Exception as e:
            logger.error("Failed to parse evaluation criteria", error=str(e))
            return []

    async def generate_anti_patterns(
        self,
        target_name: str,
        target_description: str,
    ) -> list[str]:
        """Generate anti-patterns for a target domain.

        Args:
            target_name: Target domain display name.
            target_description: Target domain description.

        Returns:
            List of anti-pattern strings.
        """
        prompt = f"""你是一个领域实践专家。为给定的目标领域生成常见的错误做法（反模式）。

目标领域：{target_name}
描述：{target_description}

请生成 5-10 个该领域中常见的错误做法或应避免的模式。每个反模式应该：
1. 描述一个具体的错误做法
2. 是该领域从业者容易犯的错误
3. 简洁明了（一句话）

返回 JSON 格式：
```json
{{
    "anti_patterns": [
        "错误做法描述",
        "错误做法描述"
    ]
}}
```"""

        result = await self._router.complete(
            task="knowledge_extraction",
            messages=[{"role": "user", "content": prompt}],
        )

        try:
            data = self._extract_json(result.content)
            return data.get("anti_patterns", [])
        except Exception as e:
            logger.error("Failed to parse anti-patterns", error=str(e))
            return []

    async def generate_terminology(
        self,
        target_name: str,
        target_description: str,
    ) -> dict[str, str]:
        """Generate domain-specific terminology for a target domain.

        Args:
            target_name: Target domain display name.
            target_description: Target domain description.

        Returns:
            Dict of term -> definition.
        """
        prompt = f"""你是一个领域术语专家。为给定的目标领域生成关键术语表。

目标领域：{target_name}
描述：{target_description}

请生成 8-15 个关键术语及其精确定义。每个术语应该：
1. 是该领域的核心概念
2. 定义简洁准确
3. 对跨领域研究者有帮助

返回 JSON 格式：
```json
{{
    "terminology": {{
        "术语名": "术语定义",
        "术语名": "术语定义"
    }}
}}
```"""

        result = await self._router.complete(
            task="knowledge_extraction",
            messages=[{"role": "user", "content": prompt}],
        )

        try:
            data = self._extract_json(result.content)
            return data.get("terminology", {})
        except Exception as e:
            logger.error("Failed to parse terminology", error=str(e))
            return {}

    async def generate_stale_ideas(
        self,
        target_name: str,
        target_description: str,
    ) -> list[str]:
        """Generate stale/overused ideas for a target domain.

        Args:
            target_name: Target domain display name.
            target_description: Target domain description.

        Returns:
            List of stale idea strings.
        """
        prompt = f"""你是一个领域创新专家。为给定的目标领域列出已被广泛使用的陈旧想法。

目标领域：{target_name}
描述：{target_description}

请列出 5-8 个在该领域中已被充分探索、缺乏新意的想法或方法。这些是：
1. 已被广泛研究和使用的方法
2. 难以再产生新突破的方向
3. 在创新性评估中应被标记为"陈旧"的

返回 JSON 格式：
```json
{{
    "stale_ideas": [
        "陈旧想法描述",
        "陈旧想法描述"
    ]
}}
```"""

        result = await self._router.complete(
            task="knowledge_extraction",
            messages=[{"role": "user", "content": prompt}],
        )

        try:
            data = self._extract_json(result.content)
            return data.get("stale_ideas", [])
        except Exception as e:
            logger.error("Failed to parse stale ideas", error=str(e))
            return []

    async def generate_full_metadata(
        self,
        target_name: str,
        target_description: str,
    ) -> dict[str, Any]:
        """Generate all metadata sections in parallel.

        Args:
            target_name: Target domain display name.
            target_description: Target domain description.

        Returns:
            Dict with keys: constraints, evaluation_criteria,
            anti_patterns, terminology, stale_ideas.
        """
        results = await asyncio.gather(
            self.generate_constraints(target_name, target_description),
            self.generate_evaluation_criteria(target_name, target_description),
            self.generate_anti_patterns(target_name, target_description),
            self.generate_terminology(target_name, target_description),
            self.generate_stale_ideas(target_name, target_description),
        )

        constraints_dicts = [
            {
                "name": c.name,
                "description": c.description,
                "severity": c.severity,
                "check_prompt": c.check_prompt,
            }
            for c in results[0]
        ]

        return {
            "constraints": constraints_dicts,
            "evaluation_criteria": results[1],
            "anti_patterns": results[2],
            "terminology": results[3],
            "stale_ideas": results[4],
        }

    async def rewrite_target_mappings(
        self,
        source_domain: Domain,
        new_target_name: str,
        new_target_description: str,
        base_target_name: str,
    ) -> list[TargetMapping]:
        """Rewrite target mappings for a source domain to a new target.

        Args:
            source_domain: Source domain with existing mappings.
            new_target_name: New target domain name.
            new_target_description: New target domain description.
            base_target_name: Original target domain name for context.

        Returns:
            List of rewritten TargetMapping objects.
        """
        structures_text = "\n".join(
            f"- {s.name} (ID: {s.id}): {s.description}\n"
            f"  关键变量: {', '.join(s.key_variables)}\n"
            f"  动态规律: {s.dynamics}"
            for s in source_domain.structures
        )

        existing_mappings_text = "\n".join(
            f"- structure: {m.structure}, target: {m.target}, observable: {m.observable}"
            for m in source_domain.target_mappings
        )

        prompt = f"""你是一个跨领域映射专家。将源领域的结构映射从一个目标领域重写为另一个目标领域。

源领域：{source_domain.name} ({source_domain.name_en or ''})
源领域描述：{source_domain.description}

源领域结构：
{structures_text}

原目标领域：{base_target_name}
原映射：
{existing_mappings_text}

新目标领域：{new_target_name}
新目标描述：{new_target_description}

请为每个 structure 重写 target 和 observable，使其适用于新目标领域的语境。
- target: 新目标领域中对应的概念
- observable: 新目标领域中可观测的代理变量或指标

返回 JSON 格式：
```json
{{
    "mappings": [
        {{
            "structure": "structure_id",
            "target": "新目标概念",
            "observable": "新可观测指标"
        }}
    ]
}}
```"""

        result = await self._router.complete(
            task="knowledge_extraction",
            messages=[{"role": "user", "content": prompt}],
        )

        try:
            data = self._extract_json(result.content)
            return [
                TargetMapping(**m)
                for m in data.get("mappings", [])
            ]
        except Exception as e:
            logger.error("Failed to parse rewritten mappings", error=str(e))
            return source_domain.target_mappings

    async def batch_rewrite_source_domains(
        self,
        source_domains: list[Domain],
        new_target_name: str,
        new_target_description: str,
        base_target_name: str,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[Domain]:
        """Batch rewrite target mappings for multiple source domains.

        Args:
            source_domains: List of source domains to rewrite.
            new_target_name: New target domain name.
            new_target_description: New target domain description.
            base_target_name: Original target domain name.
            progress_callback: Optional callback(current, total).

        Returns:
            List of Domain objects with rewritten mappings.
        """
        rewritten_domains = []
        total = len(source_domains)

        for i, domain in enumerate(source_domains):
            new_mappings = await self.rewrite_target_mappings(
                domain, new_target_name, new_target_description, base_target_name,
            )
            rewritten_domain = Domain(
                id=domain.id,
                name=domain.name,
                name_en=domain.name_en,
                description=domain.description,
                structures=domain.structures,
                target_mappings=new_mappings,
            )
            rewritten_domains.append(rewritten_domain)

            if progress_callback:
                progress_callback(i + 1, total)

        return rewritten_domains

    def _extract_json(self, content: str) -> dict[str, Any]:
        """Extract JSON from LLM response.

        Tries to find JSON in markdown code blocks first,
        then falls back to raw JSON parsing with safe_json_loads.
        """
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", content, re.DOTALL)
        text = match.group(1) if match else content
        return safe_json_loads(text)
