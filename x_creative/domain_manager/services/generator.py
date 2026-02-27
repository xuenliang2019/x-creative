"""LLM-based domain generation service."""

import json
import re
from dataclasses import dataclass

import structlog

from x_creative.core.types import DomainStructure, TargetMapping
from x_creative.domain_manager.services.search import SearchResult
from x_creative.llm.router import ModelRouter

logger = structlog.get_logger()


@dataclass
class DomainRecommendation:
    """A recommended domain from exploration."""

    domain_name: str
    domain_name_en: str
    relevance_reason: str
    potential_structures: list[str]
    # Fields for similarity detection
    is_extension: bool = False  # True if similar to existing domain
    existing_domain_id: str | None = None  # ID of similar existing domain
    existing_domain_name: str | None = None  # Name of similar existing domain


class DomainGeneratorService:
    """Service for generating domain content using LLM."""

    def __init__(self, target_domain_name: str = "开源软件开发选题") -> None:
        """Initialize the generator service.

        Args:
            target_domain_name: Display name of the target domain for LLM prompts.
        """
        self._router = ModelRouter()
        self._target_domain_name = target_domain_name

    async def close(self) -> None:
        """Close the router."""
        await self._router.close()

    async def __aenter__(self) -> "DomainGeneratorService":
        """Async context manager entry."""
        return self

    async def __aexit__(self, *args) -> None:
        """Async context manager exit."""
        await self.close()

    async def check_domain_similarity(
        self,
        recommendation: DomainRecommendation,
        existing_domains: list[tuple[str, str, str]],  # [(id, name, name_en), ...]
    ) -> DomainRecommendation:
        """Check if a recommended domain is similar to any existing domain.

        Uses LLM to determine semantic similarity.

        Args:
            recommendation: The domain recommendation to check
            existing_domains: List of (id, name, name_en) tuples for existing domains

        Returns:
            Updated recommendation with is_extension and existing_domain_* fields set
        """
        if not existing_domains:
            return recommendation

        existing_list = "\n".join(
            f"- ID: {d[0]}, 名称: {d[1]}, 英文名: {d[2] or 'N/A'}"
            for d in existing_domains
        )

        prompt = f"""你是一个学科分类专家。判断推荐的领域是否与现有领域列表中的某个领域本质上是同一概念。

推荐领域：
- 名称: {recommendation.domain_name}
- 英文名: {recommendation.domain_name_en}

现有领域列表：
{existing_list}

判断标准：
1. 如果推荐领域与某个现有领域是同一学科/同一概念的不同表述，则认为"相同"
2. 如果推荐领域是某个现有领域的子领域或高度相关领域，也认为"相同"
3. 例如："博弈论"和"Game Theory"是相同的；"进化博弈"和"博弈论"是相同的

返回 JSON 格式：
```json
{{
    "is_similar": true或false,
    "similar_domain_id": "如果相似，填写现有领域的ID；否则为null",
    "reason": "判断理由"
}}
```"""

        result = await self._router.complete(
            task="knowledge_extraction",
            messages=[{"role": "user", "content": prompt}],
        )

        try:
            data = self._extract_json(result.content)
            if data.get("is_similar"):
                similar_id = data.get("similar_domain_id")
                # Find the matching domain info
                for d_id, d_name, d_name_en in existing_domains:
                    if d_id == similar_id:
                        return DomainRecommendation(
                            domain_name=recommendation.domain_name,
                            domain_name_en=recommendation.domain_name_en,
                            relevance_reason=recommendation.relevance_reason,
                            potential_structures=recommendation.potential_structures,
                            is_extension=True,
                            existing_domain_id=d_id,
                            existing_domain_name=d_name,
                        )
        except Exception as e:
            logger.warning("Failed to parse similarity check", error=str(e))

        return recommendation

    async def filter_duplicate_structures(
        self,
        new_structures: list[DomainStructure],
        existing_structures: list[DomainStructure],
    ) -> list[DomainStructure]:
        """Filter out structures that are semantically similar to existing ones.

        Uses LLM to determine semantic similarity.

        Args:
            new_structures: List of new structures to filter
            existing_structures: List of existing structures in the domain

        Returns:
            Filtered list of new structures (excluding duplicates)
        """
        if not existing_structures or not new_structures:
            return new_structures

        existing_list = "\n".join(
            f"- ID: {s.id}, 名称: {s.name}, 描述: {s.description}"
            for s in existing_structures
        )

        new_list = "\n".join(
            f"- 索引 {i}: 名称: {s.name}, 描述: {s.description}"
            for i, s in enumerate(new_structures)
        )

        prompt = f"""你是一个学科概念分析专家。判断哪些新结构与现有结构在语义上是重复的。

现有结构：
{existing_list}

新结构：
{new_list}

判断标准：
1. 如果新结构描述的概念/规律与某个现有结构本质相同，则认为"重复"
2. 名称不同但概念相同也算重复（如"熵增定律"和"热力学第二定律"）
3. 如果新结构是现有结构的细化或变体，也算重复

返回 JSON 格式：
```json
{{
    "duplicate_indices": [重复的新结构索引列表，如 [0, 2]],
    "reasons": {{"索引": "重复原因"}}
}}
```"""

        result = await self._router.complete(
            task="knowledge_extraction",
            messages=[{"role": "user", "content": prompt}],
        )

        try:
            data = self._extract_json(result.content)
            duplicate_indices = set(data.get("duplicate_indices", []))
            return [
                s for i, s in enumerate(new_structures)
                if i not in duplicate_indices
            ]
        except Exception as e:
            logger.warning("Failed to parse duplicate check", error=str(e))
            return new_structures

    async def recommend_domains(
        self,
        research_goal: str,
        search_results: list[SearchResult],
    ) -> list[DomainRecommendation]:
        """Recommend domains based on research goal."""
        context = self._format_search_results(search_results)
        target = self._target_domain_name

        prompt = f"""你是一个跨学科研究专家。根据研究目标和搜索结果，推荐可能有价值的源领域。

研究目标：{research_goal}
目标应用领域：{target}

搜索结果摘要：
{context}

请推荐 5-8 个可能对该研究目标有启发价值的学科/领域。每个领域应该：
1. 具有可迁移到{target}领域的结构化概念
2. 与已有的常见{target}理论有足够的"距离"（远域联想）
3. 有明确的动态规律或模型

返回 JSON 格式：
```json
{{
    "recommendations": [
        {{
            "domain_name": "领域中文名",
            "domain_name_en": "Domain English Name",
            "relevance_reason": "与研究目标的关联说明",
            "potential_structures": ["可能的结构1", "可能的结构2"]
        }}
    ]
}}
```"""

        result = await self._router.complete(
            task="knowledge_extraction",
            messages=[{"role": "user", "content": prompt}],
        )

        return self._parse_recommendations(result.content)

    async def extract_structures(
        self,
        domain_name: str,
        search_results: list[SearchResult],
    ) -> list[DomainStructure]:
        """Extract structures from a domain."""
        context = self._format_search_results(search_results)
        target = self._target_domain_name

        prompt = f"""你是一个跨学科研究专家。从给定领域中提取可迁移的结构模式。

领域：{domain_name}
目标应用领域：{target}

搜索结果摘要：
{context}

请提取 3-5 个该领域中最具"可迁移性"的结构模式。每个结构应该：
1. 描述一个通用的动态规律或模式
2. 有明确的关键变量
3. 能够类比到{target}等其他领域

返回 JSON 格式：
```json
{{
    "structures": [
        {{
            "id": "structure_id_snake_case",
            "name": "结构中文名",
            "description": "结构描述（1-2句话）",
            "key_variables": ["变量1", "变量2"],
            "dynamics": "动态规律描述"
        }}
    ]
}}
```"""

        result = await self._router.complete(
            task="knowledge_extraction",
            messages=[{"role": "user", "content": prompt}],
        )

        return self._parse_structures(result.content)

    async def generate_mapping(
        self,
        structure: DomainStructure,
        search_results: list[SearchResult],
    ) -> TargetMapping:
        """Generate target mapping for a structure."""
        context = self._format_search_results(search_results)
        target = self._target_domain_name

        prompt = f"""你是一个{target}研究专家。为给定的源领域结构生成{target}领域的映射。

源结构：
- ID: {structure.id}
- 名称: {structure.name}
- 描述: {structure.description}
- 关键变量: {', '.join(structure.key_variables)}
- 动态规律: {structure.dynamics}

搜索结果摘要：
{context}

请生成这个结构在{target}领域的映射，包括：
1. target: {target}领域的目标概念
2. observable: 可观测的代理变量或指标

返回 JSON 格式：
```json
{{
    "target": "目标概念",
    "observable": "可观测代理变量"
}}
```"""

        result = await self._router.complete(
            task="knowledge_extraction",
            messages=[{"role": "user", "content": prompt}],
        )

        return self._parse_mapping(structure.id, result.content)

    def _format_search_results(self, results: list[SearchResult]) -> str:
        """Format search results for prompt context."""
        if not results:
            return "（无搜索结果）"

        lines = []
        for r in results[:10]:
            lines.append(f"- {r.title}: {r.description[:200]}")
        return "\n".join(lines)

    def _parse_recommendations(self, content: str) -> list[DomainRecommendation]:
        """Parse LLM response into recommendations."""
        try:
            data = self._extract_json(content)
            return [
                DomainRecommendation(
                    domain_name=r["domain_name"],
                    domain_name_en=r["domain_name_en"],
                    relevance_reason=r["relevance_reason"],
                    potential_structures=r["potential_structures"],
                )
                for r in data.get("recommendations", [])
            ]
        except Exception as e:
            logger.error("Failed to parse recommendations", error=str(e))
            return []

    def _parse_structures(self, content: str) -> list[DomainStructure]:
        """Parse LLM response into structures."""
        try:
            data = self._extract_json(content)
            return [
                DomainStructure(
                    id=s["id"],
                    name=s["name"],
                    description=s["description"],
                    key_variables=s["key_variables"],
                    dynamics=s["dynamics"],
                )
                for s in data.get("structures", [])
            ]
        except Exception as e:
            logger.error("Failed to parse structures", error=str(e))
            return []

    def _parse_mapping(self, structure_id: str, content: str) -> TargetMapping:
        """Parse LLM response into target mapping."""
        try:
            data = self._extract_json(content)
            return TargetMapping(
                structure=structure_id,
                target=data.get("target", ""),
                observable=data.get("observable", ""),
            )
        except Exception as e:
            logger.error("Failed to parse mapping", error=str(e))
            return TargetMapping(
                structure=structure_id,
                target="解析失败",
                observable="解析失败",
            )

    def _extract_json(self, content: str) -> dict:
        """Extract JSON from LLM response."""
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", content, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        return json.loads(content)
