"""Brave Search service for domain discovery."""

import asyncio
from dataclasses import dataclass
from typing import Any

import httpx
import structlog

from x_creative.config.settings import get_settings

logger = structlog.get_logger()


@dataclass
class SearchResult:
    """A single search result."""

    title: str
    url: str
    description: str


class DomainSearchService:
    """Service for searching domain-related information using Brave Search."""

    BRAVE_SEARCH_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"

    def __init__(self) -> None:
        """Initialize the search service."""
        self._settings = get_settings()
        self._client = httpx.AsyncClient(timeout=10.0)

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    async def __aenter__(self) -> "DomainSearchService":
        """Async context manager entry."""
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        await self.close()

    async def search_domain_concepts(
        self,
        domain_name: str,
        max_results: int = 15,
    ) -> list[SearchResult]:
        """Search for core concepts of a domain."""
        queries = self._build_bilingual_queries(
            domain_name,
            "core concepts models principles",
        )
        return await self._execute_searches(queries, max_results)

    async def search_structure_applications(
        self,
        structure_name: str,
        domain_name: str,
        max_results: int = 10,
    ) -> list[SearchResult]:
        """Search for applications of a structure in a target domain."""
        queries = [
            f"{structure_name} {domain_name} cross-domain application",
            f"{structure_name} {domain_name} analogy",
            f"{structure_name} {domain_name} 跨领域 应用",
        ]
        return await self._execute_searches(queries, max_results)

    async def search_related_domains(
        self,
        research_goal: str,
        max_results: int = 15,
    ) -> list[SearchResult]:
        """Search for domains related to a research goal."""
        queries = [
            f"{research_goal} interdisciplinary approach",
            f"{research_goal} cross-domain methods",
            f"{research_goal} 跨领域 方法",
        ]
        return await self._execute_searches(queries, max_results)

    async def search_additional_structures(
        self,
        domain_name: str,
        existing_structures: list[str],
        max_results: int = 15,
    ) -> list[SearchResult]:
        """Search for additional structures in a domain."""
        exclude_terms = " ".join(f"-{s}" for s in existing_structures[:5])
        queries = [
            f"{domain_name} important models theories {exclude_terms}",
            f"{domain_name} key concepts principles",
            f"{domain_name} 重要概念 理论模型",
        ]
        return await self._execute_searches(queries, max_results)

    def _build_bilingual_queries(
        self,
        term: str,
        context: str,
    ) -> list[str]:
        """Build queries in both Chinese and English."""
        return [
            f"{term} {context}",
            f"{term} 核心概念 理论",
        ]

    async def _execute_searches(
        self,
        queries: list[str],
        max_results: int,
    ) -> list[SearchResult]:
        """Execute multiple searches and merge results."""
        api_key = self._settings.brave_search.api_key.get_secret_value()
        if not api_key:
            logger.warning("Brave Search API key not configured")
            return []

        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": api_key,
        }

        async def _fetch_one(query: str) -> list[SearchResult]:
            """Fetch results for a single query."""
            try:
                response = await self._client.get(
                    self.BRAVE_SEARCH_ENDPOINT,
                    headers=headers,
                    params={"q": query, "count": max_results},
                )
                response.raise_for_status()
                data = response.json()
                return [
                    SearchResult(
                        title=result.get("title", ""),
                        url=result.get("url", ""),
                        description=result.get("description", ""),
                    )
                    for result in data.get("web", {}).get("results", [])
                    if result.get("url")
                ]
            except Exception as e:
                logger.warning("Search query failed", query=query, error=str(e))
                return []

        query_results = await asyncio.gather(*[_fetch_one(q) for q in queries])

        # Merge and deduplicate
        all_results: list[SearchResult] = []
        seen_urls: set[str] = set()
        for results in query_results:
            for result in results:
                if result.url not in seen_urls:
                    seen_urls.add(result.url)
                    all_results.append(result)

        return all_results
