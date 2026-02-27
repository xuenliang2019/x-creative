"""Tests for Creativity Engine components."""

import asyncio
import json
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from x_creative.core.types import (
    Domain,
    DomainStructure,
    Hypothesis,
    HypothesisScores,
    ProblemFrame,
    SearchConfig,
    TargetMapping,
)


class TestBISOModule:
    """Tests for the BISO (Bisociation) module."""

    @pytest.fixture
    def sample_domain(self) -> Domain:
        """Create a sample domain for testing."""
        return Domain(
            id="test_domain",
            name="测试领域",
            name_en="Test Domain",
            description="A test domain for unit tests",
            structures=[
                DomainStructure(
                    id="test_structure",
                    name="测试结构",
                    description="A test structure",
                    key_variables=["x", "y", "z"],
                    dynamics="Test dynamics description",
                )
            ],
            target_mappings=[
                TargetMapping(
                    structure="test_structure",
                    target="测试目标",
                    observable="test_observable",
                )
            ],
        )

    @pytest.fixture
    def sample_problem(self) -> ProblemFrame:
        """Create a sample problem frame."""
        return ProblemFrame(
            description="设计一个能实现病毒式传播的开源命令行工具",
        )

    def test_biso_module_creation(self) -> None:
        """Test creating a BISO module."""
        from x_creative.creativity.biso import BISOModule

        biso = BISOModule()
        assert biso is not None

    def test_safe_json_loads_valid_json(self) -> None:
        """Test safe_json_loads with valid JSON."""
        from x_creative.creativity.utils import safe_json_loads

        result = safe_json_loads('[{"key": "value"}]')
        assert result == [{"key": "value"}]

    def test_safe_json_loads_invalid_escape(self) -> None:
        """Test safe_json_loads with invalid escape sequences."""
        from x_creative.creativity.utils import safe_json_loads

        # This has an invalid \e escape that should be fixed
        json_with_bad_escape = '[{"formula": "x \\eq y + z"}]'
        result = safe_json_loads(json_with_bad_escape)
        assert len(result) == 1
        assert "formula" in result[0]

    def test_parse_analogies_skips_empty_observable(self, sample_domain: Domain) -> None:
        """BISO parser should discard analogies without a concrete observable."""
        from x_creative.creativity.biso import BISOModule

        biso = BISOModule()
        content = """
[
    {
        "structure_id": "test_structure",
        "analogy": "valid analogy",
        "explanation": "valid explanation",
        "observable": "signal = x / y",
        "mapping_table": [{
            "source_concept": "queue",
            "target_concept": "orderflow",
            "source_relation": "arrival > service",
            "target_relation": "buy > sell pressure",
            "mapping_type": "relation",
            "systematicity_group_id": "g1"
        }],
        "failure_modes": [{
            "scenario": "regime shift",
            "why_breaks": "mapping collapses",
            "detectable_signal": "spread spike"
        }]
    },
    {
        "structure_id": "test_structure",
        "analogy": "missing observable",
        "explanation": "bad candidate",
        "observable": ""
    },
    {
        "structure_id": "test_structure",
        "analogy": "missing field",
        "explanation": "also bad"
    }
]
"""
        parsed = biso._parse_analogies(content, sample_domain)
        assert len(parsed) == 1
        assert parsed[0].description == "valid analogy"
        assert parsed[0].observable == "signal = x / y"

    @pytest.mark.asyncio
    async def test_generate_analogies_from_domain(
        self, sample_domain: Domain, sample_problem: ProblemFrame
    ) -> None:
        """Test generating analogies from a single domain."""
        from x_creative.creativity.biso import BISOModule

        biso = BISOModule()

        with patch.object(biso, "_router") as mock_router:
            # Mock LLM response
            mock_router.complete = AsyncMock(
                return_value=MagicMock(
                    content="""
[
    {
        "structure_id": "test_structure",
        "analogy": "测试类比",
        "explanation": "这是一个测试类比的解释",
        "observable": "test_factor = x / y",
        "confidence": 0.8,
        "mapping_table": [{
            "source_concept": "A",
            "target_concept": "B",
            "source_relation": "R1",
            "target_relation": "R2",
            "mapping_type": "relation",
            "systematicity_group_id": "g1"
        }],
        "failure_modes": [{
            "scenario": "When X",
            "why_breaks": "Because Y",
            "detectable_signal": "Signal Z"
        }]
    }
]
"""
                )
            )

            analogies = await biso.generate_analogies(
                domain=sample_domain,
                problem=sample_problem,
            )

            assert len(analogies) >= 1
            assert analogies[0].source_domain == "test_domain"

    @pytest.mark.asyncio
    async def test_generate_analogies_all_domains(
        self, sample_problem: ProblemFrame
    ) -> None:
        """Test generating analogies from all domains."""
        from x_creative.creativity.biso import BISOModule

        biso = BISOModule()

        with patch.object(biso, "_router") as mock_router:
            mock_router.complete = AsyncMock(
                return_value=MagicMock(content="[]")
            )

            analogies = await biso.generate_all_analogies(
                problem=sample_problem,
                max_domains=3,  # Limit for testing
            )

            # Should have attempted calls for domains
            assert mock_router.complete.call_count >= 1

    @pytest.mark.asyncio
    async def test_generate_all_analogies_concurrent(
        self, sample_problem: ProblemFrame
    ) -> None:
        """Test that generate_all_analogies runs concurrently."""
        from x_creative.creativity.biso import BISOModule

        biso = BISOModule()
        call_times: list[float] = []

        async def mock_complete(*args: Any, **kwargs: Any) -> MagicMock:
            """Mock that records call time and simulates API delay."""
            call_times.append(time.time())
            await asyncio.sleep(0.1)  # Simulate 100ms API delay
            return MagicMock(content="[]")

        with patch.object(biso, "_router") as mock_router:
            mock_router.complete = mock_complete

            start = time.time()
            await biso.generate_all_analogies(
                problem=sample_problem,
                max_domains=5,
            )
            elapsed = time.time() - start

            # With 5 domains and 100ms delay each:
            # - Sequential would take ~500ms
            # - Concurrent should take ~100ms (+ overhead)
            assert elapsed < 0.3, f"Expected concurrent execution, but took {elapsed:.2f}s"
            assert len(call_times) == 5

            # Verify calls started nearly simultaneously (within 50ms of each other)
            if len(call_times) >= 2:
                time_spread = max(call_times) - min(call_times)
                assert time_spread < 0.05, f"Calls not concurrent, spread: {time_spread:.3f}s"

    @pytest.mark.asyncio
    async def test_generate_all_analogies_emits_domain_progress(
        self,
        sample_problem: ProblemFrame,
        sample_domain: Domain,
    ) -> None:
        """generate_all_analogies should call on_domain_complete once per domain."""
        from x_creative.creativity.biso import BISOModule

        biso = BISOModule()
        domains = [
            sample_domain.model_copy(update={"id": "d1"}),
            sample_domain.model_copy(update={"id": "d2"}),
            sample_domain.model_copy(update={"id": "d3"}),
        ]

        on_domain_complete = AsyncMock()

        with patch.object(biso, "generate_analogies", AsyncMock(return_value=[])):
            await biso.generate_all_analogies(
                problem=sample_problem,
                num_per_domain=1,
                max_concurrency=2,
                source_domains=domains,
                on_domain_complete=on_domain_complete,
            )

        assert on_domain_complete.await_count == len(domains)
        calls = list(on_domain_complete.await_args_list)
        seen_domains = {call.args[0] for call in calls}
        assert seen_domains == {"d1", "d2", "d3"}
        assert all(call.args[2] == len(domains) for call in calls)  # total domains
        completed = sorted(call.args[1] for call in calls)
        assert completed[0] == 1
        assert completed[-1] == len(domains)

    @pytest.mark.asyncio
    async def test_generate_all_analogies_with_concurrency_limit(
        self, sample_problem: ProblemFrame
    ) -> None:
        """Test that max_concurrency limits parallel execution."""
        from x_creative.creativity.biso import BISOModule

        biso = BISOModule()
        concurrent_count = 0
        max_concurrent = 0
        lock = asyncio.Lock()

        async def mock_complete(*args: Any, **kwargs: Any) -> MagicMock:
            """Mock that tracks concurrent executions."""
            nonlocal concurrent_count, max_concurrent
            async with lock:
                concurrent_count += 1
                max_concurrent = max(max_concurrent, concurrent_count)
            await asyncio.sleep(0.05)  # Simulate API delay
            async with lock:
                concurrent_count -= 1
            return MagicMock(content="[]")

        with patch.object(biso, "_router") as mock_router:
            mock_router.complete = mock_complete

            await biso.generate_all_analogies(
                problem=sample_problem,
                max_domains=6,
                max_concurrency=2,
            )

            # Max concurrent should not exceed the limit
            assert max_concurrent <= 2, f"Exceeded concurrency limit: {max_concurrent}"
            # Should have been used (at least 2 concurrent at some point with 6 domains)
            assert max_concurrent == 2, f"Expected max concurrency of 2, got {max_concurrent}"

    @pytest.mark.asyncio
    async def test_generate_all_analogies_default_uses_bounded_concurrency(
        self, sample_problem: ProblemFrame
    ) -> None:
        """Default path should still be bounded by semaphore control."""
        from x_creative.creativity.biso import BISOModule

        biso = BISOModule()
        concurrent_count = 0
        max_concurrent = 0
        lock = asyncio.Lock()

        async def mock_complete(*args: Any, **kwargs: Any) -> MagicMock:
            nonlocal concurrent_count, max_concurrent
            async with lock:
                concurrent_count += 1
                max_concurrent = max(max_concurrent, concurrent_count)
            await asyncio.sleep(0.03)
            async with lock:
                concurrent_count -= 1
            return MagicMock(content="[]")

        with patch.object(biso, "_router") as mock_router:
            mock_router.complete = mock_complete
            await biso.generate_all_analogies(
                problem=sample_problem,
                max_domains=biso._default_max_concurrency + 4,
                max_concurrency=None,
            )

        assert max_concurrent <= biso._default_max_concurrency

    @pytest.mark.asyncio
    async def test_generate_all_analogies_error_handling(
        self, sample_problem: ProblemFrame
    ) -> None:
        """Test that errors in one domain don't affect others in concurrent execution."""
        from x_creative.creativity.biso import BISOModule

        biso = BISOModule()
        call_count = 0

        async def mock_complete(*args: Any, **kwargs: Any) -> MagicMock:
            """Mock that fails on first call but succeeds on others."""
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Simulated API error")
            return MagicMock(
                content='[{"structure_id": "s1", "analogy": "test", "explanation": "e", '
                '"observable": "o", "mapping_table": [{"source_concept": "A", '
                '"target_concept": "B", "source_relation": "R1", "target_relation": "R2", '
                '"mapping_type": "relation", "systematicity_group_id": "g1"}], '
                '"failure_modes": [{"scenario": "s", "why_breaks": "w", '
                '"detectable_signal": "d"}]}]'
            )

        with patch.object(biso, "_router") as mock_router:
            mock_router.complete = mock_complete

            # Should not raise, errors are caught per-domain
            hypotheses = await biso.generate_all_analogies(
                problem=sample_problem,
                max_domains=3,
            )

            # Should have results from successful domains
            # 3 domain calls + 1 dedup call = 4 total
            assert call_count == 4
            # 2 successful domains should produce hypotheses
            assert len(hypotheses) == 2

    @pytest.mark.asyncio
    async def test_generate_all_analogies_default_allows_full_parallelism_under_limit(
        self, sample_problem: ProblemFrame
    ) -> None:
        """When domain count is below limit, all domains can run concurrently."""
        from x_creative.creativity.biso import BISOModule

        biso = BISOModule()
        concurrent_count = 0
        max_concurrent = 0
        lock = asyncio.Lock()

        async def mock_complete(*args: Any, **kwargs: Any) -> MagicMock:
            nonlocal concurrent_count, max_concurrent
            async with lock:
                concurrent_count += 1
                max_concurrent = max(max_concurrent, concurrent_count)
            await asyncio.sleep(0.02)
            async with lock:
                concurrent_count -= 1
            return MagicMock(content="[]")

        with patch.object(biso, "_router") as mock_router:
            mock_router.complete = mock_complete

            await biso.generate_all_analogies(
                problem=sample_problem,
                max_domains=5,
                max_concurrency=None,
            )

            # All 5 should run concurrently
            assert max_concurrent == 5

    @pytest.mark.asyncio
    async def test_generate_analogies_uses_biso_pool(
        self, sample_domain: Domain, sample_problem: ProblemFrame
    ) -> None:
        """When biso_pool is configured, a random model from the pool is used."""
        from x_creative.creativity.biso import BISOModule

        biso = BISOModule()
        biso._biso_pool = ["model-a", "model-b", "model-c"]

        valid_response = json.dumps([{
            "analogy": "Test analogy",
            "structure_id": "test_structure",
            "explanation": "Mapping explanation",
            "observable": "Test observable",
            "mapping_table": [{
                "source_concept": "A",
                "target_concept": "B",
                "source_relation": "R1",
                "target_relation": "R2",
                "mapping_type": "relation",
                "systematicity_group_id": "g1",
            }],
            "failure_modes": [{
                "scenario": "When X",
                "why_breaks": "Because Y",
                "detectable_signal": "Signal Z",
            }],
        }])

        with patch.object(biso, "_router") as mock_router:
            mock_router.complete = AsyncMock(
                return_value=MagicMock(content=valid_response)
            )
            await biso.generate_analogies(
                domain=sample_domain,
                problem=sample_problem,
                num_analogies=1,
            )
            call_kwargs = mock_router.complete.call_args.kwargs
            assert "model_override" in call_kwargs
            assert call_kwargs["model_override"] in ["model-a", "model-b", "model-c"]

    @pytest.mark.asyncio
    async def test_generate_analogies_no_pool_no_override(
        self, sample_domain: Domain, sample_problem: ProblemFrame
    ) -> None:
        """When biso_pool is empty, no model_override is passed."""
        from x_creative.creativity.biso import BISOModule

        biso = BISOModule()
        biso._biso_pool = []

        valid_response = json.dumps([{
            "analogy": "Test analogy",
            "structure_id": "test_structure",
            "explanation": "Mapping explanation",
            "observable": "Test observable",
            "mapping_table": [{
                "source_concept": "A",
                "target_concept": "B",
                "source_relation": "R1",
                "target_relation": "R2",
                "mapping_type": "relation",
                "systematicity_group_id": "g1",
            }],
            "failure_modes": [{
                "scenario": "When X",
                "why_breaks": "Because Y",
                "detectable_signal": "Signal Z",
            }],
        }])

        with patch.object(biso, "_router") as mock_router:
            mock_router.complete = AsyncMock(
                return_value=MagicMock(content=valid_response)
            )
            await biso.generate_analogies(
                domain=sample_domain,
                problem=sample_problem,
                num_analogies=1,
            )
            call_kwargs = mock_router.complete.call_args.kwargs
            assert "model_override" not in call_kwargs

    @pytest.mark.asyncio
    async def test_deduplicate_removes_semantic_duplicates(
        self, sample_domain: Domain, sample_problem: ProblemFrame
    ) -> None:
        """Dedup should remove hypotheses identified as duplicates by LLM."""
        from x_creative.creativity.biso import BISOModule

        biso = BISOModule()
        biso._biso_dedup_enabled = True

        h1 = Hypothesis(
            id="h1", description="Idea A", source_domain="d1",
            source_structure="s1", analogy_explanation="", observable="obs_a",
        )
        h2 = Hypothesis(
            id="h2", description="Idea B (unique)", source_domain="d2",
            source_structure="s2", analogy_explanation="", observable="obs_b",
        )
        h3 = Hypothesis(
            id="h3", description="Idea A rephrased", source_domain="d3",
            source_structure="s3", analogy_explanation="", observable="obs_a_v2",
        )

        dedup_response = json.dumps({
            "duplicate_groups": [[0, 2]],
            "reasoning": "h1 and h3 express the same core idea",
        })

        with patch.object(biso, "_router") as mock_router:
            mock_router.complete = AsyncMock(
                return_value=MagicMock(content=dedup_response)
            )
            result = await biso._deduplicate_hypotheses([h1, h2, h3])

        assert len(result) == 2
        assert result[0].id == "h1"
        assert result[1].id == "h2"

    @pytest.mark.asyncio
    async def test_deduplicate_preserves_all_when_unique(
        self, sample_domain: Domain, sample_problem: ProblemFrame
    ) -> None:
        """When LLM says no duplicates, all hypotheses are kept."""
        from x_creative.creativity.biso import BISOModule

        biso = BISOModule()
        biso._biso_dedup_enabled = True

        h1 = Hypothesis(
            id="h1", description="Idea A", source_domain="d1",
            source_structure="s1", analogy_explanation="", observable="obs_a",
        )
        h2 = Hypothesis(
            id="h2", description="Idea B", source_domain="d2",
            source_structure="s2", analogy_explanation="", observable="obs_b",
        )

        dedup_response = json.dumps({
            "duplicate_groups": [],
            "reasoning": "All hypotheses are semantically distinct.",
        })

        with patch.object(biso, "_router") as mock_router:
            mock_router.complete = AsyncMock(
                return_value=MagicMock(content=dedup_response)
            )
            result = await biso._deduplicate_hypotheses([h1, h2])

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_deduplicate_disabled_skips_llm_call(self) -> None:
        """When biso_dedup_enabled is False, no LLM call is made."""
        from x_creative.creativity.biso import BISOModule

        biso = BISOModule()
        biso._biso_dedup_enabled = False

        h1 = Hypothesis(
            id="h1", description="Idea A", source_domain="d1",
            source_structure="s1", analogy_explanation="", observable="obs_a",
        )
        h2 = Hypothesis(
            id="h2", description="Idea B", source_domain="d2",
            source_structure="s2", analogy_explanation="", observable="obs_b",
        )

        with patch.object(biso, "_router") as mock_router:
            mock_router.complete = AsyncMock()
            result = await biso._deduplicate_hypotheses([h1, h2])

        assert len(result) == 2
        mock_router.complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_deduplicate_graceful_on_llm_failure(self) -> None:
        """When LLM call fails, return original list unchanged."""
        from x_creative.creativity.biso import BISOModule

        biso = BISOModule()
        biso._biso_dedup_enabled = True

        h1 = Hypothesis(
            id="h1", description="Idea A", source_domain="d1",
            source_structure="s1", analogy_explanation="", observable="obs_a",
        )
        h2 = Hypothesis(
            id="h2", description="Idea B", source_domain="d2",
            source_structure="s2", analogy_explanation="", observable="obs_b",
        )

        with patch.object(biso, "_router") as mock_router:
            mock_router.complete = AsyncMock(side_effect=RuntimeError("LLM failed"))
            result = await biso._deduplicate_hypotheses([h1, h2])

        assert len(result) == 2
        assert result[0].id == "h1"
        assert result[1].id == "h2"

    @pytest.mark.asyncio
    async def test_deduplicate_skips_single_hypothesis(self) -> None:
        """Single hypothesis should skip dedup entirely."""
        from x_creative.creativity.biso import BISOModule

        biso = BISOModule()
        biso._biso_dedup_enabled = True

        h1 = Hypothesis(
            id="h1", description="Idea A", source_domain="d1",
            source_structure="s1", analogy_explanation="", observable="obs_a",
        )

        with patch.object(biso, "_router") as mock_router:
            mock_router.complete = AsyncMock()
            result = await biso._deduplicate_hypotheses([h1])

        assert len(result) == 1
        mock_router.complete.assert_not_called()


class TestSearchModule:
    """Tests for the SEARCH (Graph of Thoughts) module."""

    @pytest.fixture
    def sample_hypotheses(self) -> list[Hypothesis]:
        """Create sample hypotheses for testing."""
        return [
            Hypothesis(
                id="hyp_1",
                description="Hypothesis 1",
                source_domain="domain_a",
                source_structure="struct_a",
                analogy_explanation="Explanation 1",
                observable="factor_1",
                generation=0,
            ),
            Hypothesis(
                id="hyp_2",
                description="Hypothesis 2",
                source_domain="domain_b",
                source_structure="struct_b",
                analogy_explanation="Explanation 2",
                observable="factor_2",
                generation=0,
            ),
        ]

    def test_search_module_creation(self) -> None:
        """Test creating a search module."""
        from x_creative.creativity.search import SearchModule

        search = SearchModule()
        assert search is not None

    @pytest.mark.asyncio
    async def test_expand_hypothesis(self, sample_hypotheses: list[Hypothesis]) -> None:
        """Test expanding a single hypothesis."""
        from x_creative.creativity.search import SearchModule

        search = SearchModule()

        with patch.object(search, "_router") as mock_router:
            mock_router.complete = AsyncMock(
                return_value=MagicMock(
                    content="""
[
    {
        "description": "Refined hypothesis",
        "analogy_explanation": "Refined explanation",
        "observable": "refined_factor",
        "expansion_type": "refine"
    }
]
"""
                )
            )

            expanded = await search.expand_hypothesis(
                hypothesis=sample_hypotheses[0],
                expansion_types=["refine"],
            )

            assert len(expanded) >= 1
            assert expanded[0].parent_id == "hyp_1"
            assert expanded[0].generation == 1

    @pytest.mark.asyncio
    async def test_expand_hypothesis_prompt_is_domain_aware(
        self,
        sample_hypotheses: list[Hypothesis],
    ) -> None:
        """SEARCH expand prompt should use target-domain context and avoid finance-only wording."""
        from x_creative.creativity.search import SearchModule

        search = SearchModule(problem_frame=ProblemFrame(
            description="设计一个能爆发式增长的开源项目选题生成框架",
            target_domain="open_source_development",
            constraints=["必须包含可验证的可观测指标/验收步骤"],
        ))

        with patch.object(search, "_router") as mock_router:
            mock_router.complete = AsyncMock(
                return_value=MagicMock(
                    content="""
[
    {
        "description": "Refined hypothesis",
        "analogy_explanation": "Refined explanation",
        "observable": "metric: d7_retention, test: run_ci_benchmark",
        "expansion_type": "refine"
    }
]
"""
                )
            )

            await search.expand_hypothesis(
                hypothesis=sample_hypotheses[0],
                expansion_types=["refine"],
            )

            call = mock_router.complete.call_args
            assert call is not None
            messages = call.kwargs.get("messages") or call.args[1]
            prompt = messages[0]["content"]

            assert "open_source_development" in prompt
            assert "开源软件开发选题" in prompt
            assert "financial trading" not in prompt.lower()

    @pytest.mark.asyncio
    async def test_search_iteration(self, sample_hypotheses: list[Hypothesis]) -> None:
        """Test a single iteration of the search."""
        from x_creative.creativity.search import SearchModule

        search = SearchModule()
        config = SearchConfig(search_depth=1, search_breadth=2)

        with patch.object(search, "_router") as mock_router:
            mock_router.complete = AsyncMock(
                return_value=MagicMock(content="[]")
            )

            result = await search.search_iteration(
                hypotheses=sample_hypotheses,
                config=config,
            )

            # Should return at least the original hypotheses
            assert len(result) >= len(sample_hypotheses)

    @pytest.mark.asyncio
    async def test_run_search_calls_round_callback(
        self,
        sample_hypotheses: list[Hypothesis],
    ) -> None:
        """run_search should invoke on_round_complete once per depth."""
        from x_creative.creativity.search import SearchModule

        search = SearchModule()
        config = SearchConfig(search_depth=2, search_breadth=1)

        h3 = Hypothesis(
            id="hyp_3",
            description="Hypothesis 3",
            source_domain="domain_c",
            source_structure="struct_c",
            analogy_explanation="Explanation 3",
            observable="factor_3",
            generation=1,
        )
        h4 = Hypothesis(
            id="hyp_4",
            description="Hypothesis 4",
            source_domain="domain_d",
            source_structure="struct_d",
            analogy_explanation="Explanation 4",
            observable="factor_4",
            generation=2,
        )

        with patch.object(
            search,
            "search_iteration",
            AsyncMock(
                side_effect=[
                    [sample_hypotheses[0], h3],
                    [sample_hypotheses[0], h4],
                ]
            ),
        ), patch.object(
            search,
            "_prescore_hypotheses",
            AsyncMock(return_value=None),
        ):
            rounds: list[tuple[int, int, int]] = []

            async def on_round_complete(round_idx: int, pool: list[Hypothesis], new_count: int) -> None:
                rounds.append((round_idx, len(pool), new_count))

            result = await search.run_search(
                initial_hypotheses=list(sample_hypotheses),
                config=config,
                on_round_complete=on_round_complete,
            )

        assert len(result) == 4
        assert [item[0] for item in rounds] == [1, 2]
        assert [item[2] for item in rounds] == [1, 1]


class TestVerifyModule:
    """Tests for the VERIFY (scoring) module."""

    @pytest.fixture
    def sample_hypothesis(self) -> Hypothesis:
        """Create a sample hypothesis for testing."""
        return Hypothesis(
            id="hyp_test",
            description="订单流排队压力因子",
            source_domain="queueing_theory",
            source_structure="queue_dynamics",
            analogy_explanation="订单簿如同服务台队列",
            observable="bid_depth_imbalance / avg_trade_rate",
        )

    def test_verify_module_creation(self) -> None:
        """Test creating a verify module."""
        from x_creative.creativity.verify import VerifyModule

        verify = VerifyModule()
        assert verify is not None

    @pytest.mark.asyncio
    async def test_score_hypothesis(self, sample_hypothesis: Hypothesis) -> None:
        """Test scoring a single hypothesis."""
        from x_creative.creativity.verify import VerifyModule

        verify = VerifyModule()

        with patch.object(verify, "_router") as mock_router:
            mock_router.complete = AsyncMock(
                return_value=MagicMock(
                    content="""
{
    "divergence": 8.5,
    "divergence_reason": "Uses queueing theory which is rarely applied",
    "testability": 9.0,
    "testability_reason": "Clear formula with observable variables",
    "rationale": 8.0,
    "rationale_reason": "Solid economic mechanism",
    "robustness": 7.5,
    "robustness_reason": "Single condition, reasonable scope"
}
"""
                )
            )

            scored = await verify.score_hypothesis(sample_hypothesis)

            assert scored.scores is not None
            assert scored.scores.divergence == 8.5
            assert scored.scores.testability == 9.0

    @pytest.mark.asyncio
    async def test_score_hypothesis_prompt_is_domain_aware(
        self,
        sample_hypothesis: Hypothesis,
    ) -> None:
        """VERIFY prompt should use target-domain context and avoid finance-only wording."""
        from x_creative.creativity.verify import VerifyModule

        verify = VerifyModule()
        setattr(
            verify,
            "_problem_frame",
            ProblemFrame(
                description="为开源软件的爆发式增长提出可检验假说",
                target_domain="open_source_development",
                constraints=["必须给出可观测指标/验收步骤"],
            ),
        )

        with patch.object(verify, "_router") as mock_router:
            mock_router.complete = AsyncMock(
                return_value=MagicMock(
                    content='{"divergence": 5, "testability": 5, "rationale": 5, "robustness": 5, "feasibility": 5}'
                )
            )

            await verify.score_hypothesis(sample_hypothesis)

            call = mock_router.complete.call_args
            assert call is not None
            messages = call.kwargs.get("messages") or call.args[1]
            prompt = messages[0]["content"]

            assert "open_source_development" in prompt
            assert "开源软件开发选题" in prompt
            assert "financial trading" not in prompt.lower()

    @pytest.mark.asyncio
    async def test_score_hypothesis_preserves_hkg_and_aux_fields(
        self,
        sample_hypothesis: Hypothesis,
    ) -> None:
        """Scoring must not drop HKG evidence or pre-search metadata."""
        from x_creative.creativity.verify import VerifyModule
        from x_creative.hkg.types import HKGEvidence, HyperedgeSummary, HyperpathEvidence

        verify = VerifyModule()
        with_hkg = sample_hypothesis.model_copy(
            update={
                "hkg_evidence": HKGEvidence(
                    hyperpaths=[
                        HyperpathEvidence(
                            start_node_id="n1",
                            end_node_id="n2",
                            path_rank=1,
                            path_length=1,
                            hyperedges=[
                                HyperedgeSummary(
                                    edge_id="e1",
                                    nodes=["n1", "n2"],
                                    relation="r",
                                    provenance_refs=["doc/chunk"],
                                )
                            ],
                            intermediate_nodes=[],
                        )
                    ]
                ),
                "quick_score": 7.3,
            }
        )

        with patch.object(verify, "_router") as mock_router:
            mock_router.complete = AsyncMock(
                return_value=MagicMock(
                    content='{"divergence": 8, "testability": 8, "rationale": 8, "robustness": 8, "feasibility": 8}'
                )
            )

            scored = await verify.score_hypothesis(with_hkg)

        assert scored.hkg_evidence is not None
        assert scored.hkg_evidence.hyperpaths[0].hyperedges[0].provenance_refs == ["doc/chunk"]
        assert scored.quick_score == 7.3
        assert scored.scores is not None

    @pytest.mark.asyncio
    async def test_score_batch(self, sample_hypothesis: Hypothesis) -> None:
        """Test scoring multiple hypotheses."""
        from x_creative.creativity.verify import VerifyModule

        verify = VerifyModule()
        hypotheses = [sample_hypothesis, sample_hypothesis]

        with patch.object(verify, "_router") as mock_router:
            mock_router.complete = AsyncMock(
                return_value=MagicMock(
                    content="""
{
    "divergence": 8.0,
    "testability": 8.0,
    "rationale": 8.0,
    "robustness": 8.0
}
"""
                )
            )

            scored = await verify.score_batch(hypotheses)

            assert len(scored) == 2
            for h in scored:
                assert h.scores is not None

    @pytest.mark.asyncio
    async def test_score_batch_emits_progress(
        self,
        sample_hypothesis: Hypothesis,
    ) -> None:
        """score_batch should call on_progress for each completed hypothesis."""
        from x_creative.creativity.verify import VerifyModule

        verify = VerifyModule()
        hypotheses = [
            sample_hypothesis.model_copy(update={"id": "h1"}),
            sample_hypothesis.model_copy(update={"id": "h2"}),
            sample_hypothesis.model_copy(update={"id": "h3"}),
        ]

        async def _score(h: Hypothesis) -> Hypothesis:
            return h.model_copy(
                update={
                    "scores": HypothesisScores(
                        divergence=5.0,
                        testability=5.0,
                        rationale=5.0,
                        robustness=5.0,
                        feasibility=5.0,
                    )
                }
            )

        on_progress = AsyncMock()
        with patch.object(verify, "score_hypothesis", AsyncMock(side_effect=_score)):
            scored = await verify.score_batch(
                hypotheses,
                concurrency=2,
                on_progress=on_progress,
            )

        assert len(scored) == 3
        assert on_progress.await_count == 3
        calls = list(on_progress.await_args_list)
        assert {call.args[2] for call in calls} == {"h1", "h2", "h3"}
        assert all(call.args[1] == 3 for call in calls)  # total
        completed = sorted(call.args[0] for call in calls)
        assert completed == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_score_batch_fallback_preserves_hkg_evidence(
        self,
        sample_hypothesis: Hypothesis,
    ) -> None:
        """Batch fallback path should keep evidence fields when scoring fails."""
        from x_creative.creativity.verify import VerifyModule
        from x_creative.hkg.types import HKGEvidence, HyperedgeSummary, HyperpathEvidence

        verify = VerifyModule()
        with_hkg = sample_hypothesis.model_copy(
            update={
                "hkg_evidence": HKGEvidence(
                    hyperpaths=[
                        HyperpathEvidence(
                            start_node_id="n1",
                            end_node_id="n2",
                            path_rank=1,
                            path_length=1,
                            hyperedges=[
                                HyperedgeSummary(
                                    edge_id="e1",
                                    nodes=["n1", "n2"],
                                    relation="r",
                                    provenance_refs=["doc/chunk"],
                                )
                            ],
                            intermediate_nodes=[],
                        )
                    ]
                ),
                "quick_score": 6.1,
            }
        )

        with patch.object(verify, "score_hypothesis", AsyncMock(side_effect=RuntimeError("boom"))):
            scored = await verify.score_batch([with_hkg], concurrency=1)

        assert len(scored) == 1
        assert scored[0].hkg_evidence is not None
        assert scored[0].quick_score == 6.1
        assert scored[0].scores is not None


class TestCreativityEngine:
    """Tests for the main Creativity Engine."""

    @pytest.fixture
    def sample_problem(self) -> ProblemFrame:
        """Create a sample problem frame."""
        return ProblemFrame(
            description="设计一个能实现病毒式传播的开源命令行工具",
        )

    def test_engine_creation(self) -> None:
        """Test creating the creativity engine."""
        from x_creative.creativity.engine import CreativityEngine

        engine = CreativityEngine()
        assert engine is not None

    def test_engine_prepare_search_context_sets_verify_problem_frame(self) -> None:
        """Engine should propagate ProblemFrame to VERIFY for domain-aware prompting."""
        from x_creative.creativity.engine import CreativityEngine

        engine = CreativityEngine()
        problem = ProblemFrame(
            description="为开源软件的爆发式增长提出可检验假说",
            target_domain="open_source_development",
        )
        config = SearchConfig(search_depth=1, search_breadth=1, enable_transform_space=False)

        engine._prepare_search_context(problem, config)

        assert getattr(engine._search, "_problem_frame", None) == problem
        assert getattr(engine._verify, "_problem_frame", None) == problem

    @pytest.mark.asyncio
    async def test_engine_generate_basic(self, sample_problem: ProblemFrame) -> None:
        """Test basic hypothesis generation."""
        from x_creative.creativity.engine import CreativityEngine

        engine = CreativityEngine()

        # Mock the sub-modules
        with patch.object(engine, "_biso") as mock_biso, \
             patch.object(engine, "_search") as mock_search, \
             patch.object(engine, "_verify") as mock_verify, \
             patch.object(engine, "_verify_batch_dual_model", AsyncMock(return_value={})):

            mock_biso.generate_all_analogies = AsyncMock(
                return_value=[
                    Hypothesis(
                        id="raw_1",
                        description="Raw hypothesis",
                        source_domain="test",
                        source_structure="test",
                        analogy_explanation="test",
                        observable="test",
                    )
                ]
            )

            mock_search.run_search = AsyncMock(
                return_value=[
                    Hypothesis(
                        id="searched_1",
                        description="Searched hypothesis",
                        source_domain="test",
                        source_structure="test",
                        analogy_explanation="test",
                        observable="test",
                    )
                ]
            )

            scored_hypothesis = Hypothesis(
                id="scored_1",
                description="Scored hypothesis",
                source_domain="test",
                source_structure="test",
                analogy_explanation="test",
                observable="test",
                scores=HypothesisScores(
                    divergence=8.0,
                    testability=8.0,
                    rationale=8.0,
                    robustness=8.0,
                    feasibility=8.0,
                ),
            )

            mock_verify.score_batch = AsyncMock(return_value=[scored_hypothesis])

            result = await engine.generate(
                problem=sample_problem,
                config=SearchConfig(num_hypotheses=10),
            )

            assert len(result) >= 1
            assert result[0].scores is not None

    def test_engine_sort_hypotheses(self) -> None:
        """Test sorting hypotheses by composite score."""
        from x_creative.creativity.engine import CreativityEngine

        engine = CreativityEngine()

        hypotheses = [
            Hypothesis(
                id="low",
                description="Low score",
                source_domain="t",
                source_structure="t",
                analogy_explanation="t",
                observable="t",
                scores=HypothesisScores(
                    divergence=5.0,
                    testability=5.0,
                    rationale=5.0,
                    robustness=5.0,
                    feasibility=5.0,
                ),
            ),
            Hypothesis(
                id="high",
                description="High score",
                source_domain="t",
                source_structure="t",
                analogy_explanation="t",
                observable="t",
                scores=HypothesisScores(
                    divergence=9.0,
                    testability=9.0,
                    rationale=9.0,
                    robustness=9.0,
                    feasibility=9.0,
                ),
            ),
        ]

        sorted_hyps = engine.sort_by_score(hypotheses)

        assert sorted_hyps[0].id == "high"
        assert sorted_hyps[1].id == "low"


class TestSearchMappingGate:
    """Test that SEARCH gates expansion on mapping_quality."""

    def test_select_for_expansion_excludes_low_or_missing_mapping_quality(self) -> None:
        from x_creative.creativity.search import SearchModule
        from x_creative.core.types import Hypothesis, HypothesisScores

        module = SearchModule(mapping_quality_gate=6.0)

        high_score_low_mapping = Hypothesis(
            id="h1", description="H1", source_domain="d", source_structure="s",
            analogy_explanation="e", observable="o",
            scores=HypothesisScores(divergence=9, testability=9, rationale=9, robustness=9, feasibility=9),
            mapping_quality=3.0,  # Below gate
        )
        moderate_score_good_mapping = Hypothesis(
            id="h2", description="H2", source_domain="d", source_structure="s",
            analogy_explanation="e", observable="o",
            scores=HypothesisScores(divergence=7, testability=7, rationale=7, robustness=7, feasibility=7),
            mapping_quality=8.0,  # Above gate
        )
        no_mapping_quality = Hypothesis(
            id="h3", description="H3", source_domain="d", source_structure="s",
            analogy_explanation="e", observable="o",
            scores=HypothesisScores(divergence=8, testability=8, rationale=8, robustness=8, feasibility=8),
            # No mapping_quality set - should be gated out when gate enabled
        )

        selected = module._select_for_expansion(
            [high_score_low_mapping, moderate_score_good_mapping, no_mapping_quality],
            count=3,
        )

        selected_ids = {h.id for h in selected}
        assert "h1" not in selected_ids  # Gated out
        assert "h2" in selected_ids
        assert "h3" not in selected_ids  # Missing mapping_quality = gated out

    def test_select_for_expansion_no_gate_when_disabled(self) -> None:
        from x_creative.creativity.search import SearchModule
        from x_creative.core.types import Hypothesis, HypothesisScores

        module = SearchModule(mapping_quality_gate=None)

        low_mapping = Hypothesis(
            id="h1", description="H1", source_domain="d", source_structure="s",
            analogy_explanation="e", observable="o",
            scores=HypothesisScores(divergence=9, testability=9, rationale=9, robustness=9, feasibility=9),
            mapping_quality=1.0,  # Low but no gate
        )

        selected = module._select_for_expansion([low_mapping], count=3)
        assert len(selected) == 1  # Not gated out

    def test_select_for_expansion_returns_empty_when_all_gated(self) -> None:
        """If all hypotheses are gated, no candidates should be selected."""
        from x_creative.creativity.search import SearchModule
        from x_creative.core.types import Hypothesis, HypothesisScores

        module = SearchModule(mapping_quality_gate=9.0)

        h1 = Hypothesis(
            id="h1", description="H1", source_domain="d", source_structure="s",
            analogy_explanation="e", observable="o",
            scores=HypothesisScores(divergence=9, testability=9, rationale=9, robustness=9, feasibility=9),
            mapping_quality=5.0,  # Below gate
        )

        selected = module._select_for_expansion([h1], count=3)
        assert selected == []

    def test_pareto_selects_non_dominated_over_dominated(self) -> None:
        """With ParetoArchive, non-dominated hypothesis beats dominated."""
        from x_creative.creativity.search import SearchModule
        from x_creative.creativity.pareto import ParetoArchive
        from x_creative.core.types import Hypothesis, HypothesisScores

        archive = ParetoArchive()
        module = SearchModule(pareto_archive=archive)

        dominant = Hypothesis(
            id="h_dom", description="Dominant", source_domain="d", source_structure="s",
            analogy_explanation="e", observable="o",
            scores=HypothesisScores(divergence=9, testability=9, rationale=9, robustness=9, feasibility=9),
        )
        dominated = Hypothesis(
            id="h_sub", description="Dominated", source_domain="d", source_structure="s",
            analogy_explanation="e", observable="o",
            scores=HypothesisScores(divergence=3, testability=3, rationale=3, robustness=3, feasibility=3),
        )

        selected = module._select_for_expansion([dominated, dominant], count=1)
        assert len(selected) == 1
        assert selected[0].id == "h_dom"

    def test_mapping_gate_applies_before_pareto(self) -> None:
        """Mapping quality gate should filter before Pareto selection."""
        from x_creative.creativity.search import SearchModule
        from x_creative.creativity.pareto import ParetoArchive
        from x_creative.core.types import Hypothesis, HypothesisScores

        archive = ParetoArchive()
        module = SearchModule(mapping_quality_gate=6.0, pareto_archive=archive)

        # High scores but low mapping quality → gated out
        gated_out = Hypothesis(
            id="h_gated", description="Gated", source_domain="d", source_structure="s",
            analogy_explanation="e", observable="o",
            scores=HypothesisScores(divergence=10, testability=10, rationale=10, robustness=10, feasibility=10),
            mapping_quality=3.0,
        )
        # Lower scores but good mapping quality → passes gate
        passes_gate = Hypothesis(
            id="h_pass", description="Passes", source_domain="d", source_structure="s",
            analogy_explanation="e", observable="o",
            scores=HypothesisScores(divergence=6, testability=6, rationale=6, robustness=6, feasibility=6),
            mapping_quality=8.0,
        )

        selected = module._select_for_expansion([gated_out, passes_gate], count=1)
        assert selected[0].id == "h_pass"
