"""Tests for HKG integration into SearchModule."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from x_creative.core.types import Hypothesis, ProblemFrame, SearchConfig
from x_creative.creativity.search import SearchModule
from x_creative.hkg.types import HKGParams


class TestSearchModuleHKGInit:
    def test_init_without_hkg(self) -> None:
        from x_creative.creativity.search import SearchModule
        sm = SearchModule()
        assert sm._hkg_store is None
        assert sm._hkg_matcher is None
        assert sm._hkg_params is None
        assert sm._problem_frame is None

    def test_init_with_hkg(self) -> None:
        from x_creative.creativity.search import SearchModule
        from x_creative.hkg.store import HypergraphStore
        from x_creative.hkg.matcher import NodeMatcher
        from x_creative.hkg.types import HKGParams
        store = HypergraphStore()
        matcher = NodeMatcher(store)
        params = HKGParams()
        sm = SearchModule(hkg_store=store, hkg_matcher=matcher, hkg_params=params)
        assert sm._hkg_store is store
        assert sm._hkg_matcher is matcher
        assert sm._hkg_params is params


class TestSearchModuleHKGDisabled:
    """When HKG is not configured, search should behave identically."""

    @pytest.mark.asyncio
    async def test_search_without_hkg_unchanged(self) -> None:
        from x_creative.creativity.search import SearchModule
        sm = SearchModule()
        sm._router = MagicMock()
        sm._router.complete = AsyncMock(return_value=MagicMock(content="[]"))

        hyps = [Hypothesis(
            id="h1", description="test", source_domain="d",
            source_structure="s", analogy_explanation="a", observable="o",
        )]
        config = SearchConfig(search_depth=1, search_breadth=1)
        result = await sm.run_search(hyps, config)
        assert len(result) >= 1


class TestSearchModuleHKGEvents:
    @pytest.mark.asyncio
    async def test_hkg_path_not_found_event_callback(self) -> None:
        from x_creative.creativity.search import SearchModule
        from x_creative.hkg.matcher import NodeMatcher
        from x_creative.hkg.store import HypergraphStore
        from x_creative.hkg.types import HKGNode, HKGParams

        store = HypergraphStore()
        store.add_node(HKGNode(node_id="n1", name="entropy"))
        matcher = NodeMatcher(store=store)

        sm = SearchModule(
            hkg_store=store,
            hkg_matcher=matcher,
            hkg_params=HKGParams(top_n_hypotheses=1),
            problem_frame=ProblemFrame(description="unmatched target phrase"),
        )
        sm._router = MagicMock()
        sm._router.complete = AsyncMock(return_value=MagicMock(content="[]"))

        hyps = [Hypothesis(
            id="h1",
            description="unmatched source phrase",
            source_domain="d",
            source_structure="s",
            analogy_explanation="a",
            observable="o",
        )]

        events: list[tuple[str, dict]] = []

        async def on_hkg_event(event_type: str, payload: dict) -> None:
            events.append((event_type, payload))

        await sm.run_search(
            hyps,
            SearchConfig(search_depth=1, search_breadth=1),
            on_hkg_event=on_hkg_event,
        )

        assert any(event_type == "hkg_path_not_found" for event_type, _ in events)


class TestSearchPassesCacheToHKGExpand:
    """#8: HKG cache must be instantiated and passed to hyperpath_expand."""

    def test_search_module_creates_cache_when_hkg_enabled(self) -> None:
        from x_creative.hkg.cache import TraversalCache
        from x_creative.creativity.search import SearchModule
        from x_creative.hkg.types import HKGParams
        store = MagicMock()
        module = SearchModule(
            router=MagicMock(),
            hkg_store=store,
            hkg_matcher=MagicMock(),
            hkg_params=HKGParams(),
            problem_frame=ProblemFrame(description="test"),
        )
        assert isinstance(module._hkg_cache, TraversalCache)

    def test_search_module_no_cache_without_hkg(self) -> None:
        from x_creative.creativity.search import SearchModule
        module = SearchModule(router=MagicMock())
        assert module._hkg_cache is None

    @pytest.mark.asyncio
    async def test_cache_passed_to_hyperpath_expand_call(self) -> None:
        from x_creative.creativity.search import SearchModule
        from x_creative.hkg.types import HKGParams
        store = MagicMock()
        matcher = MagicMock()
        params = HKGParams(top_n_hypotheses=1)
        problem = ProblemFrame(description="test problem")
        router = MagicMock()
        router.complete = AsyncMock(return_value=MagicMock(content="[]"))

        module = SearchModule(
            router=router, hkg_store=store, hkg_matcher=matcher,
            hkg_params=params, problem_frame=problem,
        )
        hyp = Hypothesis(
            id="h1", description="test", source_domain="d", source_structure="s",
            analogy_explanation="a", observable="o",
        )
        config = SearchConfig(search_breadth=1)

        with patch("x_creative.hkg.expand.hyperpath_expand", new_callable=AsyncMock) as mock_expand:
            mock_expand.return_value = []
            await module.search_iteration([hyp], config)
            mock_expand.assert_called_once()
            _, kwargs = mock_expand.call_args
            assert kwargs.get("cache") is not None


class TestSearchHyperbridgeIntegration:
    """#1: hyperbridge must be called in SEARCH when enabled."""

    @pytest.mark.asyncio
    async def test_search_hyperbridge_called_when_enabled(self) -> None:
        store = MagicMock()
        matcher = MagicMock()
        params = HKGParams(top_n_hypotheses=2)
        problem = ProblemFrame(description="test")
        router = MagicMock()
        router.complete = AsyncMock(return_value=MagicMock(content="[]"))

        module = SearchModule(
            router=router, hkg_store=store, hkg_matcher=matcher,
            hkg_params=params, problem_frame=problem,
            enable_hyperbridge=True,
        )
        hyps = [
            Hypothesis(id="h1", description="d1", source_domain="domA",
                       source_structure="s", analogy_explanation="a", observable="o"),
            Hypothesis(id="h2", description="d2", source_domain="domB",
                       source_structure="s", analogy_explanation="a", observable="o"),
        ]
        config = SearchConfig(search_breadth=2)

        with patch("x_creative.hkg.expand.hyperpath_expand", new_callable=AsyncMock) as mock_hp, \
             patch("x_creative.hkg.expand.hyperbridge", new_callable=AsyncMock) as mock_hb:
            mock_hp.return_value = []
            mock_hb.return_value = []
            await module.search_iteration(hyps, config)
            mock_hb.assert_called()

    @pytest.mark.asyncio
    async def test_search_hyperbridge_skipped_when_disabled(self) -> None:
        store = MagicMock()
        matcher = MagicMock()
        params = HKGParams(top_n_hypotheses=2)
        problem = ProblemFrame(description="test")
        router = MagicMock()
        router.complete = AsyncMock(return_value=MagicMock(content="[]"))

        module = SearchModule(
            router=router, hkg_store=store, hkg_matcher=matcher,
            hkg_params=params, problem_frame=problem,
            enable_hyperbridge=False,
        )
        hyps = [
            Hypothesis(id="h1", description="d1", source_domain="domA",
                       source_structure="s", analogy_explanation="a", observable="o"),
            Hypothesis(id="h2", description="d2", source_domain="domB",
                       source_structure="s", analogy_explanation="a", observable="o"),
        ]
        config = SearchConfig(search_breadth=2)

        with patch("x_creative.hkg.expand.hyperpath_expand", new_callable=AsyncMock) as mock_hp, \
             patch("x_creative.hkg.expand.hyperbridge", new_callable=AsyncMock) as mock_hb:
            mock_hp.return_value = []
            await module.search_iteration(hyps, config)
            mock_hb.assert_not_called()

    @pytest.mark.asyncio
    async def test_search_hyperbridge_pairs_from_topn(self) -> None:
        store = MagicMock()
        matcher = MagicMock()
        params = HKGParams(top_n_hypotheses=4)
        problem = ProblemFrame(description="test")
        router = MagicMock()
        router.complete = AsyncMock(return_value=MagicMock(content="[]"))

        module = SearchModule(
            router=router, hkg_store=store, hkg_matcher=matcher,
            hkg_params=params, problem_frame=problem,
            enable_hyperbridge=True,
        )
        hyps = [
            Hypothesis(id=f"h{i}", description=f"d{i}", source_domain=f"dom{i}",
                       source_structure="s", analogy_explanation="a", observable="o")
            for i in range(4)
        ]
        config = SearchConfig(search_breadth=4)

        with patch("x_creative.hkg.expand.hyperpath_expand", new_callable=AsyncMock) as mock_hp, \
             patch("x_creative.hkg.expand.hyperbridge", new_callable=AsyncMock) as mock_hb:
            mock_hp.return_value = []
            mock_hb.return_value = []
            await module.search_iteration(hyps, config)
            assert mock_hb.call_count == 2


class TestSelectForExpansionScoring:
    """#2: _select_for_expansion must prioritize scored hypotheses."""

    def _make_hyp(self, hyp_id: str) -> Hypothesis:
        return Hypothesis(
            id=hyp_id, description="test", source_domain="d",
            source_structure="s", analogy_explanation="a", observable="o",
        )

    def test_select_uses_composite_score(self) -> None:
        from x_creative.core.types import HypothesisScores
        module = SearchModule()
        low = self._make_hyp("low")
        low.scores = HypothesisScores(
            divergence=3.0, testability=3.0, rationale=3.0,
            robustness=3.0, feasibility=3.0,
        )
        high = self._make_hyp("high")
        high.scores = HypothesisScores(
            divergence=9.0, testability=9.0, rationale=9.0,
            robustness=9.0, feasibility=9.0,
        )
        result = module._select_for_expansion([low, high], count=1)
        assert result[0].id == "high"

    def test_select_uses_quick_score_when_no_full_scores(self) -> None:
        module = SearchModule()
        low = self._make_hyp("low")
        low.quick_score = 3.0
        high = self._make_hyp("high")
        high.quick_score = 9.0
        result = module._select_for_expansion([low, high], count=1)
        assert result[0].id == "high"

    def test_select_prefers_composite_over_quick_score(self) -> None:
        from x_creative.core.types import HypothesisScores
        module = SearchModule()
        scored = self._make_hyp("scored")
        scored.scores = HypothesisScores(
            divergence=5.0, testability=5.0, rationale=5.0,
            robustness=5.0, feasibility=5.0,
        )
        scored.quick_score = 1.0
        quick_only = self._make_hyp("quick")
        quick_only.quick_score = 9.0
        result = module._select_for_expansion([quick_only, scored], count=2)
        assert result[0].id == "scored"


class TestSearchPreScoring:
    """#2: run_search should pre-score initial hypotheses."""

    @pytest.mark.asyncio
    async def test_search_prescoring_sets_full_scores(self) -> None:
        from x_creative.core.types import HypothesisScores

        router = MagicMock()
        router.complete = AsyncMock(return_value=MagicMock(content="[]"))
        module = SearchModule(router=router)
        hyps = [
            Hypothesis(id="h1", description="test1", source_domain="d",
                       source_structure="s", analogy_explanation="a", observable="o"),
            Hypothesis(id="h2", description="test2", source_domain="d",
                       source_structure="s", analogy_explanation="a", observable="o"),
        ]
        config = SearchConfig(search_depth=0)
        scored = [
            hyps[0].model_copy(update={"scores": HypothesisScores(
                divergence=8.0, testability=8.0, rationale=8.0, robustness=8.0, feasibility=8.0
            )}),
            hyps[1].model_copy(update={"scores": HypothesisScores(
                divergence=6.0, testability=6.0, rationale=6.0, robustness=6.0, feasibility=6.0
            )}),
        ]
        with patch("x_creative.creativity.verify.VerifyModule.score_batch", new_callable=AsyncMock) as mock_score_batch:
            mock_score_batch.return_value = scored
            result = await module.run_search(hyps, config)

        assert mock_score_batch.await_count == 1
        assert result[0].scores is not None
        assert result[1].scores is not None

    @pytest.mark.asyncio
    async def test_search_prescoring_falls_back_to_quick_score_on_failure(self) -> None:
        router = MagicMock()
        router.complete = AsyncMock(return_value=MagicMock(
            content='[{"id": "h1", "score": 7.5}, {"id": "h2", "score": 5.0}]'
        ))
        module = SearchModule(router=router)
        hyps = [
            Hypothesis(id="h1", description="test1", source_domain="d",
                       source_structure="s", analogy_explanation="a", observable="o"),
            Hypothesis(id="h2", description="test2", source_domain="d",
                       source_structure="s", analogy_explanation="a", observable="o"),
        ]
        config = SearchConfig(search_depth=0)
        with patch("x_creative.creativity.verify.VerifyModule.score_batch", new_callable=AsyncMock) as mock_score_batch:
            mock_score_batch.side_effect = RuntimeError("boom")
            result = await module.run_search(hyps, config)

        assert mock_score_batch.await_count == 1
        assert result[0].quick_score is not None
        assert result[1].quick_score is not None


class TestQuickScoreInheritance:
    """#2: Child hypotheses must inherit parent's quick_score (decayed 0.9x)."""

    def test_parse_expansions_inherits_quick_score(self) -> None:
        """_parse_expansions should set child quick_score = parent * 0.9."""
        import json
        module = SearchModule()
        parent = Hypothesis(
            id="parent1", description="parent", source_domain="d",
            source_structure="s", analogy_explanation="a", observable="o",
            quick_score=8.0,
        )
        content = json.dumps([{
            "description": "child desc",
            "analogy_explanation": "child analogy",
            "observable": "child obs",
        }])
        children = module._parse_expansions(content, parent)
        assert len(children) == 1
        assert children[0].quick_score == pytest.approx(7.2)  # 8.0 * 0.9
        assert children[0].parent_id == "parent1"

    def test_parse_expansions_no_quick_score_when_parent_none(self) -> None:
        """If parent has no quick_score, child should also have None."""
        import json
        module = SearchModule()
        parent = Hypothesis(
            id="parent2", description="parent", source_domain="d",
            source_structure="s", analogy_explanation="a", observable="o",
        )
        content = json.dumps([{
            "description": "child desc",
            "analogy_explanation": "child analogy",
            "observable": "child obs",
        }])
        children = module._parse_expansions(content, parent)
        assert len(children) == 1
        assert children[0].quick_score is None

    def test_parse_expansions_skips_empty_observable(self) -> None:
        """SEARCH text expansions must keep only hypotheses with non-empty observable."""
        import json

        module = SearchModule()
        parent = Hypothesis(
            id="parent3", description="parent", source_domain="d",
            source_structure="s", analogy_explanation="a", observable="o",
        )
        content = json.dumps([
            {
                "description": "bad child",
                "analogy_explanation": "missing observable",
                "observable": "   ",
            },
            {
                "description": "good child",
                "analogy_explanation": "has observable",
                "observable": "child_obs",
            },
        ])
        children = module._parse_expansions(content, parent)
        assert len(children) == 1
        assert children[0].observable == "child_obs"


class TestSearchCombineIntersection:
    """SEARCH combine should actually consume two hypotheses."""

    @pytest.mark.asyncio
    async def test_combine_hypotheses_uses_two_inputs(self) -> None:
        router = MagicMock()
        router.complete = AsyncMock(
            return_value=MagicMock(
                content=(
                    '[{"description":"组合假说","analogy_explanation":"A与B交汇机制",'
                    '"observable":"x+y","expansion_type":"combine"}]'
                )
            )
        )
        module = SearchModule(router=router)
        hyp_a = Hypothesis(
            id="h_a",
            description="Hypothesis A",
            source_domain="domain_a",
            source_structure="structure_a",
            analogy_explanation="A explanation",
            observable="x",
        )
        hyp_b = Hypothesis(
            id="h_b",
            description="Hypothesis B",
            source_domain="domain_b",
            source_structure="structure_b",
            analogy_explanation="B explanation",
            observable="y",
        )

        combined = await module.combine_hypotheses(hyp_a, hyp_b, max_expansions=1)

        assert len(combined) == 1
        assert combined[0].expansion_type == "combine"
        assert "[combined_with:h_b]" in combined[0].analogy_explanation

        prompt = router.complete.call_args.kwargs["messages"][0]["content"]
        assert "Hypothesis A" in prompt
        assert "Hypothesis B" in prompt
        assert hyp_a.description in prompt
        assert hyp_b.description in prompt


class TestSearchTopNIteration:
    """Each round should allow newly added hypotheses to enter top-N."""

    @pytest.mark.asyncio
    async def test_new_hypothesis_is_rescored_and_selected_next_round(self) -> None:
        from x_creative.core.types import HypothesisScores

        module = SearchModule(router=MagicMock())

        base_low = Hypothesis(
            id="base_low",
            description="low",
            source_domain="d",
            source_structure="s",
            analogy_explanation="a",
            observable="o",
            scores=HypothesisScores(
                divergence=4.0, testability=4.0, rationale=4.0, robustness=4.0, feasibility=4.0
            ),
        )
        base_high = Hypothesis(
            id="base_high",
            description="high",
            source_domain="d",
            source_structure="s",
            analogy_explanation="a",
            observable="o",
            scores=HypothesisScores(
                divergence=8.0, testability=8.0, rationale=8.0, robustness=8.0, feasibility=8.0
            ),
        )

        selected_each_round: list[list[str]] = []

        async def fake_search_iteration(
            hypotheses,  # noqa: ANN001
            config,  # noqa: ANN001
            on_hkg_event=None,  # noqa: ANN001
        ):
            selected_each_round.append([h.id for h in hypotheses])
            if len(selected_each_round) == 1:
                return hypotheses + [
                    Hypothesis(
                        id="new_child",
                        description="new child",
                        source_domain="d",
                        source_structure="s",
                        analogy_explanation="a",
                        observable="o",
                    )
                ]
            return list(hypotheses)

        async def fake_score_batch(hypotheses):  # noqa: ANN001
            scored: list[Hypothesis] = []
            for h in hypotheses:
                if h.id == "new_child":
                    scores = HypothesisScores(
                        divergence=9.5,
                        testability=9.5,
                        rationale=9.5,
                        robustness=9.5,
                        feasibility=9.5,
                    )
                    scored.append(h.model_copy(update={"scores": scores}))
                else:
                    scored.append(h)
            return scored

        with patch.object(module, "search_iteration", side_effect=fake_search_iteration), \
             patch("x_creative.creativity.verify.VerifyModule.score_batch", new_callable=AsyncMock) as mock_score_batch:
            mock_score_batch.side_effect = fake_score_batch
            await module.run_search(
                initial_hypotheses=[base_low, base_high],
                config=SearchConfig(search_depth=2, search_breadth=1),
            )

        assert selected_each_round[0] == ["base_high"]
        assert selected_each_round[1] == ["new_child"]
        assert mock_score_batch.await_count >= 1


class TestHKGCallBudgetCaps:
    """Theory §6.8.2: HKG call count must respect round budget cap."""

    @pytest.mark.asyncio
    async def test_hkg_calls_capped_by_depth_times_top_n(self) -> None:
        from x_creative.core.types import HypothesisScores
        from x_creative.hkg.matcher import NodeMatcher
        from x_creative.hkg.store import HypergraphStore
        from x_creative.hkg.types import HKGNode, HKGParams

        store = HypergraphStore()
        store.add_node(HKGNode(node_id="n1", name="entropy"))
        store.add_node(HKGNode(node_id="n2", name="volatility"))
        matcher = NodeMatcher(store)

        router = MagicMock()
        router.complete = AsyncMock(return_value=MagicMock(content="[]"))
        params = HKGParams(top_n_hypotheses=2)

        module = SearchModule(
            router=router,
            hkg_store=store,
            hkg_matcher=matcher,
            hkg_params=params,
            problem_frame=ProblemFrame(description="volatility"),
            enable_hyperbridge=False,
        )

        hypotheses = [
            Hypothesis(
                id=f"h{i}",
                description=f"hypothesis {i}",
                source_domain="d",
                source_structure="s",
                analogy_explanation="a",
                observable="o",
                scores=HypothesisScores(
                    divergence=8.0 - i * 0.1,
                    testability=8.0 - i * 0.1,
                    rationale=8.0 - i * 0.1,
                    robustness=8.0 - i * 0.1,
                    feasibility=8.0 - i * 0.1,
                ),
            )
            for i in range(5)
        ]

        config = SearchConfig(
            search_depth=3,
            search_breadth=5,
            enable_combination=False,
            enable_opposition=False,
        )

        with patch.object(module, "expand_hypothesis", AsyncMock(return_value=[])), \
             patch("x_creative.hkg.expand.hyperpath_expand", new_callable=AsyncMock) as mock_expand:
            mock_expand.return_value = []
            await module.run_search(hypotheses, config)

        # 3 rounds × top_n_hypotheses(2) = 6 max calls.
        assert mock_expand.await_count == 6
