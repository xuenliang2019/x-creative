"""Tests for SAGA detectors/auditors/evaluators/memory components."""

import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from x_creative.saga.auditors import DomainConstraintAuditor
from x_creative.saga.budget import AllocationStrategy, BudgetPolicy, CognitiveBudget
from x_creative.saga.detectors import (
    DimensionCollinearityDetector,
    ScoreCompressionDetector,
    ShallowRewriteDetector,
    SourceDomainBiasDetector,
    StructureCollapseDetector,
)
from x_creative.saga.evaluation import AdversarialChallengeEvaluator, PatternMemoryEvaluator
from x_creative.saga.events import DirectiveType, EventBus, EventType, FastAgentEvent
from x_creative.saga.memory import PatternMemory
from x_creative.saga.slow_agent import SlowAgent, BaseEvaluator
from x_creative.saga.state import CognitionAlert, SharedCognitionState


@pytest.mark.asyncio
async def test_score_compression_detector_raises_alert() -> None:
    detector = ScoreCompressionDetector(std_threshold=0.8, min_hypotheses=3)
    event = FastAgentEvent(
        event_type=EventType.VERIFY_BATCH_SCORED,
        stage="verify",
        metrics={"score_std": 0.4, "hypothesis_count": 8},
    )
    alerts = await detector.detect(event, SharedCognitionState())
    assert len(alerts) == 1
    assert alerts[0].alert_type == "score_compression"


@pytest.mark.asyncio
async def test_structure_collapse_detector_raises_alert() -> None:
    detector = StructureCollapseDetector(min_hypotheses=5)
    event = FastAgentEvent(
        event_type=EventType.SEARCH_COMPLETED,
        stage="search",
        metrics={"hypothesis_count": 10, "unique_structure_count": 1},
    )
    alerts = await detector.detect(event, SharedCognitionState())
    assert len(alerts) == 1
    assert alerts[0].alert_type == "structure_collapse"


@pytest.mark.asyncio
async def test_dimension_collinearity_detector_raises_alert() -> None:
    detector = DimensionCollinearityDetector(corr_threshold=0.7, min_hypotheses=4)
    state = SharedCognitionState(
        hypotheses_pool=[
            {
                "scores": {
                    "divergence": 6.0 + i * 0.1,
                    "testability": 6.0 + i * 0.1,
                    "rationale": 6.0 + i * 0.1,
                    "robustness": 5.0 + i * 0.1,
                    "feasibility": 5.0 + i * 0.1,
                }
            }
            for i in range(6)
        ]
    )
    alerts = await detector.detect(
        FastAgentEvent(event_type=EventType.VERIFY_COMPLETED, stage="verify"),
        state,
    )
    assert len(alerts) == 1
    assert alerts[0].alert_type == "dimension_collinearity"


@pytest.mark.asyncio
async def test_source_domain_bias_detector_raises_alert() -> None:
    detector = SourceDomainBiasDetector(f_threshold=1.2, min_domains=2, min_samples=6)
    state = SharedCognitionState(
        hypotheses_pool=[
            {"id": "a1", "source_domain": "A", "final_score": 9.2},
            {"id": "a2", "source_domain": "A", "final_score": 9.1},
            {"id": "a3", "source_domain": "A", "final_score": 9.0},
            {"id": "b1", "source_domain": "B", "final_score": 4.0},
            {"id": "b2", "source_domain": "B", "final_score": 4.1},
            {"id": "b3", "source_domain": "B", "final_score": 4.2},
        ]
    )
    alerts = await detector.detect(
        FastAgentEvent(event_type=EventType.VERIFY_BATCH_SCORED, stage="verify"),
        state,
    )
    assert len(alerts) == 1
    assert alerts[0].alert_type == "source_domain_bias"
    assert "anova_f" in alerts[0].evidence


@pytest.mark.asyncio
async def test_source_domain_bias_detector_avoids_false_positive_under_high_variance() -> None:
    detector = SourceDomainBiasDetector(f_threshold=1.2, min_domains=3, min_samples=9)
    state = SharedCognitionState(
        hypotheses_pool=[
            {"id": "a1", "source_domain": "A", "final_score": 0.0},
            {"id": "a2", "source_domain": "A", "final_score": 16.0},
            {"id": "a3", "source_domain": "A", "final_score": 0.0},
            {"id": "a4", "source_domain": "A", "final_score": 16.0},
            {"id": "b1", "source_domain": "B", "final_score": 0.0},
            {"id": "b2", "source_domain": "B", "final_score": 12.0},
            {"id": "b3", "source_domain": "B", "final_score": 0.0},
            {"id": "b4", "source_domain": "B", "final_score": 12.0},
            {"id": "c1", "source_domain": "C", "final_score": 0.0},
            {"id": "c2", "source_domain": "C", "final_score": 8.0},
            {"id": "c3", "source_domain": "C", "final_score": 0.0},
            {"id": "c4", "source_domain": "C", "final_score": 8.0},
        ]
    )
    alerts = await detector.detect(
        FastAgentEvent(event_type=EventType.VERIFY_BATCH_SCORED, stage="verify"),
        state,
    )
    assert alerts == []


@pytest.mark.asyncio
async def test_shallow_rewrite_detector_raises_alert() -> None:
    detector = ShallowRewriteDetector(
        similarity_threshold=0.85,
        score_gap_threshold=2.0,
        min_hypotheses=2,
    )
    state = SharedCognitionState(
        hypotheses_pool=[
            {
                "id": "h1",
                "description": "Use queue pressure for ticket triage",
                "final_score": 8.5,
                "parent_id": "p1",
            },
            {
                "id": "h2",
                "description": "Use queue pressure for support ticket triage",
                "final_score": 5.8,
                "parent_id": "p1",
            },
        ]
    )
    alerts = await detector.detect(
        FastAgentEvent(event_type=EventType.VERIFY_COMPLETED, stage="verify"),
        state,
    )
    assert len(alerts) == 1
    assert alerts[0].alert_type == "shallow_rewrite"


@pytest.mark.asyncio
async def test_domain_constraint_auditor_flags_lookahead() -> None:
    auditor = DomainConstraintAuditor()
    state = SharedCognitionState(
        target_domain_id="open_source_development",
        hypotheses_pool=[
            {
                "id": "h1",
                "description": "Use future return at t+1 as signal",
                "observable": "future_ret_t+1",
                "analogy_explanation": "look-ahead heavy signal",
            }
        ],
    )
    alerts = [
        CognitionAlert(
            alert_type="score_compression",
            severity="critical",
            description="compressed",
        )
    ]
    directives = await auditor.audit(
        FastAgentEvent(event_type=EventType.VERIFY_BATCH_SCORED, stage="verify"),
        alerts,
        state,
    )
    assert any(d.directive_type == DirectiveType.FLAG_HYPOTHESIS for d in directives)


@pytest.mark.asyncio
async def test_domain_constraint_auditor_flags_plugin_anti_pattern() -> None:
    auditor = DomainConstraintAuditor()
    state = SharedCognitionState(
        target_domain_id="open_source_development",
        hypotheses_pool=[
            {
                "id": "h_ap",
                "description": "该方案文档缺失，只写给作者自己看，没有快速上手指南和示例，新用户完全无法跑起来",
                "observable": "onboarding_time",
                "analogy_explanation": "poor documentation blocks adoption",
            }
        ],
    )
    alerts = [
        CognitionAlert(
            alert_type="score_compression",
            severity="critical",
            description="compressed",
        )
    ]

    directives = await auditor.audit(
        FastAgentEvent(event_type=EventType.VERIFY_BATCH_SCORED, stage="verify"),
        alerts,
        state,
    )

    flagged = [d for d in directives if d.directive_type == DirectiveType.FLAG_HYPOTHESIS]
    assert flagged
    violations = flagged[0].payload.get("violations", [])
    assert any(str(v).startswith("anti_pattern:") for v in violations)


@pytest.mark.asyncio
async def test_domain_constraint_auditor_flags_stale_idea() -> None:
    auditor = DomainConstraintAuditor()
    state = SharedCognitionState(
        target_domain_id="open_source_development",
        hypotheses_pool=[
            {
                "id": "h_stale",
                "description": "将项目上传到GitHub并编写完善的README文档和MIT许可证，通过开源发布本身来驱动用户自然增长",
                "observable": "github_star_count",
                "analogy_explanation": "open source release as growth strategy",
            }
        ],
    )
    alerts = [
        CognitionAlert(
            alert_type="score_compression",
            severity="critical",
            description="compressed",
        )
    ]

    directives = await auditor.audit(
        FastAgentEvent(event_type=EventType.VERIFY_BATCH_SCORED, stage="verify"),
        alerts,
        state,
    )

    flagged = [d for d in directives if d.directive_type == DirectiveType.FLAG_HYPOTHESIS]
    assert flagged
    violations = flagged[0].payload.get("violations", [])
    assert any(str(v).startswith("stale_idea:") for v in violations)


@pytest.mark.asyncio
async def test_domain_constraint_auditor_flags_semantic_stale_idea_variant() -> None:
    auditor = DomainConstraintAuditor()
    state = SharedCognitionState(
        target_domain_id="open_source_development",
        hypotheses_pool=[
            {
                "id": "h_stale_semantic",
                "description": "在README中请求用户点Star并在各技术社区刷榜提高项目可见度，以Star数量增长作为核心增长指标",
                "observable": "star_growth_rate",
                "analogy_explanation": "visibility metrics as growth proxy",
            }
        ],
    )
    alerts = [
        CognitionAlert(
            alert_type="score_compression",
            severity="critical",
            description="compressed",
        )
    ]

    directives = await auditor.audit(
        FastAgentEvent(event_type=EventType.VERIFY_BATCH_SCORED, stage="verify"),
        alerts,
        state,
    )
    flagged = [d for d in directives if d.directive_type == DirectiveType.FLAG_HYPOTHESIS]
    assert flagged
    violations = flagged[0].payload.get("violations", [])
    assert any(str(v).startswith("stale_idea:") for v in violations)


@pytest.mark.asyncio
async def test_adversarial_evaluator_emits_challenge() -> None:
    evaluator = AdversarialChallengeEvaluator(top_n=2)
    state = SharedCognitionState(
        hypotheses_pool=[
            {"id": "h1", "final_score": 8.0},
            {"id": "h2", "final_score": 7.0},
        ],
        active_alerts=[
            CognitionAlert(
                alert_type="score_compression",
                severity="warning",
                description="compressed",
            )
        ],
    )
    directives = await evaluator.evaluate(
        FastAgentEvent(event_type=EventType.VERIFY_COMPLETED, stage="verify"),
        state,
        CognitiveBudget(total_budget=100.0),
    )
    challenge_types = {
        str(d.payload.get("challenge_type"))
        for d in directives
        if d.directive_type == DirectiveType.INJECT_CHALLENGE
    }
    assert challenge_types == {
        "counterexample",
        "simplification_test",
        "causal_reversal",
        "operationalization",
    }
    assert any(d.directive_type == DirectiveType.RESCORE_BATCH for d in directives)


@pytest.mark.asyncio
async def test_adversarial_evaluator_calls_router_when_available() -> None:
    class _DummyCompletion:
        def __init__(self, content: str) -> None:
            self.content = content
            self.prompt_tokens = 5
            self.completion_tokens = 10
            self.finish_reason = "stop"
            self.model = "dummy/model"

    class _DummyRouter:
        def __init__(self) -> None:
            self.calls = 0

        async def complete(self, task, messages, temperature=None, max_tokens=None, **kwargs):  # noqa: ANN001, ANN002
            self.calls += 1
            assert task == "saga_adversarial"
            return _DummyCompletion(
                '{"challenge_type":"counterexample","severity":"high",'
                '"attack_question":"What breaks this claim?","risk_summary":"high risk"}'
            )

    router = _DummyRouter()
    evaluator = AdversarialChallengeEvaluator(top_n=1, router=router)
    state = SharedCognitionState(hypotheses_pool=[{"id": "h1", "final_score": 8.3}])
    directives = await evaluator.evaluate(
        FastAgentEvent(event_type=EventType.VERIFY_COMPLETED, stage="verify"),
        state,
        CognitiveBudget(total_budget=20.0),
    )
    assert router.calls == 1
    assert any(d.directive_type == DirectiveType.INJECT_CHALLENGE for d in directives)
    assert any(d.directive_type == DirectiveType.RESCORE_BATCH for d in directives)


@pytest.mark.asyncio
async def test_pattern_memory_evaluator_flags_repeats() -> None:
    memory = PatternMemory()
    evaluator = PatternMemoryEvaluator(memory=memory)
    state = SharedCognitionState(hypotheses_pool=[{"id": "h1", "description": "unique", "observable": "x"}])

    # First pass only records
    first = await evaluator.evaluate(
        FastAgentEvent(event_type=EventType.SEARCH_COMPLETED, stage="search"),
        state,
        CognitiveBudget(),
    )
    assert first == []

    # Second pass should detect repeats
    state.hypotheses_pool = [
        {"id": "h2", "description": "unique", "observable": "x"},
    ]
    second = await evaluator.evaluate(
        FastAgentEvent(event_type=EventType.SEARCH_COMPLETED, stage="search"),
        state,
        CognitiveBudget(),
    )
    assert any(d.directive_type == DirectiveType.FLAG_HYPOTHESIS for d in second)


@pytest.mark.asyncio
async def test_pattern_memory_persists_across_instances(tmp_path) -> None:  # noqa: ANN001
    path = tmp_path / "pattern_memory.json"
    memory_a = PatternMemory(storage_path=path)
    await memory_a.record_batch([{"description": "repeat me", "observable": "x"}])

    memory_b = PatternMemory(storage_path=path)
    fp = await memory_b.fingerprint("repeat me", "x")
    assert memory_b.get_count(fp) == 1


@pytest.mark.asyncio
async def test_adversarial_evaluates_all_topn() -> None:
    """#3: Adversarial evaluator must generate challenges for ALL top-N."""
    evaluator = AdversarialChallengeEvaluator(top_n=3)
    state = SharedCognitionState(
        hypotheses_pool=[
            {"id": f"h{i}", "final_score": 9.0 - i} for i in range(5)
        ],
    )
    directives = await evaluator.evaluate(
        FastAgentEvent(event_type=EventType.VERIFY_COMPLETED, stage="verify"),
        state,
        CognitiveBudget(total_budget=100.0),
    )
    challenge_directives = [
        d for d in directives if d.directive_type == DirectiveType.INJECT_CHALLENGE
    ]
    challenged_ids = {d.payload.get("hypothesis_id") for d in challenge_directives}
    assert "h0" in challenged_ids
    assert "h1" in challenged_ids
    assert "h2" in challenged_ids
    # 3 hypotheses × 4 challenge types = 12 challenges
    assert len(challenge_directives) == 12


@pytest.mark.asyncio
async def test_adversarial_each_candidate_independent() -> None:
    """#3: Each candidate's challenges reference its own hypothesis_id."""
    evaluator = AdversarialChallengeEvaluator(top_n=2)
    state = SharedCognitionState(
        hypotheses_pool=[
            {"id": "h_alpha", "final_score": 9.0},
            {"id": "h_beta", "final_score": 8.0},
        ],
    )
    directives = await evaluator.evaluate(
        FastAgentEvent(event_type=EventType.VERIFY_COMPLETED, stage="verify"),
        state,
        CognitiveBudget(total_budget=100.0),
    )
    challenges = [d for d in directives if d.directive_type == DirectiveType.INJECT_CHALLENGE]
    alpha = [d for d in challenges if d.payload.get("hypothesis_id") == "h_alpha"]
    beta = [d for d in challenges if d.payload.get("hypothesis_id") == "h_beta"]
    assert len(alpha) == 4
    assert len(beta) == 4


@pytest.mark.asyncio
async def test_adversarial_topn_falls_back_to_composite_when_final_missing() -> None:
    """#15: checkpoint top-N should remain stable before final_score exists."""
    evaluator = AdversarialChallengeEvaluator(top_n=2)
    state = SharedCognitionState(
        hypotheses_pool=[
            {
                "id": "h_low",
                "final_score": None,
                "scores": {
                    "divergence": 5.0,
                    "testability": 5.0,
                    "rationale": 5.0,
                    "robustness": 5.0,
                    "feasibility": 5.0,
                },
            },
            {
                "id": "h_top",
                "scores": {
                    "divergence": 9.0,
                    "testability": 9.0,
                    "rationale": 9.0,
                    "robustness": 9.0,
                    "feasibility": 9.0,
                },
            },
            {
                "id": "h_mid",
                "scores": {
                    "divergence": 8.0,
                    "testability": 8.0,
                    "rationale": 8.0,
                    "robustness": 8.0,
                    "feasibility": 8.0,
                },
            },
        ],
    )
    directives = await evaluator.evaluate(
        FastAgentEvent(event_type=EventType.SEARCH_COMPLETED, stage="search"),
        state,
        CognitiveBudget(total_budget=100.0),
    )
    challenge_directives = [
        d for d in directives if d.directive_type == DirectiveType.INJECT_CHALLENGE
    ]
    challenged_ids = {d.payload.get("hypothesis_id") for d in challenge_directives}
    assert challenged_ids == {"h_top", "h_mid"}
    assert len(challenge_directives) == 8


def test_slow_agent_top_hypothesis_uses_composite_fallback() -> None:
    """Top hypothesis selection should not collapse to zero when final_score is absent."""
    state = SharedCognitionState(
        hypotheses_pool=[
            {
                "id": "h_low",
                "final_score": None,
                "scores": {
                    "divergence": 4.0,
                    "testability": 4.0,
                    "rationale": 4.0,
                    "robustness": 4.0,
                    "feasibility": 4.0,
                },
            },
            {
                "id": "h_top",
                "scores": {
                    "divergence": 8.0,
                    "testability": 8.0,
                    "rationale": 8.0,
                    "robustness": 8.0,
                    "feasibility": 8.0,
                },
            },
        ]
    )
    agent = SlowAgent(event_bus=EventBus(), state=state, budget=CognitiveBudget())
    top = agent._get_top_hypothesis()
    assert top is not None
    assert top["id"] == "h_top"


class _DummyEvaluator(BaseEvaluator):
    """Evaluator that records whether it was called."""
    def __init__(self) -> None:
        self.called = False
        self.call_count = 0

    async def evaluate(self, event, state, budget):
        self.called = True
        self.call_count += 1
        return []


@pytest.mark.asyncio
async def test_evaluator_skipped_when_budget_declines() -> None:
    budget = CognitiveBudget(total_budget=100.0, strategy=AllocationStrategy.ANOMALY_DRIVEN)
    state = SharedCognitionState(
        hypotheses_pool=[{"id": "h1", "final_score": 3.0, "novelty_score": 2.0}],
    )
    evaluator = _DummyEvaluator()
    bus = EventBus()
    agent = SlowAgent(event_bus=bus, state=state, budget=budget, evaluators=[evaluator])
    event = FastAgentEvent(event_type=EventType.VERIFY_COMPLETED, stage="verify")
    await agent._run_evaluators(event)
    assert evaluator.called is False


@pytest.mark.asyncio
async def test_evaluator_runs_when_budget_approves() -> None:
    budget = CognitiveBudget(total_budget=100.0, strategy=AllocationStrategy.ANOMALY_DRIVEN)
    state = SharedCognitionState(
        hypotheses_pool=[{"id": "h1", "final_score": 8.0, "novelty_score": 5.0}],
    )
    evaluator = _DummyEvaluator()
    bus = EventBus()
    agent = SlowAgent(event_bus=bus, state=state, budget=budget, evaluators=[evaluator])
    event = FastAgentEvent(event_type=EventType.VERIFY_COMPLETED, stage="verify")
    await agent._run_evaluators(event)
    assert evaluator.called is True
    assert budget.spent == 1.0


@pytest.mark.asyncio
async def test_evaluator_skipped_when_insufficient_budget() -> None:
    budget = CognitiveBudget(total_budget=0.5, strategy=AllocationStrategy.UNIFORM)
    state = SharedCognitionState(
        hypotheses_pool=[{"id": "h1", "final_score": 8.0}],
    )
    evaluator = _DummyEvaluator()
    bus = EventBus()
    agent = SlowAgent(event_bus=bus, state=state, budget=budget, evaluators=[evaluator])
    event = FastAgentEvent(event_type=EventType.VERIFY_COMPLETED, stage="verify")
    await agent._run_evaluators(event)
    assert evaluator.called is False


@pytest.mark.asyncio
async def test_anomaly_triggers_deep_review() -> None:
    budget = CognitiveBudget(total_budget=100.0, strategy=AllocationStrategy.ANOMALY_DRIVEN)
    state = SharedCognitionState(
        hypotheses_pool=[{"id": "h1", "final_score": 2.0}],
        active_alerts=[
            CognitionAlert(alert_type="score_compression", severity="critical", description="x"),
        ],
    )
    evaluator = _DummyEvaluator()
    bus = EventBus()
    agent = SlowAgent(event_bus=bus, state=state, budget=budget, evaluators=[evaluator])
    event = FastAgentEvent(event_type=EventType.VERIFY_COMPLETED, stage="verify")
    await agent._run_evaluators(event)
    assert evaluator.called is True
    assert budget.spent == 3.0


@pytest.mark.asyncio
async def test_stage_budget_caps_evaluator_calls() -> None:
    """Evaluator should stop when verify stage allocation is exhausted."""
    budget = CognitiveBudget(
        total_budget=100.0,
        reserve_ratio=0.0,
        strategy=AllocationStrategy.ANOMALY_DRIVEN,
        stage_allocation={
            "domain_audit": 0.10,
            "biso_monitoring": 0.15,
            "search_monitoring": 0.15,
            "verify_monitoring": 0.01,
            "adversarial": 0.44,
            "global_review": 0.15,
        },
    )
    state = SharedCognitionState(
        hypotheses_pool=[{"id": "h1", "final_score": 8.0, "novelty_score": 6.0}],
    )
    evaluator = _DummyEvaluator()
    bus = EventBus()
    agent = SlowAgent(event_bus=bus, state=state, budget=budget, evaluators=[evaluator])
    event = FastAgentEvent(event_type=EventType.VERIFY_COMPLETED, stage="verify")

    await agent._run_evaluators(event)
    await agent._run_evaluators(event)

    assert evaluator.call_count == 1
    assert budget.stage_spent.get("verify_monitoring", 0.0) == pytest.approx(1.0)


class TestSemanticHasher:
    """#5: SemanticHasher must produce stable, semantically meaningful hashes."""

    @pytest.mark.asyncio
    async def test_semantic_hash_deterministic(self) -> None:
        """Same text produces same hash every time."""
        from x_creative.saga.memory.semantic_hash import SemanticHasher

        mock_embedder = AsyncMock()
        mock_embedder.embed_cached = AsyncMock(return_value=[0.1] * 128)
        hasher = SemanticHasher(embedding_service=mock_embedder, n_bits=64, embedding_dim=128)

        h1 = await hasher.hash("stock price prediction")
        h2 = await hasher.hash("stock price prediction")
        assert h1 == h2
        assert len(h1) == 64

    @pytest.mark.asyncio
    async def test_semantic_hash_similar_texts_close(self) -> None:
        """Semantically similar texts should have small Hamming distance."""
        from x_creative.saga.memory.semantic_hash import SemanticHasher, hamming_distance

        mock_embedder = AsyncMock()
        base_vec = [0.1 * i for i in range(128)]
        similar_vec = [v + 0.01 for v in base_vec]
        mock_embedder.embed_cached = AsyncMock(side_effect=[base_vec, similar_vec])

        hasher = SemanticHasher(embedding_service=mock_embedder, n_bits=64, embedding_dim=128)
        h1 = await hasher.hash("stock price prediction")
        h2 = await hasher.hash("predict stock future price")

        dist = hamming_distance(h1, h2)
        assert dist < 16

    @pytest.mark.asyncio
    async def test_semantic_hash_different_texts_far(self) -> None:
        """Semantically different texts should have large Hamming distance."""
        from x_creative.saga.memory.semantic_hash import SemanticHasher, hamming_distance

        mock_embedder = AsyncMock()
        vec_a = [1.0 if i < 64 else -1.0 for i in range(128)]
        vec_b = [-1.0 if i < 64 else 1.0 for i in range(128)]
        mock_embedder.embed_cached = AsyncMock(side_effect=[vec_a, vec_b])

        hasher = SemanticHasher(embedding_service=mock_embedder, n_bits=64, embedding_dim=128)
        h1 = await hasher.hash("thermodynamics entropy")
        h2 = await hasher.hash("medieval poetry analysis")

        dist = hamming_distance(h1, h2)
        assert dist > 20


class TestPatternMemoryWithSemanticHash:
    """#5: PatternMemory with SemanticHasher integration."""

    @pytest.mark.asyncio
    async def test_pattern_memory_fallback_without_hasher(self) -> None:
        """Without external hasher, fallback still uses semantic hash shape."""
        memory = PatternMemory()
        fp = await memory.fingerprint("Hello World", "Observable X")
        assert len(fp) == 64
        assert set(fp) <= {"0", "1"}

    @pytest.mark.asyncio
    async def test_record_batch_async(self) -> None:
        """record_batch should work as async method."""
        memory = PatternMemory()
        await memory.record_batch([{"description": "test", "observable": "obs"}])
        fp = await memory.fingerprint("test", "obs")
        assert memory.get_count(fp) == 1


@pytest.mark.asyncio
async def test_audit_runs_at_checkpoint_without_alerts() -> None:
    """#6: DomainConstraintAuditor should run even without existing critical alerts."""
    auditor = DomainConstraintAuditor()
    state = SharedCognitionState(
        target_domain_id="open_source_development",
        hypotheses_pool=[
            {
                "id": "h1",
                "description": "Uses future return at t+1",
                "observable": "future_ret_t+1",
                "analogy_explanation": "look-ahead",
            }
        ],
    )
    directives = await auditor.audit(
        FastAgentEvent(event_type=EventType.VERIFY_BATCH_SCORED, stage="verify"),
        [],
        state,
    )
    assert any(d.directive_type == DirectiveType.FLAG_HYPOTHESIS for d in directives)


@pytest.mark.asyncio
async def test_audit_checks_all_critical_constraints_via_llm() -> None:
    """#6: All plugin critical constraints should be checked via LLM."""

    class _MockRouter:
        def __init__(self):
            self.calls = 0

        async def complete(self, task, messages, temperature=None, max_tokens=None, **kw):
            self.calls += 1
            return MagicMock(content='["fat_tails"]')

    router = _MockRouter()
    auditor = DomainConstraintAuditor(router=router)
    state = SharedCognitionState(
        target_domain_id="open_source_development",
        hypotheses_pool=[
            {
                "id": "h1",
                "description": "Assume normal distribution for returns",
                "observable": "gaussian_fit",
                "analogy_explanation": "standard distribution",
            }
        ],
    )
    directives = await auditor.audit(
        FastAgentEvent(event_type=EventType.VERIFY_COMPLETED, stage="verify"),
        [],
        state,
    )
    flagged = [d for d in directives if d.directive_type == DirectiveType.FLAG_HYPOTHESIS]
    assert flagged
    violations = flagged[0].payload.get("violations", [])
    assert any("fat_tails" in str(v) for v in violations)
    assert router.calls >= 1


@pytest.mark.asyncio
async def test_audit_regex_fallback_without_router() -> None:
    """#6: Without router, auditor should still catch regex-detectable violations."""
    auditor = DomainConstraintAuditor(router=None)
    state = SharedCognitionState(
        target_domain_id="open_source_development",
        hypotheses_pool=[
            {
                "id": "h1",
                "description": "Use future returns for prediction",
                "observable": "future_ret",
                "analogy_explanation": "forward-looking",
            }
        ],
    )
    directives = await auditor.audit(
        FastAgentEvent(event_type=EventType.VERIFY_COMPLETED, stage="verify"),
        [],
        state,
    )
    assert any(d.directive_type == DirectiveType.FLAG_HYPOTHESIS for d in directives)


@pytest.mark.asyncio
async def test_t_plus_1_settlement_is_not_lookahead_violation() -> None:
    """T+1 settlement rule mention should not trigger look-ahead hard rejection."""
    auditor = DomainConstraintAuditor(router=None)
    state = SharedCognitionState(
        target_domain_id="open_source_development",
        hypotheses_pool=[
            {
                "id": "h1",
                "description": "严格遵守 T+1（交割） 规则，避免当日卖出",
                "observable": "sector_rotation_score",
                "analogy_explanation": "execution constraint, not data leak",
            }
        ],
    )
    directives = await auditor.audit(
        FastAgentEvent(event_type=EventType.VERIFY_COMPLETED, stage="verify"),
        [],
        state,
    )
    assert directives == []


@pytest.mark.asyncio
async def test_critical_constraint_heuristic_runs_without_router() -> None:
    """Critical checks should still run with heuristic fallback when router is unavailable."""
    auditor = DomainConstraintAuditor(router=None)
    constraints = [
        SimpleNamespace(
            name="onboarding_friction",
            description="solution must minimize first-use setup steps and dependencies",
        )
    ]
    violations = await auditor._check_critical_constraints(
        "This solution minimizes setup steps and dependencies explicitly.",
        constraints,
    )
    assert "critical:onboarding_friction" in violations
