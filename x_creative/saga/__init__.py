"""SAGA — Slow Agent and fast aGent Architecture.

Dual-process cognitive architecture for X-Creative, inspired by
Kahneman's System 1/System 2 and DeepMind's Talker-Reasoner.

Core components:
- SAGACoordinator: Orchestrates Fast and Slow Agent lifecycle
- FastAgent: Event-aware wrapper around CreativityEngine
- SlowAgent: Metacognitive overseer with detection/audit/evaluation layers
- EventBus: Async event bus for inter-agent communication
- SharedCognitionState: Shared state between agents
- CognitiveBudget: Controls Slow Agent's review depth

Talker-Reasoner components (solve stage):
- Reasoner: Multi-step reasoning agent (System 2) building BeliefState
- Talker: Belief-conditioned output generator (System 1)
- TalkerReasonerSolver: Orchestrates Reasoner → Talker pipeline
- BeliefState: Structured belief state shared between agents
"""

from x_creative.saga.belief import (
    BeliefState,
    EvidenceItem as SolveEvidenceItem,
    ReasoningStep,
    RefinementRound,
    RefinementTrace,
    TalkerReasonerResult,
    UserQuestion,
)
from x_creative.saga.budget import AllocationStrategy, BudgetPolicy, CognitiveBudget
from x_creative.saga.coordinator import AuditReport, SAGACoordinator, SAGAResult
from x_creative.saga.events import (
    DirectiveType,
    EventBus,
    EventType,
    FastAgentEvent,
    SlowAgentDirective,
)
from x_creative.saga.fast_agent import FastAgent
from x_creative.saga.reasoner import QualityAuditRejected, ReasonerFatalError, Reasoner
from x_creative.saga.slow_agent import (
    BaseAuditor,
    BaseDetector,
    BaseEvaluator,
    SlowAgent,
)
from x_creative.saga.solve import SAGASolver, SolveResult, TalkerReasonerSolver

# Backward compatibility alias
IdeaEvidence = SolveEvidenceItem
from x_creative.saga.state import (
    CognitionAlert,
    EvaluationAdjustments,
    GenerationMetrics,
    SharedCognitionState,
)
from x_creative.saga.talker import Talker

__all__ = [
    # Coordinator
    "SAGACoordinator",
    "SAGAResult",
    "AuditReport",
    # Agents
    "FastAgent",
    "SlowAgent",
    "SAGASolver",
    "SolveResult",
    "IdeaEvidence",
    # Talker-Reasoner
    "Reasoner",
    "QualityAuditRejected",
    "ReasonerFatalError",
    "Talker",
    "TalkerReasonerSolver",
    "TalkerReasonerResult",
    "BeliefState",
    "ReasoningStep",
    "RefinementRound",
    "RefinementTrace",
    "SolveEvidenceItem",
    "UserQuestion",
    # ABC base classes
    "BaseDetector",
    "BaseAuditor",
    "BaseEvaluator",
    # Events
    "EventBus",
    "EventType",
    "DirectiveType",
    "FastAgentEvent",
    "SlowAgentDirective",
    # State
    "SharedCognitionState",
    "GenerationMetrics",
    "CognitionAlert",
    "EvaluationAdjustments",
    # Budget
    "CognitiveBudget",
    "BudgetPolicy",
    "AllocationStrategy",
]
