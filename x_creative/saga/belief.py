"""Structured belief state for the Talker-Reasoner architecture.

The BeliefState is the core integration mechanism between Reasoner and Talker.
Each reasoning step incrementally updates it; Talker output is conditioned on it.
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ReasoningPhase(str, Enum):
    """Phases of the Reasoner's multi-step reasoning chain."""

    PROBLEM_ANALYSIS = "problem_analysis"
    HYPOTHESIS_EVALUATION = "hypothesis_evaluation"
    EVIDENCE_GATHERING = "evidence_gathering"
    CROSS_VALIDATION = "cross_validation"
    SOLUTION_PLANNING = "solution_planning"
    QUALITY_AUDIT = "quality_audit"
    BELIEF_SYNTHESIS = "belief_synthesis"


class ReasoningStep(BaseModel):
    """A single Think→Act→Observe→Update step in the reasoning chain."""

    step_number: int
    phase: ReasoningPhase
    thought: str
    action: str
    observation: str
    belief_update: str
    llm_calls: int = 0
    tokens_used: int = 0
    elapsed_seconds: float = 0.0


class ProblemAnalysis(BaseModel):
    """Reasoner's structured understanding of the problem (Step 1 output)."""

    core_challenge: str = ""
    sub_problems: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)
    implicit_constraints: list[str] = Field(default_factory=list)
    domain_context: str = ""


class HypothesisVerdict(BaseModel):
    """Reasoner's evaluation of a single hypothesis (Step 2 output element)."""

    hypothesis_id: str
    description: str
    source_domain: str
    relevance: str = ""
    strength: str = ""
    weakness: str = ""
    actionability: str = ""
    priority: int = 0  # 1 = highest


class EvidenceItem(BaseModel):
    """Web-validated evidence for a hypothesis (Step 3 output element)."""

    evidence_id: str
    hypothesis_id: str
    hypothesis_description: str
    source_domain: str
    source_structure: str
    preliminary_score: float = 0.0
    novelty_score: float = 0.0
    novelty_analysis: str = ""
    references: list[dict[str, Any]] = Field(default_factory=list)
    reasoner_assessment: str = ""


class CrossValidation(BaseModel):
    """Cross-validation analysis between hypotheses (Step 4 output)."""

    complementary_pairs: list[dict[str, str]] = Field(default_factory=list)
    contradictions: list[dict[str, str]] = Field(default_factory=list)
    dependencies: list[dict[str, str]] = Field(default_factory=list)
    synthesis: str = ""


class SolutionBlueprint(BaseModel):
    """Phased execution plan (Step 5 output)."""

    executive_summary: str = ""
    key_insights: list[str] = Field(default_factory=list)
    phases: list[dict[str, Any]] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    tools_and_resources: list[str] = Field(default_factory=list)


class QualityAssessment(BaseModel):
    """Adversarial quality audit result (Step 6 output)."""

    verdict: str = ""  # approve / revise
    risks: list[dict[str, str]] = Field(default_factory=list)
    evidence_gaps: list[str] = Field(default_factory=list)
    hallucination_flags: list[str] = Field(default_factory=list)
    improvement_suggestions: list[str] = Field(default_factory=list)


class RefinementRound(BaseModel):
    """Record of a single inner-loop refinement round (Step 5↔6 iteration)."""

    round_number: int
    high_risks_before: int
    high_risks_after: int
    risks_addressed: list[dict[str, str]] = Field(default_factory=list)


class RefinementTrace(BaseModel):
    """Full trace of the adaptive risk refinement process."""

    inner_rounds: list[RefinementRound] = Field(default_factory=list)
    outer_rounds: int = 0
    total_constraints_added: list[str] = Field(default_factory=list)
    converged: bool = False
    final_high_risk_count: int = 0


class UserClarification(BaseModel):
    """Record of a user clarification during reasoning."""

    question: str
    context: str = ""
    response: str = ""
    phase: ReasoningPhase | None = None


class UserQuestion(BaseModel):
    """A question the Reasoner poses to the user."""

    question: str
    context: str = ""
    options: list[str] = Field(default_factory=list)
    default: str = ""


class BeliefState(BaseModel):
    """Reasoner's structured belief state — core integration mechanism.

    Incrementally updated after each reasoning step.
    Talker's output is conditioned on this state.
    """

    # Step 1 output
    problem_analysis: ProblemAnalysis = Field(default_factory=ProblemAnalysis)
    # Step 2 output
    hypothesis_verdicts: list[HypothesisVerdict] = Field(default_factory=list)
    # Step 3 output
    evidence: list[EvidenceItem] = Field(default_factory=list)
    # Step 4 output
    cross_validation: CrossValidation = Field(default_factory=CrossValidation)
    # Step 5 output
    solution_blueprint: SolutionBlueprint = Field(default_factory=SolutionBlueprint)
    # Step 6 output
    quality_assessment: QualityAssessment = Field(default_factory=QualityAssessment)
    # User interactions
    user_clarifications: list[UserClarification] = Field(default_factory=list)
    # Reasoning trace
    reasoning_steps: list[ReasoningStep] = Field(default_factory=list)
    # Refinement trace
    refinement_trace: RefinementTrace = Field(default_factory=RefinementTrace)
    # Metrics
    total_llm_calls: int = 0
    total_tokens_used: int = 0
    confidence: float = 0.0


class TalkerReasonerResult(BaseModel):
    """Final output of a Talker-Reasoner solve run."""

    solution_markdown: str
    belief_state: BeliefState
    evidence: list[EvidenceItem] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
