"""Prompt templates for the Creativity Engine."""

BISO_ANALOGY_PROMPT = """You are an expert in finding creative analogies between distant domains and target research problems.

## Task
Given a source domain with its structures and a research problem, generate creative analogies that map concepts from the source domain to potential solutions or insights.

## Source Domain: {domain_name} ({domain_name_en})
Description: {domain_description}

### Available Structures:
{structures_text}

### Existing Mappings (for reference):
{mappings_text}

## Research Problem:
{problem_description}

### Target Domain: {target_domain}

### Domain Context:
{domain_context}

### Constraints:
{constraints_text}

## Instructions:
1. For each structure in the domain, think about how its dynamics could map to the target domain
2. Generate creative analogies that are NOVEL and TESTABLE
3. Propose specific observable proxy variables that could be measured or computed
4. Each analogy should lead to a concrete, actionable insight

## Output Format:
Return a JSON array of analogies. Each analogy should have:
- structure_id: ID of the source structure
- analogy: A short description of the analogy (in Chinese)
- explanation: Detailed explanation of how the analogy maps to the target domain (in Chinese)
- observable: Specific formula or measurement approach
- confidence: Your confidence in this analogy (0.0-1.0)
- mapping_table: Array of structural mapping items (at least 6 items). Each item:
  - source_concept: Source domain concept
  - target_concept: Target domain concept
  - source_relation: How concepts relate in source domain
  - target_relation: How concepts relate in target domain
  - mapping_type: "entity" | "relation" | "constraint" | "process"
  - systematicity_group_id: Group ID for related mappings (items in same group form a coherent relation system)
  - observable_link: Which observable/mechanism/failure-signal this mapping contributes to (required)
- failure_modes: Array of at least 2 failure scenarios. Each item:
  - scenario: When the mapping fails
  - why_breaks: Why it fails (which mapping item breaks)
  - detectable_signal: Observable signal for the failure

Example output:
```json
[
    {{
        "structure_id": "entropy_increase",
        "analogy": "系统熵增与状态分散度",
        "explanation": "如同热力学系统趋向最大熵，目标系统在某些阶段也会从有序趋向无序",
        "observable": "state_entropy = -sum(p_i * log(p_i))",
        "confidence": 0.7,
        "mapping_table": [
            {{
                "source_concept": "熵",
                "target_concept": "状态分散度",
                "source_relation": "熵随系统演化单调递增",
                "target_relation": "分散度在无新信息注入时趋于增大",
                "mapping_type": "relation",
                "systematicity_group_id": "entropy_dynamics",
                "observable_link": "state_entropy"
            }}
        ],
        "failure_modes": [
            {{
                "scenario": "当外部信息大量注入时",
                "why_breaks": "熵增假设依赖封闭系统，外部干预打破前提",
                "detectable_signal": "信息流速度 > 历史95分位"
            }}
        ]
    }}
]
```

Generate {num_analogies} analogies for this domain. Focus on novelty and testability.
"""

SEARCH_EXPAND_PROMPT = """You are an expert in refining and expanding hypotheses.

## Target Domain:
ID: {target_domain_id}
Name: {target_domain_name}
Description: {target_domain_description}

## Research Problem:
{problem_description}

### Problem Context:
{problem_context}

### Constraints:
{problem_constraints}

## Original Hypothesis:
ID: {hypothesis_id}
Description: {hypothesis_description}
Source Domain: {source_domain}
Source Structure: {source_structure}
Analogy Explanation: {analogy_explanation}
Observable: {observable}

## Expansion Task:
Generate variations of this hypothesis using the following expansion types: {expansion_types}

Expansion type meanings:
- refine: Make the hypothesis more precise, add specific conditions or thresholds
- variant: Create a variation using the same structure but different parameters/approach
- oppose: Create the opposite hypothesis (if X predicts up, does ~X predict down?)
- extreme: Push the hypothesis to its logical extreme

## Output Format:
Return a JSON array of new hypotheses:
```json
[
    {{
        "description": "New hypothesis description (Chinese)",
        "analogy_explanation": "Updated explanation (Chinese)",
        "observable": "New formula or calculation",
        "expansion_type": "refine|variant|combine|oppose|extreme"
    }}
]
```

Generate up to {max_expansions} expanded hypotheses. Each should be distinct and testable.
"""

SEARCH_COMBINE_PROMPT = """You are an expert in combining two hypotheses into interaction hypotheses.

## Hypothesis A:
ID: {hypothesis_a_id}
Description: {hypothesis_a_description}
Source Domain: {hypothesis_a_source_domain}
Source Structure: {hypothesis_a_source_structure}
Analogy Explanation: {hypothesis_a_analogy_explanation}
Observable: {hypothesis_a_observable}

## Hypothesis B:
ID: {hypothesis_b_id}
Description: {hypothesis_b_description}
Source Domain: {hypothesis_b_source_domain}
Source Structure: {hypothesis_b_source_structure}
Analogy Explanation: {hypothesis_b_analogy_explanation}
Observable: {hypothesis_b_observable}

## Task
Generate interaction hypotheses that require BOTH A and B, not a single-hypothesis rewrite.
Each output should describe a path intersection mechanism and a testable observable.

## Output Format:
Return a JSON array:
```json
[
    {{
        "description": "Combined interaction hypothesis (Chinese)",
        "analogy_explanation": "How A and B intersect mechanistically (Chinese)",
        "observable": "Testable formula using terms from both A and B",
        "expansion_type": "combine"
    }}
]
```

Generate up to {max_expansions} combined hypotheses. Ensure each one explicitly depends on both hypotheses.
"""

VERIFY_SCORE_PROMPT = """You are an expert evaluator of hypotheses.

## Target Domain:
ID: {target_domain_id}
Name: {target_domain_name}
Description: {target_domain_description}

### Domain Constraints:
{domain_constraints}

### Evaluation Criteria:
{domain_evaluation_criteria}

### Anti-Patterns to Avoid:
{domain_anti_patterns}

## Hypothesis to Score:
ID: {hypothesis_id}
Description: {hypothesis_description}
Source Domain: {source_domain}
Source Structure: {source_structure}
Analogy Explanation: {analogy_explanation}
Observable: {observable}

## Scoring Dimensions (0-10 scale):

### 1. Divergence (语义发散度)
How different is this from commonly known approaches in this target domain?
- 10: Completely novel concept rarely seen in this domain
- 7: Uses unusual domain mapping but related to known ideas
- 4: Standard approach with slight twist
- 1: Very common / canonical approach

### 2. Testability (可检验性)
Can this be converted into a concrete test (experiment, benchmark, prototype, ablation, etc.)?
- 10: Clear measurement or procedure, immediately implementable
- 7: Procedure is clear but needs some data/prep
- 4: Testable in principle but needs significant work to operationalize
- 1: Vague, unclear how to test

### 3. Rationale (机制合理性)
Is there a sound mechanism?
- 10: Clear mechanism with solid support
- 7: Reasonable mechanism, logically consistent
- 4: Mechanism is plausible but speculative
- 1: No clear mechanism, purely statistical

### 4. Robustness (稳健性先验)
How likely is this to avoid overfitting?
- 10: Simple, few parameters, broad applicability
- 7: Moderate complexity, reasonable scope
- 4: Multiple conditions, narrow applicability
- 1: Many conditions, very specific scenarios, high overfit risk

### 5. Feasibility (数据可获取性)
How difficult is it to obtain the data needed to implement this hypothesis?
- 10: All data/resources are easily available
- 7: Data available but requires preprocessing or combining multiple sources
- 4: Data requires special access or significant work to collect
- 1: Data is proprietary/restricted/extremely difficult to obtain

## Output Format:
Return a JSON object with scores and reasons:
```json
{{
    "divergence": 8.5,
    "divergence_reason": "Explanation for divergence score",
    "testability": 9.0,
    "testability_reason": "Explanation for testability score",
    "rationale": 7.5,
    "rationale_reason": "Explanation for rationale score",
    "robustness": 8.0,
    "robustness_reason": "Explanation for robustness score",
    "feasibility": 7.0,
    "feasibility_reason": "Explanation for feasibility score"
}}
```

Be rigorous and objective in your evaluation. Do not inflate scores.
"""

TRANSFORM_GATE_CK_VERIFY_PROMPT = """You are a logic verifier evaluating a proposed conceptual space transformation.

## Original Hypothesis:
{hypothesis_description}
Observable: {observable}

## Proposed Transformation:
{transform_summary}

## Concept Space Context:
Domain: {domain_id}
Hard Constraints: {hard_constraints}

## Evaluation Criteria:
1. **Logical consistency**: Do the transformation actions produce a coherent result? Are before_state → after_state transitions logically sound?
2. **Constraint compliance**: Does the transformed hypothesis respect all hard constraints?
3. **Testability preserved**: Does the transformation maintain or produce new testable observables?
4. **Failure mode adequacy**: Are the new failure modes realistic and detectable?

## Output Format:
Return a JSON object:
```json
{{
    "verdict": "accept" | "reject",
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation of the verdict",
    "issues": ["list of specific issues found, if any"]
}}
```

Be rigorous. Reject transformations that are logically incoherent, violate hard constraints, or lack testable consequences.
"""
