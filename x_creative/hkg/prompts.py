"""LLM prompt templates for HKG expansion and bridging."""

HYPERPATH_EXPAND_PROMPT = """You are a scientific reasoning expert. Given a hypothesis and structural evidence paths from a knowledge hypergraph, generate improved hypotheses with stronger mechanism chains.

## Current Hypothesis:
{hypothesis_description}

Source Domain: {source_domain}
Observable: {observable}

## Structural Evidence (Hyperpaths):
{structural_context}

## Research Problem:
{problem_description}

## Instructions:
Using ONLY the structural evidence paths above:
1. Identify how the intermediate nodes serve as causal/mechanism bridges
2. Generate {max_expansions} refined hypotheses that:
   - Explicitly use the intermediate nodes as mechanism steps
   - Add 1-3 testable conditions derived from the path structure
   - Propose concrete observable variables
3. Do NOT invent intermediate concepts not present in the paths
4. Each hypothesis must reference which path edges support it
5. `description` / `analogy_explanation` / `mechanism_chain` / `observable` must be non-empty
6. `testable_conditions` must contain 1-3 non-empty items

## Output Format:
Return a JSON array:
```json
[
    {{
        "description": "Refined hypothesis using path mechanism (Chinese)",
        "analogy_explanation": "How the path bridges source to target (Chinese)",
        "observable": "Concrete formula or measurement",
        "expansion_type": "hyperpath_expand",
        "mechanism_chain": "node_A -> intermediate -> node_B explanation",
        "testable_conditions": ["condition 1", "condition 2"],
        "supporting_edges": ["e1", "e2"]
    }}
]
```
"""

HYPERBRIDGE_PROMPT = """You are a creative scientific reasoning expert. Given two concepts and structural bridge paths from a knowledge hypergraph, generate novel cross-domain hypotheses.

## Concept A: {concept_a}
## Concept B: {concept_b}

## Structural Bridge Paths:
{structural_context}

## Instructions:
For each bridge path, generate 2-3 creative bridging hypotheses that:
1. Explain a non-obvious connection between Concept A and Concept B
2. Use intermediate nodes as the mechanism bridge (do NOT invent nodes)
3. Each explanation must lead to a concrete observable variable
4. Be creative but verifiable
5. Include `path_rank` (1-based, matching the path order in context)
6. `testable_conditions` must contain 1-3 non-empty items

## Output Format:
Return a JSON array:
```json
[
    {{
        "description": "Bridge hypothesis (Chinese)",
        "analogy_explanation": "How concepts connect via bridge (Chinese)",
        "observable": "Concrete formula or measurement",
        "expansion_type": "hyperbridge",
        "bridge_path": "concept_A -> intermediate -> concept_B",
        "testable_conditions": ["condition 1"],
        "path_rank": 1
    }}
]
```
"""
