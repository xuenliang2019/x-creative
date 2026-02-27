"""Prompt templates for the Answer Engine."""

PROBLEM_FRAME_PROMPT = """You are an expert problem analyst. Given a user's question, decompose it into a structured problem frame.

Available target domains: {available_domains}

User question: {question}

Respond with a JSON object containing exactly these fields:
{{
  "objective": "What the user wants to achieve (1-2 sentences)",
  "constraints": [
    {{"text": "constraint description", "source": "explicit|inferred", "confidence": 0.0-1.0}}
  ],
  "scope": {{
    "in_scope": ["topics/areas to cover"],
    "out_of_scope": ["topics/areas to exclude"]
  }},
  "definitions": {{"key_term": "definition"}},
  "success_criteria": ["how to evaluate if the answer is good"],
  "domain_hint": {{
    "domain_id": "best matching domain from available list, or 'general'",
    "confidence": 0.0-1.0
  }},
  "open_questions": ["ambiguities that could affect the answer"],
  "context": {{}}
}}

Rules:
- domain_hint.confidence should reflect how certain you are about the domain match
- If the question is clearly about one of the available domains, set confidence >= 0.7
- If unclear, set confidence lower and add to open_questions
- constraints with source="inferred" are your best guesses; mark confidence accordingly
- Respond ONLY with valid JSON, no markdown fences or extra text"""


TARGET_DOMAIN_RESOLVE_PROMPT = """You are a domain classification expert.

Given a problem frame and available target domains, determine the best matching domain.

Problem objective: {objective}
Problem description: {description}
Key terms: {terms}

Available domains:
{domain_descriptions}

Respond with JSON:
{{
  "domain_id": "best matching domain ID",
  "confidence": 0.0-1.0,
  "reasoning": "why this domain matches"
}}"""


EPHEMERAL_DOMAIN_PROMPT = """You are a domain expert. Given a problem that doesn't match any existing domain template, generate a temporary domain configuration.

Problem: {description}
Objective: {objective}

Generate a JSON domain configuration:
{{
  "id": "ephemeral_{short_id}",
  "name": "Domain name",
  "description": "What this domain covers",
  "constraints": [
    {{"name": "constraint_id", "description": "description", "severity": "critical|important|advisory"}}
  ],
  "evaluation_criteria": ["criterion1", "criterion2"],
  "anti_patterns": ["pattern to avoid"],
  "terminology": {{"term": "definition"}},
  "stale_ideas": ["overused approach"],
  "source_domains": []
}}

Rules:
- Generate 3-5 constraints (at least 1 critical)
- Generate 3-5 evaluation criteria
- Generate 2-4 anti-patterns
- Keep terminology focused on the problem domain
- source_domains will be filled separately; leave as empty list
- Respond ONLY with valid JSON"""


FRESH_DOMAIN_PROMPT = """You are a domain expert and creative thinker. Given a problem, generate a complete domain configuration INCLUDING rich source domains for bisociative creativity.

Problem: {description}
Objective: {objective}

Generate a JSON domain configuration with the following structure:
{{
  "id": "fresh_{short_id}",
  "name": "Domain name",
  "description": "What this domain covers",
  "constraints": [
    {{"name": "constraint_id", "description": "description", "severity": "critical|important|advisory"}}
  ],
  "evaluation_criteria": ["criterion1", "criterion2"],
  "anti_patterns": ["pattern to avoid"],
  "terminology": {{"term": "definition"}},
  "stale_ideas": ["overused approach"],
  "source_domains": [
    {{
      "id": "domain_id",
      "name": "域名（中文）",
      "name_en": "Domain Name (English)",
      "description": "What this source domain studies",
      "structures": [
        {{
          "id": "structure_id",
          "name": "结构名称",
          "description": "Description of this structural pattern",
          "key_variables": ["var1", "var2"],
          "dynamics": "How variables interact"
        }}
      ],
      "target_mappings": [
        {{
          "structure": "structure_id",
          "target": "How this maps to the target domain",
          "observable": "Concrete measurable indicators"
        }}
      ]
    }}
  ]
}}

Rules:
- Generate 3-5 constraints (at least 1 critical)
- Generate 3-5 evaluation criteria
- Generate 2-4 anti-patterns
- Keep terminology focused on the problem domain
- Generate 8-15 diverse source_domains from DISTANT fields (physics, biology, music theory, game theory, ecology, linguistics, etc.)
- Each source domain should have 3-5 structures with concrete key_variables and dynamics
- Each source domain must include target_mappings connecting structures to the problem domain
- Prefer DISTANT analogies over obvious ones — the goal is bisociative creativity
- Respond ONLY with valid JSON"""


SOURCE_RELEVANCE_PROMPT = """Rate each source domain's relevance to this problem (0.0-1.0).

Problem: {description}
Objective: {objective}

Source domains:
{domain_list}

Respond with JSON array of objects:
[
  {{"domain_id": "id", "relevance": 0.0-1.0, "reason": "brief justification"}}
]

Rate higher for domains whose structural patterns could provide novel insights for this problem.
Rate lower for domains too similar to the problem domain (we want distant analogies).
Respond ONLY with valid JSON."""


MAPPING_FEASIBILITY_PROMPT = """Assess whether each source domain has structurally mappable relationships to this problem. Focus on whether the domain's core dynamics (not surface features) can form systematic analogies.

Problem: {description}
Objective: {objective}

Source domains:
{domain_list}

For each domain, rate mapping feasibility (0.0-1.0):
- 0.8-1.0: Core mechanisms clearly map to the problem (feedback loops, causal chains, constraints align)
- 0.5-0.7: Some structural parallels exist but require creative bridging
- 0.2-0.4: Only surface-level similarity, no deep structural mapping possible
- 0.0-0.1: No meaningful structural relationship

Respond with JSON array:
[{{"domain_id": "id", "feasibility": 0.0-1.0, "reason": "brief justification"}}]

Respond ONLY with valid JSON."""
