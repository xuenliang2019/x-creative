<p align="center">
  <img src="open-x-creative.png" alt="X-Creative Logo" width="200">
</p>

<h1 align="center">X-Creative</h1>

<p align="center">A general-purpose research Agent Workflow system driven by <b>creativity theory</b>.</p>

Input a question, and the system automatically discovers structural isomorphisms from distant knowledge domains, generates cross-domain innovative hypotheses, validates them through multi-model verification and risk refinement, and outputs a complete research report with direct answers, traceable evidence, and risk boundaries.

```bash
x-creative answer -q "How to improve fault tolerance in distributed systems"
```

---

## Table of Contents

- [Introduction](#introduction)
- [How It Works](#how-it-works)
- [Directory Structure](#directory-structure)
- [Installation & Configuration](#installation--configuration)
- [CLI Usage](#cli-usage)
- [Python API](#python-api)
- [Configuration System](#configuration-system)
- [Target Domain Configuration](#target-domain-configuration)
- [Output Format](#output-format)
- [Development](#development)
- [Source Code Architecture](#source-code-architecture)
- [Acknowledgments](#acknowledgments)

---

## Introduction

### What Problem Does It Solve

When facing a research or innovation problem, the traditional approach is **linear thinking within a known domain** — which easily gets trapped in local optima. X-Creative takes a different approach: it searches for structural similarities in **completely unrelated knowledge domains** and uses these cross-domain analogies to generate innovative hypotheses.

For example, when you ask "How to improve user retention for an open-source project", the system might discover structural isomorphisms between issue response delays and service queue congestion from **queueing theory**, mapping relationships between community niche competition and species coexistence from **ecology**, and analogies between user churn and entropy increase from **thermodynamics** — these cross-domain perspectives often inspire innovative solutions that are difficult to discover within a single domain.

### The Principles Behind It

The core theoretical foundation is **Bisociation** — a creativity theory proposed by Arthur Koestler (1964):

> **The essence of creative acts is connecting two previously unrelated frames of reference.**

Unlike everyday association (linear thinking within a single framework), bisociation requires simultaneously operating on two independent "planes of thought" and discovering structural isomorphisms between them. Classic examples:

- Archimedes discovering the law of buoyancy in the bathtub — connecting "water level rising during bathing" with "measuring object volume"
- Darwin's natural selection — connecting "artificial breeding" with "natural species variation"

Beyond Bisociation, the system integrates three complementary creativity theories:

| Theory | Core Claim | System Mapping |
|--------|-----------|----------------|
| **Boden's Three Types of Creativity** | Combinational, exploratory, and transformational creativity | SEARCH stage's combine / refine / transform_space operators |
| **Conceptual Blending (Fauconnier-Turner)** | Bidirectional four-space blending produces emergent structures | blend_expand operator: generates novel hypotheses from hypothesis pairs |
| **C-K Theory (Hatchuel-Weil)** | Alternating expansion of concept space and knowledge space | SAGA-orchestrated C→K / K→C phase switching |

> For detailed theoretical exposition and design, see [`docs/theory.en.md`](docs/theory.en.md).

### Core Pipeline

The system translates the above theories into computable steps through a four-stage pipeline:

```
Question Input
      │
      ▼
┌──────────────────────────────────────────────────────────┐
│ BISO (Distant Domain Association)                        │
│ Generate cross-domain analogy hypotheses from 18-30      │
│ distant source domains (~50-60 hypotheses)               │
└────────────────────────┬─────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│ SEARCH (Graph-of-Thoughts Search)                        │
│ Multi-strategy hypothesis space expansion (~100-200+)    │
│ Operators: refine / variant / combine / blend / transform│
└────────────────────────┬─────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│ VERIFY (Dual-Model Verification)                         │
│ Five-dimensional scoring + logic verification +          │
│ novelty verification + mapping quality gate              │
└────────────────────────┬─────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│ SOLVE (Talker-Reasoner Reasoning)                        │
│ 7-step multi-step reasoning + web evidence collection +  │
│ adaptive risk refinement loop                            │
└────────────────────────┬─────────────────────────────────┘
                         │
                         ▼
              Research Report Output
               (Markdown + JSON)
```

### Key Features

- **Universal domain support**: Supports any target domain via YAML configuration (scientific research, product innovation, open-source development, etc.)
- **Distant domain association (BISO)**: Automatically generates cross-domain analogies from multiple source domains with feasibility filtering to avoid invalid mappings
- **Mapping quality gate**: Anti-padding rules + LLM scoring; low-quality mappings don't enter subsequent stages
- **Dual-model verification**: Logic verification + novelty verification with confidence-driven selective evaluation and position bias defense
- **User constraint system**: Constraint preflight (conflict detection) → constraint compilation (HardCore + ActiveSoftSet) → compliance audit and patch loop
- **Multi-model orchestration**: Access cloud LLMs via OpenRouter / Yunwu API with task-level routing and automatic fallback
- **SAGA dual-process cognitive architecture** (experimental): Fast Agent (System 1) + Slow Agent (System 2) metacognitive supervision
- **Hypergraph Knowledge Grounding (HKG)** (experimental): Structural evidence-driven hypothesis completion and bridging
- **Conceptual Blending** (experimental): Fauconnier-Turner four-space blending
- **Transformational Creativity** (experimental): Boden's transformational rule-breaking
- **MOME Quality-Diversity Archive** (experimental): MAP-Elites diversity maintenance on behavioral grids
- **C-K Dual-Space Scheduling** (experimental): Automatic alternating expansion of concept and knowledge spaces

---

## Directory Structure

```
x-creative/
├── x_creative/                      # Main package (source code)
│   ├── answer/                      #   Answer Engine (single-entry orchestrator)
│   ├── cli/                         #   CLI command definitions
│   ├── config/                      #   Configuration & settings
│   │   └── target_domains/          #     Built-in target domain YAMLs
│   ├── core/                        #   Core types (Hypothesis, ProblemFrame, Domain)
│   ├── creativity/                  #   Creativity engine (BISO, SEARCH, operators)
│   ├── verify/                      #   Dual-model verification system
│   ├── saga/                        #   SAGA dual-process cognitive architecture
│   ├── hkg/                         #   Hypergraph Knowledge Grounding
│   ├── llm/                         #   LLM client & routing
│   ├── session/                     #   Session management
│   ├── domain_manager/              #   Source domain TUI manager (xc-domain)
│   └── target_manager/              #   Target domain TUI manager (xc-target)
│
├── docs/                            # Documentation
│   ├── theory.md                    #   Design document (Chinese)
│   └── theory.en.md                 #   Design document (English)
│
├── local_data/                      # Session data storage (generated at runtime)
│   ├── .current_session             #   Current active session marker
│   └── <session-id>/                #   Per-session data directory
│       ├── problem.json / .md       #     Problem definition
│       ├── biso.json / .md          #     BISO stage results
│       ├── search.json / .md        #     SEARCH stage results
│       ├── verify.json / .md        #     VERIFY stage results
│       ├── solve.json / .md         #     SOLVE stage results
│       ├── answer.json / .md        #     Answer Engine final report
│       └── saga/                    #     SAGA internal state & logs
│
├── log/                             # Application logs (generated at runtime)
│   └── output.log                   #   Main log file
│
├── pyproject.toml                   # Poetry project configuration
├── poetry.lock                      # Dependency lock file
├── .python-version                  # Python version (pyenv)
├── .env.example                     # Environment variable template
├── .env                             # Actual environment variables (gitignored)
└── CLAUDE.md                        # Claude Code instructions
```

---

## Installation & Configuration

### Prerequisites

- **Python 3.12+**
- **Poetry 2.1+**
- **LLM Provider API Key** (OpenRouter or Yunwu, at least one)

### Step 1: Install Python (via pyenv)

If you don't have Python 3.12+ yet, we recommend using [pyenv](https://github.com/pyenv/pyenv) to manage Python versions:

```bash
# Install pyenv (if not installed)
curl https://pyenv.run | bash

# Install Python 3.12
pyenv install 3.12.12

# The correct version is used automatically when entering the project directory
# (the project includes a .python-version file)
cd x-creative
python --version  # Should show 3.12.12
```

### Step 2: Install Poetry

```bash
# Install Poetry (if not installed)
curl -sSL https://install.python-poetry.org | python3 -

# Verify
poetry --version  # Requires 2.1+
```

### Step 3: Clone the Project and Install Dependencies

```bash
git clone https://github.com/xuenliang2019/x-creative.git
cd x-creative

# Install all dependencies
poetry install

# Verify installation
poetry run x-creative --version
```

### Step 4: Configure API Keys

X-Creative requires LLM API access to large language models. At least one LLM Provider must be configured.

#### Set Up an OpenRouter Account (Recommended)

1. Visit https://openrouter.ai/ and create an account
2. Go to https://openrouter.ai/keys to create an API Key
3. Add credits (pay-per-use billing)

#### Set Environment Variables

Copy the example configuration file and fill in your API Key:

```bash
cp .env.example .env
```

Edit the `.env` file and set the following required fields:

```bash
# LLM Provider (at least one required)
OPENROUTER_API_KEY=sk-or-v1-your-key-here

# Or use Yunwu (https://yunwu.ai/)
# YUNWU_API_KEY=your-yunwu-key-here
# X_CREATIVE_DEFAULT_PROVIDER=yunwu

# Optional: Brave Search API Key (for novelty verification web search)
# Sign up at: https://brave.com/search/api/
# BRAVE_SEARCH_API_KEY=your-brave-api-key
```

#### Verify Configuration

```bash
# Full verification (static check + API connectivity + model availability)
poetry run x-creative config check

# Static check only (no API requests)
poetry run x-creative config check --quick
```

### Step 5: Confirm Runtime Directories

The system uses two directories at runtime:

| Directory | Purpose | Default Location | Configuration |
|-----------|---------|-----------------|---------------|
| `local_data/` | Stores session data (problem definitions, hypotheses, reports) | Project root | Environment variable `X_CREATIVE_DATA_DIR` |
| `log/` | Stores runtime logs | Project root | Auto-created |

Both directories are automatically created on first run — no manual action needed. To customize the data directory:

```bash
# Set in .env
X_CREATIVE_DATA_DIR=/path/to/your/data
```

---

## CLI Usage

All commands are available via `poetry run x-creative` or directly as `x-creative` after activating the virtual environment:

```bash
# Option 1: Via poetry run
poetry run x-creative <command>

# Option 2: Activate virtual environment first
poetry shell
x-creative <command>
```

The examples below omit the `poetry run` prefix.

### The Simplest Start: `answer`

`answer` is the system's **primary entry point** — input a question and the entire workflow runs automatically:

```bash
x-creative answer -q "How to improve fault tolerance in distributed systems"
```

This single command automatically executes: problem framing → target domain inference → source domain selection → BISO → SEARCH → VERIFY → SOLVE → report output.

#### answer Command Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--question` | `-q` | (required) | Research question |
| `--budget` | - | `60` | Cognitive budget units |
| `--target` | - | `auto` | Target domain ID (`auto` = auto-detect) |
| `--depth` | - | `3` | SEARCH depth |
| `--breadth` | - | `5` | SEARCH breadth |
| `--mode` | - | `deep_research` | Mode: `quick` / `deep_research` / `exhaustive` |
| `--no-hkg` | - | off | Disable Hypergraph Knowledge Grounding |
| `--no-saga` | - | off | Disable SAGA supervision |
| `--fresh` | - | off | Skip pre-defined YAML domains, generate from scratch via LLM |
| `--output` | `-o` | - | Save Markdown report to file |

#### answer Usage Examples

```bash
# Auto-detect target domain
x-creative answer -q "How to improve fault tolerance in distributed systems"

# Specify target domain + export report
x-creative answer -q "How to improve fault tolerance in distributed systems" \
  --target open_source_development \
  --depth 2 \
  --output report.md

# Quick mode (reduced search depth, faster results)
x-creative answer -q "Explore user retention improvement strategies" --mode quick

# Disable SAGA supervision (faster, lower cost)
x-creative answer -q "Test question" --no-saga --no-hkg

# Fresh generation mode (skip pre-defined YAML, LLM generates domains from scratch)
x-creative answer -q "Explore the impact of quantum computing on cryptography" --fresh
```

#### answer Output

The `answer` command generates reports in two formats (saved in the session directory):

- **Markdown report** (`answer.md`): Direct answer, key evidence, risk boundaries, hypothesis rankings, methodology appendix
- **JSON structured data** (`answer.json`): All metadata (session ID, target domain, source domain count, search rounds, budget consumption)

If the system cannot determine the problem domain (confidence < 0.3), it will interactively ask a clarifying question before continuing.

---

### Step-by-Step Workflow: `session` + `run`

When you need finer control, use the step-by-step workflow: manually create a session and execute each stage individually.

#### Managing Sessions

```bash
# Create a new session (automatically set as current)
x-creative session new "Research topic"

# Custom session ID
x-creative session new "Research topic" --id my-research

# List all sessions
x-creative session list

# View current session status
x-creative session status

# Switch current session
x-creative session switch my-research

# Delete a session
x-creative session delete my-research
```

#### Executing the Pipeline Step by Step

The workflow consists of 4 base stages (`problem` → `biso` → `search` → `verify`) and 1 optional solving stage (`solve`):

```bash
# 1. Define the research problem
x-creative run problem -d "Explore innovative methods to improve user retention" \
  --target-domain open_source_development \
  --constraint "No increase in operational costs" \
  --constraint "Implementable within 2 weeks"

# 2. Run BISO (distant domain association)
x-creative run biso --num-per-domain 3

# 3. View intermediate results
x-creative show biso --top 5

# 4. Run SEARCH (hypothesis expansion)
x-creative run search --depth 2

# 5. Run VERIFY (dual-model verification)
x-creative run verify --threshold 6.0 --top 20

# 6. View final verification results
x-creative show verify --top 10

# 7. Run SOLVE (deep reasoning)
x-creative run solve --max-ideas 8 --auto-refine
```

You can also run all base stages at once:

```bash
x-creative run all \
  -d "Explore new solutions" \
  --target-domain general \
  --num-per-domain 2 \
  --depth 2 \
  --top 30
```

#### Detailed Parameters for Each run Stage

**`run problem` — Define the research problem**

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--description` | `-d` | (interactive) | Problem description |
| `--target-domain` | `-t` | `general` | Target domain ID |
| `--context` | `-c` | `{}` | Domain context (JSON) |
| `--constraint` | - | `[]` | Constraints (can be used multiple times) |
| `--session` | `-s` | current | Specify session ID |
| `--force` | - | off | Force re-execution |

`--context` is a free-form JSON object that provides domain background information to the LLM (both keys and values can be freely defined):

```bash
x-creative run problem -d "Design an open-source CLI tool" \
  --target-domain open_source_development \
  --context '{"platform": "github", "language": "rust", "target_users": "developers"}' \
  --constraint "Must be implementable as a single-binary CLI tool"
```

**`run biso` — Distant domain association**

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--num-per-domain` | `-n` | `3` | Number of hypotheses per source domain |

**`run search` — Hypothesis expansion**

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--depth` | `-d` | `3` | Search depth (iteration rounds) |
| `--breadth` | `-b` | `5` | Search breadth (expansions per round) |

**`run verify` — Dual-model verification**

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--threshold` | `-t` | `5.0` | Minimum score threshold |
| `--top` | - | `50` | Output top N hypotheses |

**`run solve` — Talker-Reasoner deep reasoning**

Prerequisite: The current session's `verify` stage must be completed.

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--max-ideas` | - | `8` | Select top N hypotheses from verify results |
| `--max-web-results` | - | `8` | Max web search results per round |
| `--auto-refine / --no-auto-refine` | - | on | Enable/disable adaptive risk refinement loop |
| `--inner-max` | - | `3` | Max inner loop iterations |
| `--outer-max` | - | `2` | Max outer loop iterations |
| `--no-interactive` | - | off | Disable interactive questions |
| `--force` | - | off | Overwrite existing results |

All `run` subcommands support `--session <id>` and `--force` options.

---

### Viewing Results: `show`

```bash
# View results for each stage
x-creative show problem [--session <id>] [--raw]
x-creative show biso [--top 10] [--raw]
x-creative show search [--top 10] [--raw]
x-creative show verify [--top 10] [--raw]

# View stage Markdown report
x-creative show report <stage>  # stage: problem, biso, search, verify

# Export report to file
x-creative show report verify --output report.md
```

---

### Managing Source Domains: `domains` and `xc-domain`

```bash
# List source domain library
x-creative domains list --target open_source_development

# View details of a specific source domain
x-creative domains show thermodynamics --target open_source_development
```

`xc-domain` is a standalone TUI tool for interactively managing source domain configurations:

```bash
poetry run xc-domain
```

Supports three operations:
- **Manual domain addition**: Enter domain name → Brave Search for core concepts → LLM generates structures and mappings → review and save
- **Automatic domain exploration**: Enter research goal → recommend 5-8 candidate source domains → smart deduplication (detects overlap with existing domains) → batch generation
- **Extend existing structures**: Add new structures to existing domains

---

### Managing Target Domains: `xc-target`

```bash
poetry run xc-target
```

A TUI tool for creating and managing target domain configurations:
- **Creation wizard**: Basic info → LLM generates metadata in parallel (constraints, evaluation_criteria, anti_patterns, terminology, stale_ideas) → optional copy from existing source domains → save
- **View & edit**: Tab-based viewing of each section, with selective regeneration support

---

### Quick Generation: `generate`

No session needed — generate hypotheses directly:

```bash
# Basic usage
x-creative generate "Explore new user growth strategies"

# With parameters
x-creative generate "Research problem" \
  --num-hypotheses 30 \
  --search-depth 2 \
  --output hypotheses.json

# Quick test
x-creative generate "Test problem" -n 5 -d 1
```

---

### Hypergraph Knowledge Tools: `hkg`

Build and query hypergraph knowledge graphs (experimental):

```bash
# Import hypergraph data from target domain YAML
x-creative hkg ingest --source yaml \
  --path x_creative/config/target_domains/open_source_development.yaml \
  --output local_data/hkg_store.json

# Build index (with optional embedding index)
x-creative hkg build-index --store local_data/hkg_store.json
x-creative hkg build-index --store local_data/hkg_store.json --embedding

# Query shortest hyperpaths
x-creative hkg traverse --store local_data/hkg_store.json \
  --start "entropy,temperature" --end "volatility" --K 3

# View hypergraph statistics
x-creative hkg stats --store local_data/hkg_store.json
```

---

### Configuration Management: `config`

```bash
# Show current configuration
x-creative config show

# Initialize user config file (~/.config/x-creative/config.yaml)
x-creative config init

# Show config file path
x-creative config path

# Validate configuration (3 stages: static check → API connectivity → model availability)
x-creative config check

# Static check only
x-creative config check --quick
```

---

### ConceptSpace Management: `concept-space`

ConceptSpace defines the rule space for transformational creativity:

```bash
# Validate ConceptSpace YAML
x-creative concept-space validate path/to/concept_space.yaml

# Compare differences between two versions
x-creative concept-space diff old_space.yaml new_space.yaml
```

---

### Command Quick Reference

| Command | Purpose |
|---------|---------|
| `x-creative answer -q "question"` | Single-entry deep research (recommended, runs full workflow) |
| `x-creative generate "question"` | Quick hypothesis generation (no session needed) |
| `x-creative session new "topic"` | Create a new research session |
| `x-creative run problem -d "description"` | Define the research problem |
| `x-creative run biso` | Distant domain association |
| `x-creative run search` | Hypothesis space expansion |
| `x-creative run verify` | Dual-model verification & filtering |
| `x-creative run solve` | Deep reasoning & solving |
| `x-creative run all -d "description"` | Run all base stages at once |
| `x-creative show <stage>` | View results for a stage |
| `x-creative domains list` | List source domain library |
| `x-creative config check` | Validate configuration |
| `x-creative hkg ingest` | Import hypergraph data |
| `xc-domain` | Source domain TUI manager |
| `xc-target` | Target domain TUI manager |

---

## Python API

### Single-Entry AnswerEngine (Recommended)

```python
import asyncio
from x_creative.answer.engine import AnswerEngine
from x_creative.answer.types import AnswerConfig

async def main():
    # Simplest usage: one question, full workflow runs automatically
    engine = AnswerEngine()
    pack = await engine.answer("How to discover new solutions using cross-disciplinary perspectives")

    # Output Markdown report
    print(pack.answer_md)

    # Access structured data
    print(f"Target domain: {pack.answer_json['metadata']['target_domain']}")
    print(f"Hypotheses: {pack.answer_json['metadata']['total_hypotheses_generated']}")

asyncio.run(main())
```

```python
# Custom configuration
config = AnswerConfig(
    budget=200,
    mode="exhaustive",          # quick / deep_research / exhaustive
    target_domain="auto",       # auto = auto-detect
    search_depth=4,
    search_breadth=8,
    hkg_enabled=True,
    saga_enabled=True,
    fresh=False,                # True = LLM generates domains from scratch
)

engine = AnswerEngine(config=config)
pack = await engine.answer("Your research question")

# Handle clarification requests
if pack.needs_clarification:
    print(f"Clarification needed: {pack.clarification_question}")
    pack = await engine.answer("Question with additional context")
```

### CreativityEngine (Low-Level API)

```python
import asyncio
from x_creative.core.types import ProblemFrame, SearchConfig
from x_creative.creativity.engine import CreativityEngine

async def main():
    problem = ProblemFrame(
        description="Explore innovative methods to improve system reliability",
        target_domain="engineering",
        constraints=["No hardware cost increase", "Maintain backward compatibility"],
        context={"system_type": "distributed", "scale": "large"},
    )

    config = SearchConfig(
        num_hypotheses=20,
        search_depth=2,
        search_breadth=3,
    )

    engine = CreativityEngine()
    try:
        hypotheses = await engine.generate(problem, config)
        for hyp in hypotheses[:5]:
            score = hyp.final_score if hyp.final_score is not None else hyp.composite_score()
            print(f"[{score:.1f}] {hyp.description}")
            print(f"  Observable: {hyp.observable}")
    finally:
        await engine.close()

asyncio.run(main())
```

### Exploring the Domain Library

```python
from x_creative.core.domain_loader import DomainLibrary

library = DomainLibrary.from_target_domain("open_source_development")

for domain in library:
    print(f"{domain.id}: {domain.name}")

queueing = library.get("queueing_theory")
for structure in queueing.structures:
    print(f"  - {structure.name}: {structure.description}")
```

---

## Configuration System

Configuration is loaded in the following priority order (higher priority overrides lower):

1. **Environment variables** (highest priority)
2. **`.env` file** (project directory)
3. **User config file** (`~/.config/x-creative/config.yaml`)
4. **Default values**

> Recommended: Use the `.env` file to manage configuration. Copy `.env.example` to get started.

### .env File Configuration (Recommended)

```bash
cp .env.example .env
# Edit .env to fill in API Key and other settings
```

Example `.env` file contents:

```bash
# API Keys (required)
OPENROUTER_API_KEY=sk-or-v1-your-key-here
X_CREATIVE_DEFAULT_PROVIDER=openrouter

# Optional API Keys
# YUNWU_API_KEY=your-yunwu-key-here
# BRAVE_SEARCH_API_KEY=your-brave-api-key-here

# Basic configuration
X_CREATIVE_DEFAULT_NUM_HYPOTHESES=50
X_CREATIVE_DEFAULT_SEARCH_DEPTH=3

# Score weights (all five must sum to 1.0)
X_CREATIVE_SCORE_WEIGHT_DIVERGENCE=0.21
X_CREATIVE_SCORE_WEIGHT_TESTABILITY=0.26
X_CREATIVE_SCORE_WEIGHT_RATIONALE=0.21
X_CREATIVE_SCORE_WEIGHT_ROBUSTNESS=0.17
X_CREATIVE_SCORE_WEIGHT_FEASIBILITY=0.15
```

### User Config File

```bash
x-creative config init  # Creates ~/.config/x-creative/config.yaml
```

```yaml
openrouter:
  api_key: "sk-or-v1-your-key-here"
default_provider: "openrouter"
brave_search:
  api_key: "your-brave-api-key-here"
default_num_hypotheses: 50
default_search_depth: 3
```

### Model Routing Configuration

Different tasks use different models. Below are the defaults (overridable via `.env`):

| Task | Default Model | Temperature | Description |
|------|--------------|-------------|-------------|
| creativity | anthropic/claude-sonnet-4 | 0.9 | Creative generation, distant domain association |
| analogical_mapping | anthropic/claude-sonnet-4 | 0.7 | Analogical mapping |
| structured_search | openai/gpt-5.2 | 0.5 | Graph of Thoughts search |
| hypothesis_scoring | anthropic/claude-3-haiku | 0.3 | Hypothesis scoring |
| logic_verification | openai/gpt-5.2 | 0.2 | Logic verifier |
| novelty_verification | google/gemini-3-flash-preview | 0.3 | Novelty verifier |
| reasoner_step | anthropic/claude-sonnet-4 | 0.3 | Reasoner multi-step reasoning |
| talker_output | anthropic/claude-sonnet-4 | 0.2 | Talker solution generation |
| saga_adversarial | google/gemini-3-flash-preview | 0.4 | SAGA adversarial evaluation |
| blend_expansion | anthropic/claude-sonnet-4 | 0.8 | Conceptual Blending |
| transform_space | openai/gpt-5.2 | 0.6 | Transformational Creativity |

Each task has a fallback model list for automatic switching when the primary model fails.

Custom model configuration (in `.env`):

```bash
# Override model for a specific task
X_CREATIVE_TASK_ROUTING__CREATIVITY__MODEL=anthropic/claude-3-opus
X_CREATIVE_TASK_ROUTING__CREATIVITY__TEMPERATURE=0.95

# Override verifier models
X_CREATIVE_VERIFIERS__LOGIC__MODEL=google/gemini-3-pro-preview
X_CREATIVE_VERIFIERS__NOVELTY__MODEL=google/gemini-3-pro-preview
```

> Model names must use the `provider/model` format.

### Experimental Feature Configuration

The following features are disabled by default and must be explicitly enabled in `.env`:

<details>
<summary><b>HKG Hypergraph Knowledge Grounding</b></summary>

```bash
X_CREATIVE_HKG_ENABLED=true
X_CREATIVE_HKG_STORE_PATH=local_data/hkg_store.json
X_CREATIVE_HKG_K=3                              # Number of shortest paths to return
X_CREATIVE_HKG_IS=1                              # Minimum shared nodes between adjacent hyperedges
X_CREATIVE_HKG_ENABLE_STRUCTURAL_SCORING=true    # Enable VERIFY structural evidence scoring
X_CREATIVE_HKG_ENABLE_HYPERBRIDGE=false          # Enable hyperbridge
```
</details>

<details>
<summary><b>MOME Quality-Diversity Archive</b></summary>

```bash
X_CREATIVE_MOME_ENABLED=true
X_CREATIVE_MOME_CELL_CAPACITY=10                 # Max hypotheses per grid cell
```
</details>

<details>
<summary><b>QD-Pareto Selection</b></summary>

```bash
X_CREATIVE_PARETO_SELECTION_ENABLED=true
X_CREATIVE_PARETO_NOVELTY_BINS=5
X_CREATIVE_PARETO_WN_MIN=0.15
X_CREATIVE_PARETO_WN_MAX=0.55
X_CREATIVE_PARETO_GAMMA=2.0
```
</details>

<details>
<summary><b>C-K Dual-Space Scheduling</b></summary>

```bash
X_CREATIVE_CK_ENABLED=true
X_CREATIVE_CK_MIN_PHASE_DURATION_S=10.0          # Anti-oscillation minimum phase duration (seconds)
X_CREATIVE_CK_MAX_K_EXPANSION_PER_SESSION=5      # Max K-expansions per session
```
</details>

<details>
<summary><b>Creativity Search Operators</b></summary>

```bash
X_CREATIVE_ENABLE_BLENDING=true                   # Conceptual Blending
X_CREATIVE_ENABLE_TRANSFORM_SPACE=true            # Transformational Creativity
X_CREATIVE_MAX_BLEND_PAIRS=3
X_CREATIVE_MAX_TRANSFORM_HYPOTHESES=2
X_CREATIVE_RUNTIME_PROFILE=research               # interactive / research
```
</details>

<details>
<summary><b>SAGA Cognitive Budget Allocation</b></summary>

All 7 items must sum to 100%:

```bash
X_CREATIVE_COGNITIVE_BUDGET_EMERGENCY_RESERVE=10.0
X_CREATIVE_COGNITIVE_BUDGET_DOMAIN_AUDIT=9.0
X_CREATIVE_COGNITIVE_BUDGET_BISO_MONITOR=13.5
X_CREATIVE_COGNITIVE_BUDGET_SEARCH_MONITOR=13.5
X_CREATIVE_COGNITIVE_BUDGET_VERIFY_MONITOR=18.0
X_CREATIVE_COGNITIVE_BUDGET_ADVERSARIAL=22.5
X_CREATIVE_COGNITIVE_BUDGET_GLOBAL_REVIEW=13.5
```
</details>

<details>
<summary><b>VERIFY Confidence & Anomaly Detection</b></summary>

```bash
X_CREATIVE_MULTI_SAMPLE_EVALUATIONS=3             # Multi-sample count (1-9)
X_CREATIVE_POSITION_BIAS_CONFIDENCE_FACTOR=0.7    # Position bias decay factor
X_CREATIVE_FINAL_SCORE_LOGIC_WEIGHT=0.4           # Logic verification weight (must sum to 1.0 with below)
X_CREATIVE_FINAL_SCORE_NOVELTY_WEIGHT=0.6         # Novelty verification weight
X_CREATIVE_MAPPING_QUALITY_GATE_ENABLED=true      # Mapping quality gate
X_CREATIVE_MAPPING_QUALITY_GATE_THRESHOLD=6.0     # Minimum mapping quality score
X_CREATIVE_MAX_CONSTRAINTS=15                      # Non-user constraint pool limit
```
</details>

> For a complete list of configuration options, see the `.env.example` file.

---

## Target Domain Configuration

X-Creative defines target domains via YAML files. **The filename (without `.yaml`) is the domain ID**.

### Config File Lookup Order

| Priority | Directory | Description |
|----------|-----------|-------------|
| 1 | `~/.config/x-creative/domains/` | User-defined |
| 2 | `x_creative/config/target_domains/` | Built-in |

### Built-in Domains

| Domain ID | Name | Description |
|-----------|------|-------------|
| `open_source_development` | Open Source Software Development | Open-source collaboration, maintenance, and community governance research |

### Creating Custom Domains

```bash
# Create the config directory
mkdir -p ~/.config/x-creative/domains

# Create a YAML file (filename = domain ID)
vim ~/.config/x-creative/domains/product.yaml
```

YAML format:

```yaml
id: product
name: Product Innovation
description: Internet product feature innovation and user experience optimization

# Optional: domain constraints
constraints:
  - name: user_value
    description: Features must create clear value for users
    severity: critical       # critical | important | advisory
  - name: feasibility
    description: Solutions must be implementable with the existing tech stack
    severity: important

# Optional: evaluation criteria
evaluation_criteria:
  - User value clarity
  - Implementation complexity
  - Measurable success metrics

# Optional: anti-patterns
anti_patterns:
  - Technology for technology's sake, ignoring user needs
  - Feature bloat without focus

# Optional: terminology
terminology:
  DAU: Daily Active Users
  Retention rate: Percentage of users who continue using the product after a given period

# Optional: stale ideas (for novelty checking)
stale_ideas:
  - Outdated idea 1

# Embedded source domains
source_domains:
  - id: queueing_theory
    name: Queueing Theory
    name_en: Queueing Theory
    description: Mathematical theory studying queuing phenomena
    structures:
      - id: queue_dynamics
        name: Queue Dynamics
        description: Relationship between arrival rate and service rate
        key_variables: [arrival_rate, service_rate, queue_length]
        dynamics: Queue grows when arrival rate exceeds service rate
    target_mappings:
      - structure: queue_dynamics
        target: User waiting experience
        observable: Average wait time / user abandonment rate
```

Using a custom domain:

```bash
x-creative run problem -d "Improve user retention" --target-domain product
x-creative answer -q "How to improve user retention" --target product
```

You can also use the `xc-target` TUI tool to interactively create target domains.

---

## Output Format

### Hypothesis Structure

Each generated hypothesis contains the following fields:

```json
{
  "id": "hyp_abc12345",
  "description": "Queueing theory-based issue response priority model",
  "source_domain": "queueing_theory",
  "source_structure": "queue_dynamics",
  "analogy_explanation": "The arrival rate difference between issue submission and processing resembles service pressure in queueing systems...",
  "observable": "weekly_active_users / total_installs",
  "mapping_table": [...],
  "failure_modes": [...],
  "scores": {
    "divergence": 8.5,
    "testability": 9.0,
    "rationale": 8.0,
    "robustness": 7.5,
    "feasibility": 8.2
  },
  "final_score": 7.9
}
```

### Scoring Dimensions

| Dimension | Description | High Score Characteristics |
|-----------|-------------|--------------------------|
| **Divergence** | Semantic distance from known methods | Uses rare domain mappings, conceptually novel |
| **Testability** | Can it be converted into testable signals | Clear formula, accessible data, immediately implementable |
| **Rationale** | Does it have reasonable domain mechanisms | Clear causal logic, not purely statistical |
| **Robustness** | Overfitting risk assessment | Simple conditions, broad applicability, few parameters |
| **Feasibility** | Data and engineering implementation accessibility | Clear data sources, controllable costs, clear path to deployment |

### Session Data Structure

Each session's data is saved in `{X_CREATIVE_DATA_DIR}/{session-id}/`:

```
local_data/<session-id>/
├── session.json              # Session metadata
├── problem.json / .md        # Problem definition
├── biso.json / .md           # BISO results
├── search.json / .md         # SEARCH results
├── verify.json / .md         # VERIFY results
├── solve.json / .md          # SOLVE results
├── answer.json / .md         # Answer Engine final report
└── saga/                     # SAGA data (only when enabled)
    ├── events.jsonl           # Fast Agent event log
    ├── directives.jsonl       # Slow Agent directive log
    └── reasoning_trace.jsonl  # Reasoner reasoning step log
```

---

## Development

### Running Tests

```bash
# Run all unit tests
poetry run pytest tests/unit/ -v

# Run tests with coverage report
poetry run pytest tests/unit/ --cov=x_creative --cov-report=term-missing

# Run integration tests (requires API key)
OPENROUTER_API_KEY=your-key poetry run pytest tests/integration/ -v
```

### Code Quality

```bash
# Code formatting
poetry run ruff format .

# Linting
poetry run ruff check .

# Type checking
poetry run mypy x_creative/
```

---

## Source Code Architecture

```
x_creative/
├── answer/             # Answer Engine (single-entry orchestrator)
│   ├── engine.py       #   AnswerEngine - top-level orchestration
│   ├── types.py        #   AnswerConfig, AnswerPack
│   ├── problem_frame.py #  ProblemFrameBuilder - problem framing
│   ├── target_resolver.py # TargetDomainResolver - target domain inference
│   ├── source_selector.py # SourceDomainSelector - source domain selection
│   └── pack_builder.py #   AnswerPackBuilder - output assembly
├── core/               # Core data types
│   ├── types.py        #   Domain, Hypothesis, ProblemFrame, etc.
│   ├── plugin.py       #   Target domain plugin system
│   └── domain_loader.py #  Source domain library loading
├── llm/                # LLM integration layer
│   ├── client.py       #   OpenRouter / Yunwu client
│   └── router.py       #   Multi-model router (task-level routing + fallback)
├── creativity/         # Creativity engine
│   ├── engine.py       #   Main engine (orchestrates BISO → SEARCH → VERIFY)
│   ├── biso.py         #   BISO distant domain association
│   ├── search.py       #   SEARCH structured search (Graph of Thoughts)
│   └── operators/      #   Search operators: refine, variant, combine, blend, transform, etc.
├── verify/             # Dual-model verification system
│   ├── verifiers.py    #   LogicVerifier, NoveltyVerifier
│   ├── mapping_scorer.py # Mapping quality gate (rules + LLM hybrid scoring)
│   └── scoring.py      #   Composite scoring and final score calculation
├── saga/               # SAGA dual-process cognitive architecture
│   ├── coordinator.py  #   SAGACoordinator (Fast/Slow Agent orchestration)
│   ├── solve.py        #   TalkerReasonerSolver (multi-step reasoning)
│   ├── reasoner.py     #   Reasoner (System 2, 7-step reasoning)
│   ├── belief.py       #   BeliefState management
│   ├── detectors/      #   Statistical anomaly detectors
│   ├── auditors/       #   Domain constraint auditors
│   └── memory/         #   Cross-session pattern memory
├── hkg/                # Hypergraph Knowledge Grounding
│   ├── store.py        #   HypergraphStore
│   ├── traversal.py    #   k-shortest hyperpaths
│   ├── matcher.py      #   NodeMatcher (exact/alias/embedding)
│   └── expand.py       #   hyperpath_expand + hyperbridge
├── config/             # Configuration
│   ├── settings.py     #   Pydantic Settings (100+ config fields)
│   ├── checker.py      #   Config validation (3-stage check)
│   └── target_domains/ #   Built-in target domain YAMLs
├── session/            # Session management
│   ├── manager.py      #   SessionManager
│   └── report.py       #   Stage report generation
├── domain_manager/     # xc-domain TUI tool
├── target_manager/     # xc-target TUI tool
└── cli/                # Command-line interface
    ├── main.py         #   Typer entry point
    ├── answer.py       #   answer subcommand
    ├── run.py          #   run subcommand
    ├── session.py      #   session subcommand
    ├── show.py         #   show subcommand
    ├── hkg.py          #   hkg subcommand
    └── concept_space_cli.py # concept-space subcommand
```

---

## Acknowledgments

- [OpenRouter](https://openrouter.ai/) / [Yunwu](https://yunwu.ai/) — Unified LLM API interface
- [Bisociation](https://en.wikipedia.org/wiki/Bisociation) — Arthur Koestler's creativity theory
- [Graph of Thoughts](https://arxiv.org/abs/2308.09687) — Graph-structured thought exploration
- [Conceptual Blending](https://en.wikipedia.org/wiki/Conceptual_blending) — Fauconnier & Turner's conceptual integration theory
- [Transformational Creativity](https://en.wikipedia.org/wiki/Computational_creativity) — Margaret Boden's transformational creativity theory
- [C-K Theory](https://en.wikipedia.org/wiki/C-K_theory) — Hatchuel & Weil's concept-knowledge design theory
- [MAP-Elites](https://arxiv.org/abs/1504.04909) — Quality-diversity optimization algorithm
- [Higher-Order Knowledge Representations](https://arxiv.org/abs/2601.04878) — MIT higher-order knowledge representation (HKG theoretical foundation)

## License

MIT License
