# BISO LLM Pool Design

## Problem

Bisociation uses a single LLM for all domain analogy generation. Even with high temperature, output diversity is limited by the model's inherent biases.

## Solution

Add a configurable LLM pool (`biso_pool`) for the Bisociation process. When configured, each domain randomly selects a model from the pool, increasing diversity through model-level variation.

## Design Decisions

- **Scope**: BISO only. Other tasks (scoring, verification) are unaffected.
- **Granularity**: Per domain — each `generate_analogies(domain)` call picks one random model.
- **Temperature**: Shared from the `creativity` task config. No per-model temperature in pool.
- **Fallback**: Empty pool = current behavior (creativity task's primary model).
- **Failure handling**: Pool model failure falls back to creativity task's fallback chain.

## Configuration

### config.yaml

```yaml
biso_pool:
  - "google/gemini-2.5-pro"
  - "anthropic/claude-sonnet-4"
  - "deepseek/deepseek-r1"
```

### Environment variable

```
X_CREATIVE_BISO_POOL='["google/gemini-2.5-pro","anthropic/claude-sonnet-4"]'
```

### Not configured (default)

```yaml
biso_pool: []  # Falls back to creativity task's primary model
```

## Changes

### 1. `x_creative/config/settings.py`

Add `biso_pool: list[str]` field to `Settings`.

### 2. `x_creative/llm/router.py`

Add `model_override` parameter to `ModelRouter.complete()`. When set, uses this model instead of the task's primary model, but retains the fallback chain.

### 3. `x_creative/creativity/biso.py`

In `generate_analogies()`, if `biso_pool` is non-empty, randomly select a model and pass it as `model_override` to `router.complete()`.

### 4. Tests

- `BISOModule`: Verify `model_override` is passed when pool is configured.
- `ModelRouter`: Verify `model_override` takes precedence over config primary model.
- `Settings`: Verify `biso_pool` loads from config/env.
