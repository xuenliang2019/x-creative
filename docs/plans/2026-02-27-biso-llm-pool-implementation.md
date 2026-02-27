# BISO LLM Pool Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a configurable LLM model pool for Bisociation so each domain randomly selects a different model, increasing hypothesis diversity.

**Architecture:** New `biso_pool: list[str]` field in `Settings`. `ModelRouter.complete()` gains a `model_override` parameter. `BISOModule.generate_analogies()` randomly picks a model from the pool per domain call.

**Tech Stack:** Python, Pydantic Settings, pytest, AsyncMock

---

### Task 1: Add `biso_pool` to Settings

**Files:**
- Modify: `x_creative/config/settings.py` (add field around line 406, near other BISO settings)
- Test: `tests/unit/test_config.py`

**Step 1: Write the failing test**

In `tests/unit/test_config.py`, add to `TestSettings`:

```python
def test_biso_pool_default_empty(self) -> None:
    """biso_pool defaults to empty list."""
    settings = Settings(_env_file=None)
    assert settings.biso_pool == []

def test_biso_pool_from_init(self) -> None:
    """biso_pool can be set via constructor."""
    settings = Settings(
        _env_file=None,
        biso_pool=["google/gemini-2.5-pro", "anthropic/claude-sonnet-4"],
    )
    assert len(settings.biso_pool) == 2
    assert "google/gemini-2.5-pro" in settings.biso_pool
```

**Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/unit/test_config.py::TestSettings::test_biso_pool_default_empty tests/unit/test_config.py::TestSettings::test_biso_pool_from_init -v`
Expected: FAIL — `biso_pool` field doesn't exist yet.

**Step 3: Implement — add `biso_pool` field to `Settings`**

In `x_creative/config/settings.py`, add after `biso_max_concurrency` (around line 406):

```python
# BISO LLM pool for diversity
biso_pool: list[str] = Field(default_factory=list)
```

**Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/unit/test_config.py::TestSettings::test_biso_pool_default_empty tests/unit/test_config.py::TestSettings::test_biso_pool_from_init -v`
Expected: PASS

**Step 5: Commit**

```bash
git add x_creative/config/settings.py tests/unit/test_config.py
git commit -m "feat: add biso_pool setting for LLM diversity"
```

---

### Task 2: Add `model_override` to `ModelRouter.complete()`

**Files:**
- Modify: `x_creative/llm/router.py` (modify `complete()` signature and logic)
- Test: `tests/unit/test_llm.py`

**Step 1: Write the failing test**

In `tests/unit/test_llm.py`, add to `TestModelRouter`:

```python
@pytest.mark.asyncio
async def test_router_complete_with_model_override(self) -> None:
    """model_override replaces the primary model but keeps fallbacks."""
    mock_client = MagicMock()
    mock_client.complete = AsyncMock(
        return_value=MagicMock(
            content="Response",
            model="override/model",
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            finish_reason="stop",
        )
    )
    router = ModelRouter(client=mock_client)
    result = await router.complete(
        task="creativity",
        messages=[{"role": "user", "content": "test"}],
        model_override="override/model",
    )
    assert result.content == "Response"
    # Verify the override model was used, not the default creativity model
    call_args = mock_client.complete.call_args
    assert call_args.kwargs["model"] == "override/model"

@pytest.mark.asyncio
async def test_router_complete_override_falls_back_on_failure(self) -> None:
    """When override model fails, router falls back to task fallback chain."""
    mock_client = MagicMock()
    mock_client.complete = AsyncMock(
        side_effect=[
            Exception("Override model failed"),
            MagicMock(
                content="Fallback response",
                model="fallback/model",
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
                finish_reason="stop",
            ),
        ]
    )
    router = ModelRouter(client=mock_client)
    result = await router.complete(
        task="creativity",
        messages=[{"role": "user", "content": "test"}],
        model_override="override/model",
    )
    assert result.content == "Fallback response"
    assert mock_client.complete.call_count == 2
```

**Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/unit/test_llm.py::TestModelRouter::test_router_complete_with_model_override tests/unit/test_llm.py::TestModelRouter::test_router_complete_override_falls_back_on_failure -v`
Expected: FAIL — `model_override` parameter doesn't exist yet.

**Step 3: Implement — add `model_override` to `ModelRouter.complete()`**

In `x_creative/llm/router.py`, modify the `complete()` method signature and the model selection logic:

```python
async def complete(
    self,
    task: str,
    messages: list[dict[str, Any]],
    temperature: float | None = None,
    max_tokens: int | None = None,
    model_override: str | None = None,
    **kwargs: Any,
) -> CompletionResult:
```

And change the model list construction:

```python
config = self.get_config(task)

# model_override replaces the primary model but keeps fallback chain
if model_override:
    models_to_try = [model_override] + config.fallback
else:
    models_to_try = [config.model] + config.fallback
```

**Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/unit/test_llm.py::TestModelRouter -v`
Expected: All PASS (new tests + existing tests unchanged).

**Step 5: Commit**

```bash
git add x_creative/llm/router.py tests/unit/test_llm.py
git commit -m "feat: add model_override parameter to ModelRouter.complete()"
```

---

### Task 3: Wire BISO pool random selection into `BISOModule`

**Files:**
- Modify: `x_creative/creativity/biso.py` (add pool reading + random selection)
- Test: `tests/unit/test_creativity.py`

**Step 1: Write the failing tests**

In `tests/unit/test_creativity.py`, add to `TestBISOModule`:

```python
@pytest.mark.asyncio
async def test_generate_analogies_uses_biso_pool(
    self, sample_domain: Domain, sample_problem: ProblemFrame
) -> None:
    """When biso_pool is configured, a random model from the pool is used."""
    biso = BISOModule()
    # Set pool directly for testing
    biso._biso_pool = ["model-a", "model-b", "model-c"]

    valid_response = json.dumps([{
        "analogy": "Test analogy",
        "structure_id": "predator_prey",
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
        # Verify model_override was passed and is from the pool
        call_kwargs = mock_router.complete.call_args.kwargs
        assert "model_override" in call_kwargs
        assert call_kwargs["model_override"] in ["model-a", "model-b", "model-c"]

@pytest.mark.asyncio
async def test_generate_analogies_no_pool_no_override(
    self, sample_domain: Domain, sample_problem: ProblemFrame
) -> None:
    """When biso_pool is empty, no model_override is passed."""
    biso = BISOModule()
    biso._biso_pool = []

    valid_response = json.dumps([{
        "analogy": "Test analogy",
        "structure_id": "predator_prey",
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
```

**Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/unit/test_creativity.py::TestBISOModule::test_generate_analogies_uses_biso_pool tests/unit/test_creativity.py::TestBISOModule::test_generate_analogies_no_pool_no_override -v`
Expected: FAIL — `_biso_pool` attribute doesn't exist yet in BISOModule.

**Step 3: Implement — add pool logic to BISOModule**

In `x_creative/creativity/biso.py`:

1. Add `import random` at top.

2. In `__init__`, after the existing code, add:
```python
self._biso_pool: list[str] = list(get_settings().biso_pool)
```

3. In `generate_analogies()`, change the `router.complete` call:
```python
# Build optional model override from BISO pool
pool_kwargs: dict[str, Any] = {}
if self._biso_pool:
    selected_model = random.choice(self._biso_pool)
    pool_kwargs["model_override"] = selected_model
    logger.debug("BISO pool selected model", model=selected_model, domain=domain.id)

result = await self._router.complete(
    task="creativity",
    messages=[{"role": "user", "content": prompt}],
    max_tokens=32768,
    **pool_kwargs,
)
```

4. Add `from typing import Any` if not already imported (check first — it's not currently imported in biso.py).

**Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/unit/test_creativity.py::TestBISOModule -v`
Expected: All PASS (new tests + existing tests).

**Step 5: Run full test suite to check for regressions**

Run: `poetry run pytest tests/unit/ -v`
Expected: All PASS.

**Step 6: Commit**

```bash
git add x_creative/creativity/biso.py tests/unit/test_creativity.py
git commit -m "feat: wire BISO LLM pool random model selection"
```

---

### Task 4: Update config template and example

**Files:**
- Modify: `x_creative/config/settings.py` (update template in `init_user_config()`)

**Step 1: Add `biso_pool` to the config.yaml template**

In `x_creative/config/settings.py`, in the `init_user_config()` function, add to the template string after the generation settings section:

```yaml
# BISO LLM pool for diversity (optional)
# When configured, each domain randomly selects a model from this pool.
# When empty or unset, uses the creativity task's primary model.
# biso_pool:
#   - "google/gemini-2.5-pro"
#   - "anthropic/claude-sonnet-4"
#   - "deepseek/deepseek-r1"
```

**Step 2: Run full test suite**

Run: `poetry run pytest tests/unit/ -v`
Expected: All PASS.

**Step 3: Commit**

```bash
git add x_creative/config/settings.py
git commit -m "docs: add biso_pool to config template"
```
