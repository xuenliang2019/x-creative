# BISO Hypothesis Deduplication Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add post-generation semantic deduplication to BISOModule so redundant hypotheses are discarded before entering the SEARCH stage.

**Architecture:** After all domains finish parallel generation, `_deduplicate_hypotheses()` sends hypothesis summaries to an LLM (via `knowledge_extraction` task) which returns duplicate groups. The first hypothesis in each group is kept; the rest are discarded. Controlled by `biso_dedup_enabled` setting.

**Tech Stack:** Python, Pydantic Settings, pytest, AsyncMock

---

### Task 1: Add `biso_dedup_enabled` to Settings

**Files:**
- Modify: `x_creative/config/settings.py:416` (after `biso_pool` field)
- Test: `tests/unit/test_config.py`

**Step 1: Write the failing test**

In `tests/unit/test_config.py`, add to `TestSettings` (after existing BISO pool tests):

```python
def test_biso_dedup_enabled_default(self) -> None:
    """biso_dedup_enabled defaults to True."""
    settings = Settings(_env_file=None)
    assert settings.biso_dedup_enabled is True
```

**Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/unit/test_config.py::TestSettings::test_biso_dedup_enabled_default -v`
Expected: FAIL — `biso_dedup_enabled` field doesn't exist yet.

**Step 3: Implement — add `biso_dedup_enabled` field**

In `x_creative/config/settings.py`, add after the `biso_pool` field (after line 416):

```python
    # BISO semantic deduplication
    biso_dedup_enabled: bool = True
```

**Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/unit/test_config.py::TestSettings::test_biso_dedup_enabled_default -v`
Expected: PASS

**Step 5: Commit**

```bash
git add x_creative/config/settings.py tests/unit/test_config.py
git commit -m "feat: add biso_dedup_enabled setting"
```

---

### Task 2: Add `BISO_DEDUP_PROMPT` template

**Files:**
- Modify: `x_creative/creativity/prompts.py` (add after `BISO_ANALOGY_PROMPT`, before `SEARCH_EXPAND_PROMPT`)
- No test file for this task (prompt template is a string constant, tested via integration in Task 4)

**Step 1: Add the prompt template**

In `x_creative/creativity/prompts.py`, add after `BISO_ANALOGY_PROMPT` (after line 86) and before `SEARCH_EXPAND_PROMPT` (line 88):

```python
BISO_DEDUP_PROMPT = """You are an expert at identifying semantically duplicate research hypotheses.

## Task
Given a numbered list of hypotheses, identify groups of hypotheses that express the same core insight or mechanism — even if they use different wording, come from different source domains, or propose different observables.

Two hypotheses are "duplicates" if a researcher would say: "These are essentially the same idea."

## Hypotheses
{hypotheses_text}

## Output Format
Return a JSON object with duplicate groups. Each group is a list of hypothesis indices (0-based) that are semantically equivalent:

```json
{{
    "duplicate_groups": [[0, 3, 7], [2, 5]],
    "reasoning": "Brief explanation of why each group is considered duplicate"
}}
```

Rules:
- Only include groups with 2+ members (skip unique hypotheses).
- If no duplicates exist, return: {{"duplicate_groups": [], "reasoning": "All hypotheses are semantically distinct."}}
- Be conservative: when in doubt, do NOT mark as duplicate. Only flag clear semantic overlaps.
"""
```

**Step 2: Commit**

```bash
git add x_creative/creativity/prompts.py
git commit -m "feat: add BISO_DEDUP_PROMPT template"
```

---

### Task 3: Implement `_deduplicate_hypotheses()` in BISOModule

**Files:**
- Modify: `x_creative/creativity/biso.py` (add method + wire into `generate_all_analogies`)
- Test: `tests/unit/test_creativity.py`

**Step 1: Write the failing tests**

In `tests/unit/test_creativity.py`, add to `TestBISOModule` (after existing pool tests):

```python
@pytest.mark.asyncio
async def test_deduplicate_removes_semantic_duplicates(
    self, sample_domain: Domain, sample_problem: ProblemFrame
) -> None:
    """Dedup should remove hypotheses identified as duplicates by LLM."""
    from x_creative.creativity.biso import BISOModule

    biso = BISOModule()
    biso._biso_dedup_enabled = True

    h1 = Hypothesis(
        id="h1", description="Idea A", source_domain="d1",
        source_structure="s1", analogy_explanation="", observable="obs_a",
    )
    h2 = Hypothesis(
        id="h2", description="Idea B (unique)", source_domain="d2",
        source_structure="s2", analogy_explanation="", observable="obs_b",
    )
    h3 = Hypothesis(
        id="h3", description="Idea A rephrased", source_domain="d3",
        source_structure="s3", analogy_explanation="", observable="obs_a_v2",
    )

    dedup_response = json.dumps({
        "duplicate_groups": [[0, 2]],
        "reasoning": "h1 and h3 express the same core idea",
    })

    with patch.object(biso, "_router") as mock_router:
        mock_router.complete = AsyncMock(
            return_value=MagicMock(content=dedup_response)
        )
        result = await biso._deduplicate_hypotheses([h1, h2, h3])

    assert len(result) == 2
    assert result[0].id == "h1"
    assert result[1].id == "h2"

@pytest.mark.asyncio
async def test_deduplicate_preserves_all_when_unique(
    self, sample_domain: Domain, sample_problem: ProblemFrame
) -> None:
    """When LLM says no duplicates, all hypotheses are kept."""
    from x_creative.creativity.biso import BISOModule

    biso = BISOModule()
    biso._biso_dedup_enabled = True

    h1 = Hypothesis(
        id="h1", description="Idea A", source_domain="d1",
        source_structure="s1", analogy_explanation="", observable="obs_a",
    )
    h2 = Hypothesis(
        id="h2", description="Idea B", source_domain="d2",
        source_structure="s2", analogy_explanation="", observable="obs_b",
    )

    dedup_response = json.dumps({
        "duplicate_groups": [],
        "reasoning": "All hypotheses are semantically distinct.",
    })

    with patch.object(biso, "_router") as mock_router:
        mock_router.complete = AsyncMock(
            return_value=MagicMock(content=dedup_response)
        )
        result = await biso._deduplicate_hypotheses([h1, h2])

    assert len(result) == 2

@pytest.mark.asyncio
async def test_deduplicate_disabled_skips_llm_call(self) -> None:
    """When biso_dedup_enabled is False, no LLM call is made."""
    from x_creative.creativity.biso import BISOModule

    biso = BISOModule()
    biso._biso_dedup_enabled = False

    h1 = Hypothesis(
        id="h1", description="Idea A", source_domain="d1",
        source_structure="s1", analogy_explanation="", observable="obs_a",
    )
    h2 = Hypothesis(
        id="h2", description="Idea B", source_domain="d2",
        source_structure="s2", analogy_explanation="", observable="obs_b",
    )

    with patch.object(biso, "_router") as mock_router:
        mock_router.complete = AsyncMock()
        result = await biso._deduplicate_hypotheses([h1, h2])

    assert len(result) == 2
    mock_router.complete.assert_not_called()

@pytest.mark.asyncio
async def test_deduplicate_graceful_on_llm_failure(self) -> None:
    """When LLM call fails, return original list unchanged."""
    from x_creative.creativity.biso import BISOModule

    biso = BISOModule()
    biso._biso_dedup_enabled = True

    h1 = Hypothesis(
        id="h1", description="Idea A", source_domain="d1",
        source_structure="s1", analogy_explanation="", observable="obs_a",
    )
    h2 = Hypothesis(
        id="h2", description="Idea B", source_domain="d2",
        source_structure="s2", analogy_explanation="", observable="obs_b",
    )

    with patch.object(biso, "_router") as mock_router:
        mock_router.complete = AsyncMock(side_effect=RuntimeError("LLM failed"))
        result = await biso._deduplicate_hypotheses([h1, h2])

    assert len(result) == 2
    assert result[0].id == "h1"
    assert result[1].id == "h2"

@pytest.mark.asyncio
async def test_deduplicate_skips_single_hypothesis(self) -> None:
    """Single hypothesis should skip dedup entirely."""
    from x_creative.creativity.biso import BISOModule

    biso = BISOModule()
    biso._biso_dedup_enabled = True

    h1 = Hypothesis(
        id="h1", description="Idea A", source_domain="d1",
        source_structure="s1", analogy_explanation="", observable="obs_a",
    )

    with patch.object(biso, "_router") as mock_router:
        mock_router.complete = AsyncMock()
        result = await biso._deduplicate_hypotheses([h1])

    assert len(result) == 1
    mock_router.complete.assert_not_called()
```

**Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/unit/test_creativity.py::TestBISOModule::test_deduplicate_removes_semantic_duplicates tests/unit/test_creativity.py::TestBISOModule::test_deduplicate_preserves_all_when_unique tests/unit/test_creativity.py::TestBISOModule::test_deduplicate_disabled_skips_llm_call tests/unit/test_creativity.py::TestBISOModule::test_deduplicate_graceful_on_llm_failure tests/unit/test_creativity.py::TestBISOModule::test_deduplicate_skips_single_hypothesis -v`
Expected: FAIL — `_deduplicate_hypotheses` method doesn't exist.

**Step 3: Implement `_deduplicate_hypotheses()` in BISOModule**

In `x_creative/creativity/biso.py`:

1. Add import at top (line 16, after existing prompt import):

```python
from x_creative.creativity.prompts import BISO_ANALOGY_PROMPT, BISO_DEDUP_PROMPT
```

(Replace the existing single import of `BISO_ANALOGY_PROMPT`.)

2. In `__init__`, after `self._biso_pool` (line 52), add:

```python
self._biso_dedup_enabled: bool = settings.biso_dedup_enabled
```

3. Add the `_deduplicate_hypotheses` method after `_parse_analogies` (after line 251):

```python
async def _deduplicate_hypotheses(
    self, hypotheses: list[Hypothesis]
) -> list[Hypothesis]:
    """Remove semantically duplicate hypotheses using LLM judgment.

    Args:
        hypotheses: List of hypotheses to deduplicate.

    Returns:
        Deduplicated list (order preserved, first in each group kept).
    """
    if not self._biso_dedup_enabled or len(hypotheses) <= 1:
        return hypotheses

    # Build summary text for LLM
    lines: list[str] = []
    for i, h in enumerate(hypotheses):
        lines.append(f"[{i}] {h.description} | Observable: {h.observable}")
    hypotheses_text = "\n".join(lines)

    prompt = BISO_DEDUP_PROMPT.format(hypotheses_text=hypotheses_text)

    try:
        result = await self._router.complete(
            task="knowledge_extraction",
            messages=[{"role": "user", "content": prompt}],
        )

        # Parse duplicate groups from response
        json_str = result.content
        json_start = json_str.find("{")
        json_end = json_str.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            data = json.loads(json_str[json_start:json_end])
        else:
            data = json.loads(json_str)

        duplicate_groups: list[list[int]] = data.get("duplicate_groups", [])

        # Collect indices to remove (all except first in each group)
        remove_indices: set[int] = set()
        for group in duplicate_groups:
            if not isinstance(group, list) or len(group) < 2:
                continue
            valid = [idx for idx in group if isinstance(idx, int) and 0 <= idx < len(hypotheses)]
            if len(valid) >= 2:
                for idx in valid[1:]:
                    remove_indices.add(idx)

        deduplicated = [h for i, h in enumerate(hypotheses) if i not in remove_indices]

        logger.info(
            "biso_dedup_complete",
            original_count=len(hypotheses),
            deduplicated_count=len(deduplicated),
            removed_count=len(hypotheses) - len(deduplicated),
            num_groups=len(duplicate_groups),
        )

        return deduplicated

    except Exception as e:
        logger.warning(
            "biso_dedup_failed",
            error=str(e),
            hypothesis_count=len(hypotheses),
        )
        return hypotheses
```

4. In `generate_all_analogies`, after the flatten loop (after line 362) and before the final log (line 364), add the dedup call:

```python
        # Flatten results
        all_hypotheses: list[Hypothesis] = []
        for hypotheses in results:
            all_hypotheses.extend(hypotheses)

        # Semantic deduplication
        all_hypotheses = await self._deduplicate_hypotheses(all_hypotheses)

        logger.info(
```

**Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/unit/test_creativity.py::TestBISOModule -v`
Expected: All PASS (new tests + existing tests).

**Step 5: Run full test suite to check for regressions**

Run: `poetry run pytest tests/unit/ -v`
Expected: All PASS.

**Step 6: Commit**

```bash
git add x_creative/creativity/biso.py x_creative/creativity/prompts.py tests/unit/test_creativity.py
git commit -m "feat: add BISO hypothesis semantic deduplication"
```

---

### Task 4: Verify `generate_all_analogies` calls dedup

**Files:**
- Test: `tests/unit/test_creativity.py`
- No implementation changes (integration wiring test)

**Step 1: Write the integration test**

In `tests/unit/test_creativity.py`, add to `TestBISOModule`:

```python
@pytest.mark.asyncio
async def test_generate_all_analogies_calls_dedup(
    self, sample_domain: Domain, sample_problem: ProblemFrame
) -> None:
    """generate_all_analogies should call _deduplicate_hypotheses on results."""
    from x_creative.creativity.biso import BISOModule

    biso = BISOModule()

    valid_response = json.dumps([{
        "analogy": "Test analogy",
        "structure_id": "test_structure",
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

    domains = [
        sample_domain.model_copy(update={"id": "d1"}),
        sample_domain.model_copy(update={"id": "d2"}),
    ]

    with patch.object(biso, "_router") as mock_router:
        mock_router.complete = AsyncMock(
            return_value=MagicMock(content=valid_response)
        )
        with patch.object(
            biso, "_deduplicate_hypotheses", wraps=biso._deduplicate_hypotheses
        ) as mock_dedup:
            await biso.generate_all_analogies(
                problem=sample_problem,
                source_domains=domains,
            )
            mock_dedup.assert_called_once()
            # The argument should be a list of hypotheses from both domains
            call_args = mock_dedup.call_args[0][0]
            assert len(call_args) == 2
```

**Step 2: Run test to verify it passes**

Run: `poetry run pytest tests/unit/test_creativity.py::TestBISOModule::test_generate_all_analogies_calls_dedup -v`
Expected: PASS (implementation from Task 3 already wires this).

**Step 3: Run full BISO test suite**

Run: `poetry run pytest tests/unit/test_creativity.py::TestBISOModule -v`
Expected: All PASS.

**Step 4: Commit**

```bash
git add tests/unit/test_creativity.py
git commit -m "test: add integration test for BISO dedup in generate_all_analogies"
```

---

### Task 5: Update config template

**Files:**
- Modify: `x_creative/config/settings.py` (update template in `init_user_config()`)

**Step 1: Add `biso_dedup_enabled` to the config.yaml template**

In `x_creative/config/settings.py`, in the `init_user_config()` function, find the existing `biso_pool` template comment block and add after it:

```yaml
# BISO semantic deduplication (default: true)
# When enabled, uses LLM to identify and discard semantically duplicate
# hypotheses after BISO generation, before entering SEARCH stage.
# biso_dedup_enabled: true
```

**Step 2: Run full test suite**

Run: `poetry run pytest tests/unit/ -v`
Expected: All PASS.

**Step 3: Commit**

```bash
git add x_creative/config/settings.py
git commit -m "docs: add biso_dedup_enabled to config template"
```
