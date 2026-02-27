# BISO Hypothesis Deduplication Design

## Problem

The BISO LLM pool randomly assigns models to source domains, but there is no mechanism to ensure the generated hypotheses are semantically diverse. Different models may produce similar hypotheses for the same or different domains, wasting downstream SEARCH/VERIFY resources.

The existing diversity mechanisms (MOME novelty_distance, Pareto crowding_distance) only operate downstream — they filter after the fact but cannot prevent redundant generation at the BISO stage.

## Solution

Add a **post-generation semantic deduplication** step inside `BISOModule.generate_all_analogies()`. After all domains finish parallel generation, use a single LLM call to identify semantically duplicate hypothesis groups and discard redundant ones.

```
generate_all_analogies():
    domains 并行生成 → flatten all_hypotheses
    → _deduplicate_hypotheses(all_hypotheses)   # NEW
    → return deduplicated
```

## Design Decisions

- **Timing**: Post-generation batch dedup. Preserves current parallel architecture.
- **Method**: LLM semantic judgment (consistent with `domain_manager`'s dedup pattern).
- **Granularity**: Overall semantic similarity — two hypotheses with the same core insight/mechanism count as duplicates even if from different source domains or worded differently.
- **Handling**: Discard duplicates, keep the first generated in each group.
- **Scope**: BISOModule internal only. No new public API surface.
- **Fallback**: LLM dedup failure → log warning, return original list (graceful degradation).

## Configuration

In `Settings`:

```python
biso_dedup_enabled: bool = Field(default=True)
```

No similarity threshold needed — LLM makes the semantic judgment.

## Changes

### 1. `x_creative/config/settings.py`

Add `biso_dedup_enabled: bool` field near existing BISO settings.

### 2. `x_creative/creativity/prompts.py`

Add `BISO_DEDUP_PROMPT` template:

- Input: Numbered list of hypothesis summaries (description + observable only, no mapping_table/failure_modes to save tokens).
- Output: JSON — `{"duplicate_groups": [[1, 5, 12], [3, 8]]}` where indices in the same group are semantically duplicate.
- Each group keeps index 0 (the first), discards the rest.

### 3. `x_creative/creativity/biso.py`

Add `_deduplicate_hypotheses()` private async method:

```python
async def _deduplicate_hypotheses(
    self, hypotheses: list[Hypothesis]
) -> list[Hypothesis]:
```

Logic:
1. If `biso_dedup_enabled` is False or len(hypotheses) <= 1: return as-is.
2. Build summary list: `[(index, description, observable)]` for each hypothesis.
3. If count > 60: split into batches of 60, dedup each batch, then dedup across batches.
4. Call `self._router.complete(task="knowledge_extraction", ...)` with `BISO_DEDUP_PROMPT`.
5. Parse response → identify duplicate groups → keep first in each group.
6. On failure: log warning, return original list.

Call site in `generate_all_analogies()`: after flatten, before return.

### 4. Logging

After dedup completes:

```python
logger.info(
    "biso_dedup_complete",
    original_count=len(all_hypotheses),
    deduplicated_count=len(deduplicated),
    removed_count=len(all_hypotheses) - len(deduplicated),
)
```

### 5. Tests

- `test_deduplicate_removes_semantic_duplicates`: Mock LLM to return duplicate groups, verify correct hypotheses are removed.
- `test_deduplicate_preserves_unique`: Mock LLM to return no duplicates, verify all kept.
- `test_deduplicate_disabled`: Set `biso_dedup_enabled=False`, verify no LLM call made.
- `test_deduplicate_llm_failure_graceful`: Mock LLM to raise, verify original list returned.
- `test_deduplicate_skips_single_hypothesis`: Single hypothesis → no LLM call, returned as-is.

## Edge Cases

| Case | Behavior |
|------|----------|
| 0 or 1 hypotheses | Skip dedup, return as-is |
| LLM call fails | Log warning, return original list |
| All hypotheses duplicate | Keep one per group (never empty) |
| > 60 hypotheses | Batch processing (60 per batch) |
| `biso_dedup_enabled = False` | Skip entirely |
