# Engram — Evaluation Methodology

## Running the Evals

```bash
cd backend
cp .env.example .env
# Add your DASHSCOPE_API_KEY to .env (optional for offline suites)
python -m evals.harness --verbose
```

The harness runs three suites entirely offline (no LLM API call needed for the contradiction and retrieval suites — they test the apply logic and retrieval ranking directly against ground-truth fixtures).

## Suite 1: Contradiction Detection

Tests whether the Forgetting Engine correctly resolves conflicting memories.

**Method:** For each case, two memories are written with controlled age offsets. The ground-truth finding (what Qwen3 thinking-mode would output) is injected directly to test `_apply_findings()` — the logic that actually deprecates memories in the database.

This tests the critical path: given a LLM finding, does the system correctly update the memory store?

**8 test cases:**
- 6 true contradictions (location updates, API deprecations, schedule changes, metric updates)
- 2 non-contradictions (additive facts that should both survive)

**Metrics:**
- **Precision** = TP / (TP + FP) — did we deprecate something we shouldn't have?
- **Recall** = TP / (TP + FN) — did we miss a real contradiction?
- **F1** = harmonic mean

**Results: Precision=1.000, Recall=1.000, F1=1.000**

## Suite 2: Multi-signal Retrieval

Tests whether the retrieval system surfaces the most relevant memory for a given query, within a token budget.

**Method:** Seed memories are written and activated. A query is issued. We check whether the expected content appears in the top-1 result and (for R005) that deprecated memories are excluded.

**5 test cases:**
- R001: Revenue fact (semantic)
- R002: Deployment procedure (procedural)
- R003: Database config (entity overlap)
- R004: Token budget enforcement — high-importance memory must rank first even under a tight 300-token budget with larger low-importance memories competing
- R005: Deprecated memory exclusion — Lagos location deprecated, only London returned

**Metrics:**
- **Top-1 Accuracy** = correct memory ranked first
- **MRR (Mean Reciprocal Rank)** = 1/rank of first correct result, averaged

**Results: Top-1 Accuracy=100%, MRR=1.000**

R004 is the most technically interesting: even with a 300-token budget and three large 200-token padding memories scoring higher on recency, the `importance_score=0.9` memory gets packed in first because the fused relevance score weights importance at 0.10 — enough to differentiate it.

## Suite 3: Decay Correctness

Tests whether the decay scoring formula archives the right memories.

**Decay formula:**
```
score = 0.4 × recency + 0.3 × frequency + 0.3 × importance
recency = exp(-0.023 × age_days)   # half-life ≈ 30 days
frequency = min(1.0, access_count / 10)
archive if score < 0.06
```

**3 test cases:**
- D001: Fresh (0 days), importance=0.9 → should NOT archive (score=0.67)
- D002: Old (100 days), importance=0.04, never accessed → SHOULD archive (score=0.052)
- D003: Old (60 days), importance=0.6, accessed 25× → should NOT archive (score=0.58)

**Results: 3/3 correct**

## Baseline Comparison

The learning curve visualization compares Engram against a stateless baseline (same Qwen3 model with no memory system). After N sessions on the same workflow:

| Metric | Session 1 | Session 5 | Change |
|--------|-----------|-----------|--------|
| Tokens used | ~3,200 | ~800 | −75% |
| Steps taken | ~8 | ~2 | −75% |
| Task success | variable | consistent | ↑ |

*(Numbers from live demo sessions — exact values vary by workflow complexity)*

## Limitations

- Contradiction detection precision/recall are measured against injected ground-truth findings, not live Qwen3 outputs. Live LLM accuracy will vary by model version and prompt tuning.
- Retrieval uses offline hash-based embeddings in this environment. With `DASHSCOPE_API_KEY` set, DashScope `text-embedding-v3` produces semantically richer vectors and higher retrieval accuracy.
- The learning curve numbers depend on the workflow complexity and whether the playbook accurately captures the session steps.
