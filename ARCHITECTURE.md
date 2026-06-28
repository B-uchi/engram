# Engram — Architecture

## Overview

Engram is a memory agent that learns workflows, gets faster at them, and self-corrects when they go stale. It combines two mechanisms no existing agent demo tackles together:

- **Procedural memory (1C)**: watches a workflow, distills it into a playbook, executes it faster and with fewer tokens each session
- **The Forgetting Engine (1A)**: a background consolidator that detects contradictions, decays stale memories, and keeps the playbook accurate as the world changes

## System Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                      USER (Browser)                         │
│  ┌────────────┐  ┌──────────────┐  ┌──────────────────────┐ │
│  │  Chat tab  │  │ Memory Store │  │  Learning Curve tab  │ │
│  │ (SSE stream│  │   tab        │  │  (Recharts)          │ │
│  │  per turn) │  │              │  │                      │ │
│  └──────┬─────┘  └──────┬───────┘  └──────────┬───────────┘ │
└─────────┼───────────────┼──────────────────────┼────────────┘
          │               │                      │
          ▼               ▼                      ▼
┌─────────────────────────────────────────────────────────────┐
│              Next.js Frontend (Vercel / ECS)                 │
│  lib/api.ts — typed fetch + SSE stream helper               │
└────────────────────────────┬────────────────────────────────┘
                             │ HTTP / SSE
                             ▼
┌─────────────────────────────────────────────────────────────┐
│              FastAPI Backend (Alibaba Cloud ECS)             │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                   Active Agent                      │   │
│  │           (Qwen3-32B, thinking=False)               │   │
│  │                                                     │   │
│  │  Session start:                                     │   │
│  │    → query_memory() prime context                   │   │
│  │    → get_playbook() load if exists                  │   │
│  │                                                     │   │
│  │  Per turn:                                          │   │
│  │    → query_memory() before answering                │   │
│  │    → write_memory() after learning                  │   │
│  │    → record_step_result() per playbook step         │   │
│  │                                                     │   │
│  │  Session end (reflection):                          │   │
│  │    → write_playbook() distill workflow              │   │
│  │    → deprecate_memory() for stale facts             │   │
│  └────────────────────┬────────────────────────────────┘   │
│                       │ MCP protocol (SSE transport)        │
│  ┌────────────────────▼────────────────────────────────┐   │
│  │              Custom MCP Server (/mcp/sse)            │   │
│  │                                                     │   │
│  │  Tools: write_memory · write_playbook               │   │
│  │         query_memory · get_playbook                 │   │
│  │         deprecate_memory · update_memory            │   │
│  │         list_playbooks · record_step_result         │   │
│  └────────────────────┬────────────────────────────────┘   │
│                       │                                     │
│  ┌────────────────────▼────────────────────────────────┐   │
│  │                  Memory Store                       │   │
│  │                                                     │   │
│  │  SQLite (structured)  +  ChromaDB (vector search)   │   │
│  │                                                     │   │
│  │  Multi-signal retrieval:                            │   │
│  │    0.40 × semantic (DashScope text-embedding-v3)    │   │
│  │    0.25 × BM25 keyword                              │   │
│  │    0.15 × entity/tag overlap                        │   │
│  │    0.10 × recency (exponential decay)               │   │
│  │    0.10 × importance score                          │   │
│  │  → greedy budget-pack into token ceiling            │   │
│  │                                                     │   │
│  │  Tables: memories · playbooks · playbook_steps      │   │
│  │          sessions · consolidation_runs              │   │
│  │          memory_diffs · memory_tags                 │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │            The Forgetting Engine                    │   │
│  │         (Qwen3-32B, thinking=True, async)           │   │
│  │                                                     │   │
│  │  Triggered: session end + cron every 30 min         │   │
│  │                                                     │   │
│  │  Phase 1 — Activate pending memories               │   │
│  │  Phase 2 — Decay scoring                           │   │
│  │    score = 0.4×recency + 0.3×frequency + 0.3×imp   │   │
│  │    score < 0.06 → archive                          │   │
│  │  Phase 3 — Contradiction detection (Qwen3 thinking) │   │
│  │    batch 50 memories → LLM finds conflicts         │   │
│  │    loser deprecated, winner survives               │   │
│  │    audit trail: deprecated_reason + superseded_by  │   │
│  │  Phase 4 — Playbook step review                    │   │
│  │    steps with success_rate < 0.5 → flagged stale   │   │
│  │                                                     │   │
│  │  Output: ConsolidationRun + MemoryDiff[] for UI     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                             │
                             ▼ DashScope API
┌─────────────────────────────────────────────────────────────┐
│                   Alibaba Cloud / Qwen                       │
│                                                             │
│  qwen3-32b          — active agent (non-thinking)           │
│  qwen3-32b          — consolidator (thinking mode)          │
│  text-embedding-v3  — memory vector embeddings              │
└─────────────────────────────────────────────────────────────┘
```

## Memory Schema

```
Memory
  id, session_id, content, summary
  memory_type: episodic | semantic | procedural
  status: pending → active → deprecated | archived
  importance_score: 0.0–1.0 (set at write time)
  access_count, last_accessed_at, recency_score
  deprecated_at, deprecated_reason, superseded_by → Memory

Playbook
  id, name, description, version
  session_count, avg_steps_taken, avg_tokens_used

PlaybookStep
  playbook_id, step_number, title, description
  tool_used, decision_point, edge_cases[]
  success_count, failure_count → success_rate

ConsolidationRun
  started_at, triggered_by
  memories_activated/deprecated/archived
  contradictions_found/resolved
  diff_json → MemoryDiff[]

Session
  playbook_id, task_completed
  steps_taken, tokens_used (learning curve metrics)
```

## Key Design Decisions

**Why two Qwen3 instances?**
The active agent uses `thinking=False` for low latency during conversation. The consolidator uses `thinking=True` because contradiction detection requires deep reasoning over a large memory corpus — the extra compute is spent offline, not in the critical path.

**Why multi-signal retrieval?**
Semantic similarity alone fails on exact-term queries ("Stripe API key") and doesn't account for memory importance or recency. Fusing 5 signals with greedy budget-packing ensures the most decision-relevant memories fit within the context limit.

**Why budget-pack instead of top-K?**
Top-K can exceed the context window with long memories. Budget-packing guarantees the context limit is never breached while maximizing information density.

**Why deprecate instead of delete?**
The audit trail (deprecated_reason, superseded_by, deprecated_at) is essential for debugging and for showing judges the agent is making principled decisions, not randomly forgetting things.

**Why session-level metrics?**
The learning curve (tokens and steps per session) is the single most compelling demo artifact — it visually proves the agent gets faster and more efficient as the playbook matures.

## Evaluation Results

Run `python -m evals.harness --verbose` from the `backend/` directory.

| Suite | Cases | Pass Rate | Key Metric |
|-------|-------|-----------|------------|
| Contradiction Detection | 8 | 100% | Precision=1.00, Recall=1.00, F1=1.00 |
| Retrieval | 5 | 100% | MRR=1.000 |
| Decay | 3 | 100% | All decay thresholds correct |
| **Overall** | **16** | **100%** | |
