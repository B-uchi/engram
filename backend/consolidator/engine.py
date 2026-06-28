"""
engram/backend/consolidator/engine.py

The Forgetting Engine — Engram's background consolidator.

Runs asynchronously between sessions (on session end + scheduled cron).
Uses Qwen3 in THINKING MODE for deep reasoning over memory contradictions.

What it does per run:
  1. Scans all PENDING memories → activates them (no conflicts found)
  2. Scans all ACTIVE memories for decay → archives low-value ones
  3. Runs contradiction detection across the full memory corpus
  4. For each contradiction: deprecates the loser, records why
  5. Scans playbook steps for failure patterns → flags stale steps
  6. Writes a ConsolidationRun with a full diff for the UI

This is the "1A" component of Engram — the part that makes memory
stay correct over time rather than accumulating stale facts.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from typing import Optional

import dashscope
import structlog
from qwen_agent.llm import get_chat_model

from memory.store import MemoryStore
from memory.schema import (
    ConsolidationRun,
    Memory,
    MemoryDiff,
    MemoryStatus,
    MemoryType,
    new_id,
    utcnow,
)

log = structlog.get_logger(__name__)

CONTRADICTION_DETECTION_PROMPT = """You are analyzing a set of memories for contradictions, staleness, and redundancy.

## Memory Corpus
{memories_json}

## Your Task
Analyze these memories carefully. Identify:

1. **CONTRADICTIONS**: Two memories that assert conflicting facts
   - Example: Memory A says "user lives in Lagos", Memory B says "user moved to London"
   - Rule: The NEWER memory wins, unless there's evidence the older one is still valid
   - Output: {{"type": "contradiction", "deprecated_id": "...", "winner_id": "...", "reason": "..."}}

2. **REDUNDANCIES**: Multiple memories that say essentially the same thing
   - Keep the most complete/recent one, deprecate the others
   - Output: {{"type": "redundancy", "deprecated_ids": ["..."], "kept_id": "...", "reason": "..."}}

3. **MERGES**: Fragments that should be consolidated into one summary
   - Output: {{"type": "merge", "source_ids": ["..."], "merged_content": "...", "reason": "..."}}

4. **STALE_PATTERN**: A playbook step that has failed repeatedly and should be flagged
   - Output: {{"type": "stale_step", "step_id": "...", "reason": "..."}}

5. **CLEAN**: No issues found
   - Output: {{"type": "clean"}}

Respond with ONLY a JSON array of findings. No preamble, no explanation outside the JSON.
Example:
[
  {{"type": "contradiction", "deprecated_id": "abc123", "winner_id": "def456", "reason": "Memory def456 is newer and explicitly states the user moved"}},
  {{"type": "clean"}}
]
"""

PLAYBOOK_REVIEW_PROMPT = """You are reviewing a playbook for staleness and correctness.

## Playbook: {playbook_name}
{playbook_json}

## Recent Session Failures
{failures_json}

## Your Task
Review this playbook against the recent failures. Identify:

1. Steps that consistently fail and need updating
2. Steps that are outdated based on recent session observations
3. Missing steps that should be added based on what sessions actually do
4. Steps that can be merged or simplified

For each issue found, output:
{{"step_id": "...", "action": "update|deprecate|add", "new_content": "...", "reason": "..."}}

If the playbook is correct, output: {{"action": "none", "reason": "Playbook is accurate"}}

Respond with ONLY a JSON array. No preamble.
"""


class ForgettingEngine:
    def __init__(self, store: MemoryStore):
        self.store = store
        # DashScope keys are region-bound; our account is International (Singapore).
        # The SDK defaults to the China endpoint, which rejects intl keys.
        dashscope.base_http_api_url = os.getenv(
            "DASHSCOPE_BASE_URL", "https://dashscope-intl.aliyuncs.com/api/v1"
        )
        self._llm_cfg = {
            "model": os.getenv("CONSOLIDATOR_MODEL", "qwen3-32b"),
            "model_type": "qwen_dashscope",
            "generate_cfg": {
                "enable_thinking": True,   # Thinking mode for deep reasoning
                "top_p": 0.9,
                "max_input_tokens": 58000,  # Consolidator gets more context
                # Match Qwen-Agent's cumulative-streaming assumption — the intl
                # endpoint otherwise defaults to delta streaming. See active.py.
                "incremental_output": False,
            },
        }
        self._llm = get_chat_model(self._llm_cfg)

        self._decay_archive_threshold = float(
            os.getenv("DECAY_ARCHIVE_THRESHOLD", "0.15")
        )
        self._decay_deprecate_threshold = float(
            os.getenv("DECAY_DEPRECATE_THRESHOLD", "0.06")
        )

    async def run(self, triggered_by: str = "scheduler") -> ConsolidationRun:
        """
        Execute a full consolidation cycle.
        Returns the ConsolidationRun record with the complete diff.
        """
        run = ConsolidationRun(
            id=new_id(),
            started_at=utcnow(),
            triggered_by=triggered_by,
        )
        diffs: list[dict] = []

        log.info("Consolidation started", triggered_by=triggered_by)
        start_time = time.time()

        try:
            # Phase 1: Activate all pending memories
            activated = await self._activate_pending(run, diffs)
            run.memories_activated = activated

            # Phase 2: Decay scoring — archive low-value memories
            archived = await self._run_decay(run, diffs)
            run.memories_archived = archived

            # Phase 3: Contradiction detection via Qwen3 thinking mode
            contradictions_found, contradictions_resolved = await self._detect_contradictions(
                run, diffs
            )
            run.contradictions_found = contradictions_found
            run.contradictions_resolved = contradictions_resolved

            # Phase 4: Playbook review — stale step detection
            playbooks_updated = await self._review_playbooks(run, diffs)
            run.playbooks_updated = playbooks_updated

            run.memories_deprecated = sum(
                1 for d in diffs if d.get("action") == "deprecated"
            ) + sum(
                len(d.get("source_ids", [])) for d in diffs if d.get("action") == "merged"
            )

            run.completed_at = utcnow()
            run.diff_json = json.dumps(diffs)
            run.memories_scanned = len(await self.store.get_all_active_memories())
            self._attach_diff_rows(run, diffs)

        except Exception as e:
            run.error = str(e)
            run.completed_at = utcnow()
            log.error("Consolidation failed", error=str(e))

        await self.store.save_consolidation_run(run)

        elapsed = time.time() - start_time
        log.info(
            "Consolidation complete",
            elapsed=round(elapsed, 2),
            activated=run.memories_activated,
            deprecated=run.memories_deprecated,
            archived=run.memories_archived,
            contradictions=run.contradictions_resolved,
        )
        return run

    def _attach_diff_rows(self, run: ConsolidationRun, diffs: list) -> None:
        """Persist each diff dict as a MemoryDiff row (cascade-saved with the run)."""
        for d in diffs:
            memory_id = (
                d.get("memory_id")
                or d.get("new_memory_id")
                or d.get("kept_id")
                or (d.get("source_ids") or [None])[0]
            )
            if not memory_id:
                continue
            run.diffs.append(
                MemoryDiff(
                    id=new_id(),
                    memory_id=memory_id,
                    action=d.get("action", "updated"),
                    reason=d.get("reason", ""),
                    before_content=d.get("before_content"),
                    after_content=d.get("after_content")
                    or d.get("merged_content")
                    or d.get("content_preview"),
                )
            )

    async def _activate_pending(
        self, run: ConsolidationRun, diffs: list
    ) -> int:
        """Move PENDING memories to ACTIVE after basic validation."""
        all_memories = await self.store.get_all_active_memories()
        pending = [m for m in all_memories if m.status == MemoryStatus.PENDING]
        count = 0

        for memory in pending:
            # Basic validation: non-empty content
            if memory.content and len(memory.content.strip()) > 5:
                memory.activate()
                diffs.append({
                    "action": "activated",
                    "memory_id": memory.id,
                    "memory_type": memory.memory_type,
                    "reason": "Passed validation, moved from PENDING to ACTIVE",
                    "content_preview": memory.content[:100],
                })
                count += 1
            else:
                memory.archive()
                diffs.append({
                    "action": "archived",
                    "memory_id": memory.id,
                    "reason": "Empty or trivial content",
                })

        # Persist status changes
        from sqlalchemy import select
        from sqlalchemy.ext.asyncio import AsyncSession

        async with self.store._session() as db:
            for memory in pending:
                await db.merge(memory)
            await db.commit()

        return count

    async def _run_decay(self, run: ConsolidationRun, diffs: list) -> int:
        """
        Score all memories for decay and archive low-value ones.
        Decay score = recency_weight * recency + frequency_weight * access_frequency
                     + importance_weight * importance_score
        """
        all_memories = await self.store.get_all_active_memories()
        active = [m for m in all_memories if m.status == MemoryStatus.ACTIVE]
        now = datetime.now(timezone.utc)
        archived_count = 0

        from sqlalchemy.ext.asyncio import AsyncSession

        async with self.store._session() as db:
            for memory in active:
                created_at = memory.created_at
                if created_at.tzinfo is None:
                    from datetime import timezone as _tz
                    created_at = created_at.replace(tzinfo=_tz.utc)
                age_days = (now - created_at).total_seconds() / 86400
                if memory.last_accessed_at:
                    laa = memory.last_accessed_at
                    if laa.tzinfo is None:
                        from datetime import timezone as _tz2
                        laa = laa.replace(tzinfo=_tz2.utc)
                    days_since_access = (now - laa).total_seconds() / 86400
                else:
                    days_since_access = age_days

                # Recency: exponential decay, half-life 30 days
                import math
                recency = math.exp(-0.023 * age_days)

                # Access frequency: normalized, capped at 1.0
                frequency = min(1.0, memory.access_count / 10.0)

                # Composite decay score
                decay_score = (
                    0.4 * recency
                    + 0.3 * frequency
                    + 0.3 * memory.importance_score
                )

                # Update recency score on the record
                memory.recency_score = recency

                if decay_score < self._decay_deprecate_threshold:
                    memory.archive()
                    archived_count += 1
                    diffs.append({
                        "action": "archived",
                        "memory_id": memory.id,
                        "memory_type": memory.memory_type,
                        "reason": f"Decay score {decay_score:.3f} below threshold {self._decay_deprecate_threshold}",
                        "decay_score": round(decay_score, 3),
                        "age_days": round(age_days, 1),
                        "access_count": memory.access_count,
                        "importance": memory.importance_score,
                    })
                await db.merge(memory)  # Persist status change AND recency_score update

            await db.commit()

        return archived_count

    async def _detect_contradictions(
        self, run: ConsolidationRun, diffs: list
    ) -> tuple[int, int]:
        """
        Use Qwen3 thinking mode to detect contradictions across the memory corpus.
        Batches memories to stay within the LLM context limit.
        """
        all_memories = await self.store.get_all_active_memories()
        # Facts only — procedural memories are handled by playbook versioning/review.
        active = [
            m for m in all_memories
            if m.status == MemoryStatus.ACTIVE and m.memory_type != MemoryType.PROCEDURAL
        ]

        if len(active) < 2:
            return 0, 0

        # Build a compact representation for the LLM
        memories_payload = [
            {
                "id": m.id,
                "content": m.content,
                "memory_type": m.memory_type,
                "created_at": m.created_at.isoformat(),
                "importance": m.importance_score,
            }
            for m in active
        ]

        # Batch into chunks of 50 memories to avoid context overflow
        batch_size = 50
        found = 0
        resolved = 0

        for i in range(0, len(memories_payload), batch_size):
            batch = memories_payload[i : i + batch_size]
            findings = await self._run_contradiction_llm(batch)
            f, r = await self._apply_findings(findings, diffs)
            found += f
            resolved += r

        return found, resolved

    async def _run_contradiction_llm(self, memories_payload: list[dict]) -> list[dict]:
        """Call Qwen3 thinking mode to find contradictions in a batch."""
        prompt = CONTRADICTION_DETECTION_PROMPT.format(
            memories_json=json.dumps(memories_payload, indent=2, ensure_ascii=False)
        )

        messages = [{"role": "user", "content": prompt}]
        response_text = ""

        try:
            for chunk in self._llm.chat(messages=messages, stream=True):
                for msg in chunk:
                    if isinstance(msg, dict) and msg.get("role") == "assistant":
                        content = msg.get("content", "")
                        if content:
                            response_text = content
        except Exception as e:
            log.error("Contradiction LLM call failed", error=str(e))
            return []

        # Parse JSON from response — strip any markdown fences
        clean = response_text.strip()
        if clean.startswith("```"):
            lines = clean.split("\n")
            clean = "\n".join(
                l for l in lines if not l.strip().startswith("```")
            )

        try:
            findings = json.loads(clean)
            if not isinstance(findings, list):
                findings = [findings]
            return findings
        except json.JSONDecodeError as e:
            log.warning("Failed to parse contradiction findings", error=str(e), raw=clean[:200])
            return []

    async def _apply_findings(
        self, findings: list[dict], diffs: list
    ) -> tuple[int, int]:
        """Apply contradiction findings to the memory store."""
        found = 0
        resolved = 0

        for finding in findings:
            ftype = finding.get("type", "clean")

            if ftype == "clean":
                continue

            elif ftype == "contradiction":
                found += 1
                deprecated_id = finding.get("deprecated_id")
                winner_id = finding.get("winner_id")
                reason = finding.get("reason", "Contradiction detected")

                if deprecated_id and winner_id:
                    try:
                        result = await self.store.deprecate_memory(
                            memory_id=deprecated_id,
                            reason=f"CONTRADICTION: {reason}",
                            superseded_by_id=winner_id,
                        )
                        if result is None:
                            log.warning("Contradiction target not found", id=deprecated_id)
                        else:
                            diffs.append({
                                "action": "deprecated",
                                "memory_id": deprecated_id,
                                "superseded_by": winner_id,
                                "reason": reason,
                            })
                            resolved += 1
                    except Exception as e:
                        log.error("Failed to deprecate", id=deprecated_id, error=str(e))

            elif ftype == "redundancy":
                found += 1
                deprecated_ids = finding.get("deprecated_ids", [])
                kept_id = finding.get("kept_id")
                reason = finding.get("reason", "Redundancy")

                for dep_id in deprecated_ids:
                    if dep_id and dep_id != kept_id:
                        try:
                            result = await self.store.deprecate_memory(
                                memory_id=dep_id,
                                reason=f"REDUNDANT: {reason}",
                                superseded_by_id=kept_id,
                            )
                            if result is None:
                                log.warning("Redundant target not found", id=dep_id)
                            else:
                                diffs.append({
                                    "action": "deprecated",
                                    "memory_id": dep_id,
                                    "reason": f"Merged into {kept_id}: {reason}",
                                })
                                resolved += 1
                        except Exception as e:
                            log.error("Failed to deprecate redundant", id=dep_id, error=str(e))

            elif ftype == "merge":
                found += 1
                source_ids = finding.get("source_ids", [])
                merged_content = finding.get("merged_content", "")
                reason = finding.get("reason", "Merge")

                if merged_content and source_ids:
                    # Write the merged memory
                    try:
                        if source_ids:
                            new_mem = await self.store.write_memory(
                                session_id="consolidator",
                                content=merged_content,
                                memory_type=MemoryType.SEMANTIC,
                                importance_score=0.7,
                                summary=merged_content[:200],
                            )
                            await self.store.activate_memory(new_mem.id)

                            for src_id in source_ids:
                                await self.store.deprecate_memory(
                                    memory_id=src_id,
                                    reason=f"MERGED: {reason}",
                                    superseded_by_id=new_mem.id,
                                )
                            diffs.append({
                                "action": "merged",
                                "source_ids": source_ids,
                                "new_memory_id": new_mem.id,
                                "merged_content": merged_content[:200],
                                "reason": reason,
                            })
                            resolved += 1
                    except Exception as e:
                        log.error("Merge failed", error=str(e))

        return found, resolved

    async def _review_playbooks(self, run: ConsolidationRun, diffs: list) -> int:
        """
        Review playbook steps for staleness based on failure patterns.
        Flags steps with high failure rates for the agent to re-examine.
        """
        playbooks = await self.store.list_playbooks()
        updated = 0

        for pb_summary in playbooks:
            playbook = await self.store.get_playbook(pb_summary["name"])
            if not playbook:
                continue

            # Flag failing steps, but only after >= 3 executions.
            stale_steps = [
                s for s in playbook["steps"]
                if s.get("attempts", 0) >= 3 and s["success_rate"] < 0.5
            ]

            if not stale_steps:
                continue

            # Flag them as needing review
            for step in stale_steps:
                diffs.append({
                    "action": "stale_step_flagged",
                    "playbook_name": playbook["name"],
                    "step_id": step["id"],
                    "step_title": step["title"],
                    "success_rate": step["success_rate"],
                    "reason": f"Step '{step['title']}' has success rate {step['success_rate']:.1%} — likely stale",
                })
                updated += 1

        return updated
