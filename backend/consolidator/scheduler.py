"""
engram/backend/consolidator/scheduler.py

Manages when the Forgetting Engine runs:
  - On session end (immediate, lightweight)
  - On a cron schedule (full deep scan, every 30 min by default)
  - On manual trigger via API
"""

from __future__ import annotations

import os

import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from consolidator.engine import ForgettingEngine
from memory.store import MemoryStore

log = structlog.get_logger(__name__)


class ConsolidationScheduler:
    def __init__(self, store: MemoryStore):
        self.store = store
        self.engine = ForgettingEngine(store)
        self._scheduler = AsyncIOScheduler()
        self._running = False

    def start(self) -> None:
        """Start the background scheduler."""
        cron_expr = os.getenv("CONSOLIDATION_CRON", "*/30 * * * *")

        # Parse cron into APScheduler CronTrigger
        parts = cron_expr.strip().split()
        if len(parts) == 5:
            minute, hour, day, month, day_of_week = parts
            trigger = CronTrigger(
                minute=minute,
                hour=hour,
                day=day,
                month=month,
                day_of_week=day_of_week,
            )
        else:
            # Fallback: every 30 minutes
            trigger = IntervalTrigger(minutes=30)

        self._scheduler.add_job(
            self._scheduled_run,
            trigger=trigger,
            id="consolidation_scheduled",
            replace_existing=True,
        )
        self._scheduler.start()
        self._running = True
        log.info("Consolidation scheduler started", cron=cron_expr)

    def stop(self) -> None:
        if self._running:
            self._scheduler.shutdown(wait=False)
            self._running = False
            log.info("Consolidation scheduler stopped")

    async def _scheduled_run(self) -> None:
        log.info("Scheduled consolidation triggered")
        await self.engine.run(triggered_by="scheduler")

    async def trigger_on_session_end(self, session_id: str) -> None:
        """
        Lightweight consolidation triggered immediately after a session ends.
        Activates pending memories from the session; full scan deferred to cron.
        """
        log.info("Session-end consolidation triggered", session_id=session_id[:8])
        await self.store.activate_pending_memories(session_id)
        # Run a targeted contradiction check for the new session's memories only
        # (full scan runs on cron schedule to avoid latency impact)

    async def trigger_manual(self) -> dict:
        """Manually trigger a full consolidation run (via API endpoint)."""
        log.info("Manual consolidation triggered")
        run = await self.engine.run(triggered_by="manual")
        return {
            "run_id": run.id,
            "completed_at": run.completed_at.isoformat() if run.completed_at else None,
            "memories_activated": run.memories_activated,
            "memories_deprecated": run.memories_deprecated,
            "memories_archived": run.memories_archived,
            "contradictions_resolved": run.contradictions_resolved,
            "error": run.error,
        }
