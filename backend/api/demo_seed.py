"""
engram/backend/api/demo_seed.py

Seeds the database with a realistic multi-session workflow demo.
Creates 3 sessions of increasing efficiency + a deliberate contradiction
so the memory graph, learning curve, and forgetting engine all have
interesting data to display from the first launch.

Called via POST /demo/seed (protected: only runs if DB is empty or force=True)
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

import structlog
from sqlalchemy import update, select

from memory.schema import Memory, MemoryType, MemoryStatus, new_id, utcnow

if TYPE_CHECKING:
    from memory.store import MemoryStore

log = structlog.get_logger(__name__)


async def seed_demo(store: "MemoryStore", force: bool = False) -> dict:
    """
    Populate the store with a realistic 3-session workflow demo.

    Workflow: "weekly_revenue_report"
    - Pull Stripe revenue for last 7 days
    - Compare to $50k monthly target (prorated)
    - Flag anomalies if delta > 10%
    - Draft Slack summary

    Session 1: Manual (no playbook) — agent learns the workflow
    Session 2: Uses playbook — faster, fewer steps
    Session 3: Playbook step fails (API endpoint changed) — contradiction introduced

    After seeding, run a consolidation to activate and process everything.
    """

    # Check if already seeded
    existing = await store.get_all_active_memories()
    if existing and not force:
        return {"ok": False, "reason": "Already seeded. Pass force=true to re-seed.", "existing_memories": len(existing)}

    now = datetime.now(timezone.utc)

    # ── SESSION 1 — 3 days ago, manual run ───────────────────────────────

    s1_time = now - timedelta(days=3, hours=2)
    s1_id = new_id()

    s1_memories = [
        {
            "content": "User's name is Buchi, fullstack developer building Signalpad",
            "type": MemoryType.SEMANTIC,
            "importance": 0.75,
            "tags": ["user", "name", "context"],
            "age_offset": timedelta(days=3, hours=2),
        },
        {
            "content": "Monthly revenue target is $50,000",
            "type": MemoryType.SEMANTIC,
            "importance": 0.90,
            "tags": ["revenue", "target", "monthly", "50000"],
            "age_offset": timedelta(days=3, hours=1, minutes=55),
        },
        {
            "content": "Stripe API key environment variable: STRIPE_SECRET_KEY (live key in .env)",
            "type": MemoryType.SEMANTIC,
            "importance": 0.85,
            "tags": ["stripe", "api", "key", "environment"],
            "age_offset": timedelta(days=3, hours=1, minutes=50),
        },
        {
            "content": "Revenue report uses Stripe /v1/charges endpoint to fetch last 7 days of charges",
            "type": MemoryType.PROCEDURAL,
            "importance": 0.80,
            "tags": ["stripe", "charges", "endpoint", "revenue", "report"],
            "age_offset": timedelta(days=3, hours=1, minutes=45),
        },
        {
            "content": "Anomaly threshold is 10% variance from prorated daily target ($50k / 30 = $1,667/day)",
            "type": MemoryType.SEMANTIC,
            "importance": 0.78,
            "tags": ["anomaly", "threshold", "variance", "daily"],
            "age_offset": timedelta(days=3, hours=1, minutes=40),
        },
        {
            "content": "Slack webhook for #revenue-alerts: https://hooks.slack.com/services/T0xxx/B0xxx/xxxxx",
            "type": MemoryType.SEMANTIC,
            "importance": 0.88,
            "tags": ["slack", "webhook", "revenue", "alerts"],
            "age_offset": timedelta(days=3, hours=1, minutes=35),
        },
        {
            "content": "Session 1 observation: report took 8 manual steps, 3200 tokens. No playbook existed.",
            "type": MemoryType.EPISODIC,
            "importance": 0.50,
            "tags": ["session", "observation", "manual"],
            "age_offset": timedelta(days=3, hours=1, minutes=5),
        },
    ]

    for m_data in s1_memories:
        m = await store.write_memory(
            session_id=s1_id,
            content=m_data["content"],
            memory_type=m_data["type"],
            importance_score=m_data["importance"],
            tags=m_data["tags"],
        )
        # Backdate
        backdated = now - m_data["age_offset"]
        async with store._session() as db:
            await db.execute(update(Memory).where(Memory.id == m.id).values(
                created_at=backdated, status=MemoryStatus.ACTIVE
            ))
            await db.commit()

    # Write Session 1 playbook (distilled from the manual run)
    pb_v1 = await store.write_playbook(
        name="weekly_revenue_report",
        description="Pull weekly Stripe revenue, compare to $50k target, flag anomalies, post to Slack",
        session_id=s1_id,
        steps=[
            {
                "title": "Fetch Stripe charges",
                "description": "Call Stripe /v1/charges?created[gte]={7_days_ago}&limit=100 with STRIPE_SECRET_KEY",
                "tool_used": "stripe_api",
                "expected_output": "JSON array of charge objects with amount, currency, created",
                "decision_point": False,
                "edge_cases": ["API rate limit: retry with exponential backoff", "Empty response: check date range"],
            },
            {
                "title": "Sum revenue",
                "description": "Sum all charge amounts in cents, convert to dollars, filter status=succeeded",
                "tool_used": "code_interpreter",
                "expected_output": "Float: total revenue in USD for the 7-day window",
                "decision_point": False,
                "edge_cases": ["Multi-currency: convert to USD using latest FX rate"],
            },
            {
                "title": "Compare to prorated target",
                "description": "Target = $50000/30*7 = $11,667. Calculate variance: (actual - target) / target * 100",
                "tool_used": "code_interpreter",
                "expected_output": "Float: variance percentage. Positive = above target.",
                "decision_point": True,
                "edge_cases": ["Weekend effect: Friday-Sunday typically lower by 15-20%"],
            },
            {
                "title": "Flag anomaly if needed",
                "description": "If abs(variance) > 10%, write_memory with importance 0.9 flagging the anomaly",
                "tool_used": "write_memory",
                "expected_output": "Memory written with anomaly details",
                "decision_point": True,
                "edge_cases": ["False positive on holidays"],
            },
            {
                "title": "Draft Slack message",
                "description": "Draft a concise summary: Revenue this week: $X (Y% vs target). [Anomaly: Z] if flagged.",
                "tool_used": "slack_webhook",
                "expected_output": "200 OK from Slack webhook",
                "decision_point": False,
                "edge_cases": ["Webhook timeout: retry once"],
            },
        ],
    )

    # Backdate the playbook memory
    async with store._session() as db:
        await db.execute(update(Memory).where(Memory.session_id == s1_id, Memory.memory_type == MemoryType.PROCEDURAL).values(
            created_at=now - timedelta(days=3, hours=1),
        ))
        await db.commit()

    # Record Session 1 metrics
    sess1 = await store.create_session(pb_v1.id)
    await store.end_session(sess1.id, True, 8, 3200, [], playbook_used=False)

    log.info("Session 1 seeded", memories=len(s1_memories))

    # ── SESSION 2 — 2 days ago, used playbook ────────────────────────────

    s2_id = new_id()

    s2_memories = [
        {
            "content": "Revenue report ran successfully using playbook v1. Actual: $9,840 vs target $11,667. Variance: -15.6% — within weekend effect range, no anomaly flagged.",
            "type": MemoryType.EPISODIC,
            "importance": 0.65,
            "tags": ["revenue", "session2", "result"],
            "age_offset": timedelta(days=2, hours=1),
        },
        {
            "content": "Stripe /v1/charges step succeeded in 1.2s. API response consistent.",
            "type": MemoryType.EPISODIC,
            "importance": 0.45,
            "tags": ["stripe", "performance", "session2"],
            "age_offset": timedelta(days=2, hours=0, minutes=55),
        },
    ]

    for m_data in s2_memories:
        m = await store.write_memory(
            session_id=s2_id,
            content=m_data["content"],
            memory_type=m_data["type"],
            importance_score=m_data["importance"],
            tags=m_data["tags"],
        )
        backdated = now - m_data["age_offset"]
        async with store._session() as db:
            await db.execute(update(Memory).where(Memory.id == m.id).values(
                created_at=backdated, status=MemoryStatus.ACTIVE
            ))
            await db.commit()

    sess2 = await store.create_session(pb_v1.id)
    await store.end_session(sess2.id, True, 3, 820, [], playbook_used=True)

    log.info("Session 2 seeded")

    # ── SESSION 3 — today, API changed, contradiction introduced ─────────

    s3_id = new_id()

    # The contradicting memory — Stripe deprecated /v1/charges
    m_contradiction = await store.write_memory(
        session_id=s3_id,
        content="CRITICAL: Stripe deprecated /v1/charges endpoint. All calls must now use /v1/payment_intents. The /charges endpoint returns 410 Gone as of June 2026.",
        memory_type=MemoryType.SEMANTIC,
        importance_score=0.98,
        tags=["stripe", "deprecated", "charges", "payment_intents", "critical"],
    )

    s3_memories = [
        {
            "content": "Session 3: Step 1 (Fetch Stripe charges) FAILED — 410 Gone from /v1/charges endpoint",
            "type": MemoryType.EPISODIC,
            "importance": 0.80,
            "tags": ["stripe", "failure", "session3", "410"],
            "age_offset": timedelta(hours=1, minutes=30),
        },
        {
            "content": "Recovered: switched to /v1/payment_intents endpoint. Revenue fetch succeeded on retry.",
            "type": MemoryType.EPISODIC,
            "importance": 0.75,
            "tags": ["stripe", "recovery", "payment_intents", "session3"],
            "age_offset": timedelta(hours=1, minutes=20),
        },
        {
            "content": "Revenue this week: $12,340 (+5.8% vs target $11,667). No anomaly. Posted to #revenue-alerts.",
            "type": MemoryType.EPISODIC,
            "importance": 0.60,
            "tags": ["revenue", "result", "session3", "slack"],
            "age_offset": timedelta(hours=1, minutes=10),
        },
    ]

    for m_data in s3_memories:
        m = await store.write_memory(
            session_id=s3_id,
            content=m_data["content"],
            memory_type=m_data["type"],
            importance_score=m_data["importance"],
            tags=m_data["tags"],
        )
        backdated = now - m_data["age_offset"]
        async with store._session() as db:
            await db.execute(update(Memory).where(Memory.id == m.id).values(
                created_at=backdated, status=MemoryStatus.ACTIVE
            ))
            await db.commit()

    # Activate contradiction memory
    async with store._session() as db:
        await db.execute(update(Memory).where(Memory.id == m_contradiction.id).values(
            created_at=now - timedelta(hours=2),
            status=MemoryStatus.ACTIVE
        ))
        await db.commit()

    sess3 = await store.create_session(pb_v1.id)
    await store.end_session(sess3.id, True, 4, 1100, [], playbook_used=True)

    log.info("Session 3 seeded with contradiction")

    # ── Run consolidation to process everything ───────────────────────────

    from consolidator.engine import ForgettingEngine
    engine = ForgettingEngine(store)
    run = await engine.run(triggered_by="demo_seed")

    # The contradiction LLM call won't work without a key,
    # but we can manually apply the known contradiction for demo purposes
    # Find the old /charges memory and deprecate it
    async with store._session() as db:
        result = await db.execute(
            select(Memory).where(
                Memory.content.like("%/v1/charges%"),
                Memory.status == MemoryStatus.ACTIVE,
            )
        )
        old_charges_mem = result.scalars().first()

    if old_charges_mem:
        await store.deprecate_memory(
            memory_id=old_charges_mem.id,
            reason="CONTRADICTION: Stripe deprecated /v1/charges endpoint June 2026. Superseded by payment_intents.",
            superseded_by_id=m_contradiction.id,
        )
        log.info("Demo contradiction applied", deprecated=old_charges_mem.id[:8])

    all_mems = await store.get_all_active_memories()
    sessions = await store.get_session_metrics()

    return {
        "ok": True,
        "seeded": {
            "sessions": len(sessions),
            "memories": len(all_mems),
            "contradiction_applied": old_charges_mem is not None,
            "consolidation_run_id": run.id,
        },
    }
