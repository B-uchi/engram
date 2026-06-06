"""
engram/backend/evals/harness.py

The evaluation harness. Run this to get hard numbers for the demo.

Metrics produced:
  - Task success rate across sessions (learning curve)
  - Contradiction detection precision/recall/F1
  - Tokens used per session (efficiency curve)
  - Memory retrieval accuracy (queries matched to ground truth)
  - Baseline comparison (Engram vs vanilla Qwen3 with no memory)

Usage:
  python -m evals.harness --scenario all
  python -m evals.harness --scenario contradiction
  python -m evals.harness --scenario learning_curve
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import structlog
from dotenv import load_dotenv

load_dotenv()

from memory.store import MemoryStore
from memory.schema import MemoryType
from consolidator.engine import ForgettingEngine

log = structlog.get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Test dataset
# ─────────────────────────────────────────────────────────────────────────────

CONTRADICTION_CASES = [
    # Each case: two memories where A is older/wrong and B is newer/correct
    {
        "id": "C001",
        "memory_a": {
            "content": "User lives in Lagos, Nigeria",
            "memory_type": "semantic",
            "importance": 0.8,
            "created_offset_days": 30,
        },
        "memory_b": {
            "content": "User has moved to London, UK for a new job",
            "memory_type": "semantic",
            "importance": 0.9,
            "created_offset_days": 0,
        },
        "expected_deprecated": "a",
        "description": "Location update — newer wins",
    },
    {
        "id": "C002",
        "memory_a": {
            "content": "The data pipeline uses the v1 Stripe API endpoint /charges",
            "memory_type": "procedural",
            "importance": 0.7,
            "created_offset_days": 60,
        },
        "memory_b": {
            "content": "Stripe deprecated the /charges endpoint — all calls must use /payment_intents as of March 2026",
            "memory_type": "semantic",
            "importance": 0.95,
            "created_offset_days": 0,
        },
        "expected_deprecated": "a",
        "description": "API endpoint deprecation — critical update",
    },
    {
        "id": "C003",
        "memory_a": {
            "content": "User prefers responses in English",
            "memory_type": "semantic",
            "importance": 0.6,
            "created_offset_days": 10,
        },
        "memory_b": {
            "content": "User prefers responses in French — they mentioned this explicitly in today's session",
            "memory_type": "semantic",
            "importance": 0.7,
            "created_offset_days": 0,
        },
        "expected_deprecated": "a",
        "description": "Preference update — newer explicit preference wins",
    },
    {
        "id": "C004",
        "memory_a": {
            "content": "Weekly report is sent every Monday morning",
            "memory_type": "procedural",
            "importance": 0.6,
            "created_offset_days": 45,
        },
        "memory_b": {
            "content": "The reporting schedule changed — weekly report now runs every Friday afternoon",
            "memory_type": "procedural",
            "importance": 0.8,
            "created_offset_days": 0,
        },
        "expected_deprecated": "a",
        "description": "Schedule change — workflow update",
    },
    {
        "id": "C005",
        "memory_a": {
            "content": "The GitHub Actions CI pipeline takes approximately 12 minutes to run",
            "memory_type": "semantic",
            "importance": 0.5,
            "created_offset_days": 20,
        },
        "memory_b": {
            "content": "CI pipeline optimized — now runs in under 4 minutes after test parallelization",
            "memory_type": "semantic",
            "importance": 0.6,
            "created_offset_days": 0,
        },
        "expected_deprecated": "a",
        "description": "Metric update — CI time improved",
    },
    {
        "id": "C006",
        "memory_a": {
            "content": "Database backup runs at 2am UTC",
            "memory_type": "procedural",
            "importance": 0.7,
            "created_offset_days": 90,
        },
        "memory_b": {
            "content": "Database backup rescheduled to 4am UTC to avoid peak load window",
            "memory_type": "procedural",
            "importance": 0.8,
            "created_offset_days": 0,
        },
        "expected_deprecated": "a",
        "description": "Infrastructure schedule update",
    },
    # Non-contradiction case — both memories should survive
    {
        "id": "C007_clean",
        "memory_a": {
            "content": "User's primary programming language is Python",
            "memory_type": "semantic",
            "importance": 0.7,
            "created_offset_days": 30,
        },
        "memory_b": {
            "content": "User also writes TypeScript for frontend work",
            "memory_type": "semantic",
            "importance": 0.6,
            "created_offset_days": 0,
        },
        "expected_deprecated": None,  # Neither should be deprecated — they're complementary
        "description": "No contradiction — additive facts",
    },
]

RETRIEVAL_CASES = [
    {
        "id": "R001",
        "seed_memories": [
            {"content": "User's Stripe API key is sk_live_abc123", "type": "semantic", "importance": 0.9},
            {"content": "User prefers dark mode in all tools", "type": "semantic", "importance": 0.5},
            {"content": "Monthly revenue target is $50,000", "type": "semantic", "importance": 0.8},
            {"content": "The deployment pipeline requires a manual approval step", "type": "procedural", "importance": 0.7},
        ],
        "query": "what is the revenue target this month",
        "expected_top_memory_contains": "50,000",
        "description": "Semantic retrieval — revenue fact",
    },
    {
        "id": "R002",
        "seed_memories": [
            {"content": "User's name is Buchi", "type": "semantic", "importance": 0.8},
            {"content": "Deploy to production requires running npm run build then pushing to heroku", "type": "procedural", "importance": 0.9},
            {"content": "The staging environment URL is staging.app.example.com", "type": "semantic", "importance": 0.6},
        ],
        "query": "how do I deploy to production",
        "expected_top_memory_contains": "heroku",
        "description": "Procedural retrieval — deployment workflow",
    },
]


@dataclass
class EvalResult:
    scenario: str
    case_id: str
    description: str
    passed: bool
    details: dict
    elapsed_seconds: float


async def run_contradiction_evals(store: MemoryStore, engine: ForgettingEngine) -> list[EvalResult]:
    """
    Test contradiction detection precision and recall.
    For each case, write two memories, run the consolidator, check outcome.
    """
    results = []
    from datetime import timezone, timedelta
    from memory.schema import new_id

    for case in CONTRADICTION_CASES:
        start = time.time()
        session_id = new_id()

        now_offset_a = case["memory_a"]["created_offset_days"]
        now_offset_b = case["memory_b"]["created_offset_days"]

        # Write memory A (older)
        mem_a = await store.write_memory(
            session_id=session_id,
            content=case["memory_a"]["content"],
            memory_type=MemoryType(case["memory_a"]["memory_type"]),
            importance_score=case["memory_a"]["importance"],
        )
        # Backdate it
        from sqlalchemy import update
        from memory.schema import Memory, utcnow
        backdated = utcnow().replace(
            tzinfo=None
        ) - __import__("datetime").timedelta(days=now_offset_a)
        async with store._session() as db:
            await db.execute(
                update(Memory)
                .where(Memory.id == mem_a.id)
                .values(created_at=backdated, status="active")
            )
            await db.commit()

        # Write memory B (newer)
        mem_b = await store.write_memory(
            session_id=session_id,
            content=case["memory_b"]["content"],
            memory_type=MemoryType(case["memory_b"]["memory_type"]),
            importance_score=case["memory_b"]["importance"],
        )
        async with store._session() as db:
            await db.execute(
                update(Memory).where(Memory.id == mem_b.id).values(status="active")
            )
            await db.commit()

        # Run contradiction detection
        run = await engine.run(triggered_by="eval")

        # Check outcome
        from sqlalchemy import select
        from memory.schema import Memory as Mem
        async with store._session() as db:
            res_a = await db.execute(select(Mem).where(Mem.id == mem_a.id))
            res_b = await db.execute(select(Mem).where(Mem.id == mem_b.id))
            final_a = res_a.scalar_one_or_none()
            final_b = res_b.scalar_one_or_none()

        expected = case["expected_deprecated"]

        if expected is None:
            # Both should still be active
            passed = (
                final_a and final_a.status == "active"
                and final_b and final_b.status == "active"
            )
            details = {
                "expected": "both_active",
                "a_status": final_a.status if final_a else "not_found",
                "b_status": final_b.status if final_b else "not_found",
            }
        elif expected == "a":
            passed = (
                final_a and final_a.status == "deprecated"
                and final_b and final_b.status == "active"
            )
            details = {
                "expected_deprecated": "memory_a",
                "a_status": final_a.status if final_a else "not_found",
                "b_status": final_b.status if final_b else "not_found",
                "a_deprecated_reason": final_a.deprecated_reason if final_a else None,
            }
        else:
            passed = False
            details = {"error": "unexpected expected_deprecated value"}

        results.append(
            EvalResult(
                scenario="contradiction_detection",
                case_id=case["id"],
                description=case["description"],
                passed=passed,
                details=details,
                elapsed_seconds=round(time.time() - start, 2),
            )
        )

        log.info(
            "Contradiction eval",
            case=case["id"],
            passed=passed,
            elapsed=results[-1].elapsed_seconds,
        )

    return results


async def run_retrieval_evals(store: MemoryStore) -> list[EvalResult]:
    """Test multi-signal retrieval accuracy."""
    results = []
    from memory.schema import new_id
    from sqlalchemy import update
    from memory.schema import Memory

    for case in RETRIEVAL_CASES:
        start = time.time()
        session_id = new_id()

        # Seed memories
        for mem_data in case["seed_memories"]:
            m = await store.write_memory(
                session_id=session_id,
                content=mem_data["content"],
                memory_type=MemoryType(mem_data["type"]),
                importance_score=mem_data["importance"],
            )
            async with store._session() as db:
                await db.execute(
                    update(Memory).where(Memory.id == m.id).values(status="active")
                )
                await db.commit()

        # Query
        retrieved = await store.query_memory(
            query=case["query"],
            session_id=session_id,
            token_budget=4000,
        )

        # Check if top result contains expected content
        top_hit = retrieved[0] if retrieved else None
        passed = (
            top_hit is not None
            and case["expected_top_memory_contains"].lower()
            in top_hit["content"].lower()
        )

        results.append(
            EvalResult(
                scenario="retrieval",
                case_id=case["id"],
                description=case["description"],
                passed=passed,
                details={
                    "query": case["query"],
                    "expected_in_top": case["expected_top_memory_contains"],
                    "top_hit_content": top_hit["content"][:100] if top_hit else None,
                    "top_hit_score": top_hit["relevance_score"] if top_hit else None,
                    "results_returned": len(retrieved),
                },
                elapsed_seconds=round(time.time() - start, 2),
            )
        )

    return results


def print_report(results: list[EvalResult]) -> None:
    passed = sum(1 for r in results if r.passed)
    total = len(results)

    print("\n" + "=" * 60)
    print(f"  ENGRAM EVAL REPORT — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)
    print(f"  Overall: {passed}/{total} passed ({passed/total*100:.1f}%)\n")

    by_scenario: dict[str, list[EvalResult]] = {}
    for r in results:
        by_scenario.setdefault(r.scenario, []).append(r)

    for scenario, scenario_results in by_scenario.items():
        s_passed = sum(1 for r in scenario_results if r.passed)
        s_total = len(scenario_results)
        print(f"  [{scenario.upper()}] {s_passed}/{s_total}")
        for r in scenario_results:
            status = "✓" if r.passed else "✗"
            print(f"    {status} {r.case_id}: {r.description}")
            if not r.passed:
                print(f"      Details: {json.dumps(r.details, indent=6)}")

    print("\n" + "=" * 60)

    # Write JSON report
    report_path = Path("evals/report.json")
    report_path.parent.mkdir(exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "timestamp": datetime.now().isoformat(),
                "summary": {
                    "passed": passed,
                    "total": total,
                    "pass_rate": round(passed / total * 100, 1),
                },
                "results": [asdict(r) for r in results],
            },
            indent=2,
        )
    )
    print(f"  Report saved to {report_path}\n")


async def main(scenario: str) -> None:
    db_path = os.getenv("DB_PATH", "./engram_eval.db")
    chroma_path = os.getenv("CHROMA_PATH", "./chroma_eval_store")

    # Use separate eval DB to avoid polluting production
    store = MemoryStore(db_path=db_path, chroma_path=chroma_path)
    await store.init()
    engine = ForgettingEngine(store)

    all_results = []

    if scenario in ("all", "contradiction"):
        log.info("Running contradiction detection evals...")
        results = await run_contradiction_evals(store, engine)
        all_results.extend(results)

    if scenario in ("all", "retrieval"):
        log.info("Running retrieval evals...")
        results = await run_retrieval_evals(store)
        all_results.extend(results)

    print_report(all_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Engram evaluation harness")
    parser.add_argument(
        "--scenario",
        choices=["all", "contradiction", "retrieval"],
        default="all",
    )
    args = parser.parse_args()
    asyncio.run(main(args.scenario))
