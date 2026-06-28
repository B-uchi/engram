"""
engram/backend/api/main.py

FastAPI application entry point.
All routes are properly typed and return structured responses.
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from typing import Optional

import structlog
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

from memory.store import MemoryStore
from memory.schema import MemoryType
from agents.active import ActiveAgent
from consolidator.scheduler import ConsolidationScheduler

log = structlog.get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Shared app state
# ─────────────────────────────────────────────────────────────────────────────

store: Optional[MemoryStore] = None
active_agent: Optional[ActiveAgent] = None
scheduler: Optional[ConsolidationScheduler] = None

# In-memory session message history (keyed by session_id)
# In production this would be persisted, but for hackathon scope this is fine
_session_histories: dict[str, list[dict]] = {}
_session_step_logs: dict[str, list[str]] = {}


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan — startup & shutdown
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global store, active_agent, scheduler

    db_path = os.getenv("DB_PATH", "./engram.db")
    chroma_path = os.getenv("CHROMA_PATH", "./chroma_store")

    store = MemoryStore(db_path=db_path, chroma_path=chroma_path)
    await store.init()

    active_agent = ActiveAgent(store=store)

    scheduler = ConsolidationScheduler(store=store)
    scheduler.start()

    log.info("Engram backend started")
    yield

    scheduler.stop()
    log.info("Engram backend stopped")


# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Engram API",
    description="Memory agent backend — Track 1: MemoryAgent",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Request / Response models
# ─────────────────────────────────────────────────────────────────────────────

class StartSessionRequest(BaseModel):
    playbook_name: Optional[str] = Field(None, description="Optionally pre-load a specific playbook")

class StartSessionResponse(BaseModel):
    session_id: str
    playbook: Optional[dict] = None
    initial_memories: list[dict] = []

class ChatRequest(BaseModel):
    session_id: str
    message: str

class EndSessionRequest(BaseModel):
    session_id: str
    task_completed: bool
    session_summary: str
    playbook_used: bool = False

class WriteMemoryRequest(BaseModel):
    session_id: str
    content: str
    memory_type: str = "semantic"
    importance_score: float = 0.5
    tags: Optional[list[str]] = None

class QueryMemoryRequest(BaseModel):
    query: str
    session_id: str
    token_budget: int = 4000


# ─────────────────────────────────────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "service": "engram-api"}


# ─────────────────────────────────────────────────────────────────────────────
# Session routes
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/sessions/start", response_model=StartSessionResponse)
async def start_session(req: StartSessionRequest):
    """
    Start a new session. Optionally pre-loads a playbook and primes
    the context with relevant memories.
    """
    playbook = None
    if req.playbook_name:
        playbook = await store.get_playbook(req.playbook_name)

    # Link the session to the loaded playbook so run count / averages update.
    session = await store.create_session(
        playbook_id=playbook["id"] if playbook else None
    )

    # Prime context — load relevant memories even before first message
    initial_memories = await active_agent.get_session_context(
        query=req.playbook_name or "general workflow",
        session_id=session.id,
    )

    _session_histories[session.id] = []
    _session_step_logs[session.id] = []

    return StartSessionResponse(
        session_id=session.id,
        playbook=playbook,
        initial_memories=initial_memories,
    )


@app.post("/sessions/{session_id}/chat")
async def chat(session_id: str, req: ChatRequest):
    """
    Send a message to the active agent. Returns streaming response.
    """
    if session_id not in _session_histories:
        raise HTTPException(status_code=404, detail="Session not found. Call /sessions/start first.")

    async def event_stream():
        history = _session_histories[session_id]
        async for chunk in active_agent.run_session(
            user_message=req.message,
            session_id=session_id,
            message_history=history,
        ):
            # Track step logs from agent responses
            if chunk.get("type") == "done" and chunk.get("content"):
                history.append({"role": "user", "content": req.message})
                history.append({"role": "assistant", "content": chunk["content"]})
                _session_step_logs[session_id].append(
                    f"User: {req.message[:80]} → Agent: {chunk['content'][:80]}"
                )

            yield f"data: {json.dumps(chunk)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/sessions/{session_id}/end")
async def end_session(session_id: str, req: EndSessionRequest):
    """
    End a session. Triggers:
    1. Reflection step (agent writes playbook)
    2. Session-end consolidation (activates pending memories)
    3. Records session metrics for learning curve
    """
    if session_id not in _session_histories:
        raise HTTPException(status_code=404, detail="Session not found")

    history = _session_histories[session_id]
    steps_taken = _session_step_logs.get(session_id, [])

    # 1. Reflection — agent writes playbook and semantic memories
    reflection_result = await active_agent.run_reflection(
        session_id=session_id,
        session_summary=req.session_summary,
        steps_taken=steps_taken,
        message_history=history,
    )

    # 2. Estimate token usage from history
    tokens_used = sum(
        len(m.get("content", "").split()) * 1.3  # Rough token estimate
        for m in history
        if isinstance(m.get("content"), str)
    )

    # 3. Record session metrics
    await store.end_session(
        session_id=session_id,
        task_completed=req.task_completed,
        steps_taken=len(steps_taken),
        tokens_used=int(tokens_used),
        messages=history,
        playbook_used=req.playbook_used,
    )

    # 4. Trigger session-end consolidation (activates pending memories)
    await scheduler.trigger_on_session_end(session_id)

    # Cleanup in-memory state
    del _session_histories[session_id]
    del _session_step_logs[session_id]

    return {
        "ok": True,
        "session_id": session_id,
        "reflection": reflection_result,
        "steps_recorded": len(steps_taken),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Memory routes
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/memory/write")
async def write_memory(req: WriteMemoryRequest):
    memory = await store.write_memory(
        session_id=req.session_id,
        content=req.content,
        memory_type=MemoryType(req.memory_type),
        importance_score=req.importance_score,
        tags=req.tags,
    )
    return {"ok": True, "memory_id": memory.id, "status": memory.status}


@app.post("/memory/query")
async def query_memory(req: QueryMemoryRequest):
    memories = await store.query_memory(
        query=req.query,
        session_id=req.session_id,
        token_budget=req.token_budget,
    )
    return {"ok": True, "memories": memories, "count": len(memories)}


@app.get("/memory/all")
async def get_all_memories():
    """Active + pending + deprecated memories for the graph (includes superseded_by)."""
    memories = await store.get_all_memories_for_graph()
    return {
        "memories": [
            {
                "id": m.id,
                "content": m.content,
                "memory_type": m.memory_type,
                "status": m.status,
                "importance_score": m.importance_score,
                "recency_score": m.recency_score,
                "access_count": m.access_count,
                "superseded_by": m.superseded_by,
                "deprecated_reason": m.deprecated_reason,
                "created_at": m.created_at.isoformat(),
            }
            for m in memories
        ]
    }


# ─────────────────────────────────────────────────────────────────────────────
# Playbook routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/playbooks")
async def list_playbooks():
    playbooks = await store.list_playbooks()
    return {"playbooks": playbooks}


@app.get("/playbooks/{name}")
async def get_playbook(name: str):
    playbook = await store.get_playbook(name)
    if not playbook:
        raise HTTPException(status_code=404, detail=f"Playbook '{name}' not found")
    return playbook


# ─────────────────────────────────────────────────────────────────────────────
# Consolidation routes
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/consolidation/trigger")
async def trigger_consolidation():
    """Manually trigger a full consolidation run (for demo and testing)."""
    result = await scheduler.trigger_manual()
    return result


@app.get("/consolidation/history")
async def get_consolidation_history():
    runs = await store.get_recent_consolidation_runs(limit=20)
    return {"runs": runs}


@app.get("/consolidation/status")
async def get_consolidation_status():
    """Report whether the Forgetting Engine scheduler is running and when it next fires."""
    return scheduler.status()


# ─────────────────────────────────────────────────────────────────────────────
# Eval / metrics routes (for the learning curve visualization)
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/metrics/sessions")
async def get_session_metrics():
    """Returns per-session metrics for learning curve chart."""
    metrics = await store.get_session_metrics()
    return {"sessions": metrics}


@app.get("/metrics/summary")
async def get_summary_metrics():
    """High-level summary metrics for the dashboard."""
    sessions = await store.get_session_metrics()
    memories = await store.get_all_active_memories()
    runs = await store.get_recent_consolidation_runs(limit=1)

    total_sessions = len(sessions)
    completed_sessions = sum(1 for s in sessions if s["task_completed"])
    avg_steps = (
        sum(s["steps_taken"] or 0 for s in sessions) / total_sessions
        if total_sessions > 0 else 0
    )
    avg_tokens = (
        sum(s["tokens_used"] or 0 for s in sessions) / total_sessions
        if total_sessions > 0 else 0
    )

    # Improvement: first third vs last third (2 sessions → session 1 vs session 2).
    improvement = None
    if total_sessions >= 2:
        third = max(1, total_sessions // 3)
        early = sessions[:third]
        late = sessions[-third:]
        early_tokens = sum(s["tokens_used"] or 0 for s in early) / len(early)
        late_tokens = sum(s["tokens_used"] or 0 for s in late) / len(late)
        if early_tokens > 0:
            improvement = round((1 - late_tokens / early_tokens) * 100, 1)

    last_run = runs[0] if runs else None
    hygiene = await store.get_hygiene_counts()

    return {
        "total_sessions": total_sessions,
        "completed_sessions": completed_sessions,
        "success_rate": round(completed_sessions / total_sessions * 100, 1) if total_sessions > 0 else 0,
        "avg_steps": round(avg_steps, 1),
        "avg_tokens": round(avg_tokens),
        "token_efficiency_improvement_pct": improvement,
        "active_memories": sum(1 for m in memories if m.status == "active"),
        "pending_memories": sum(1 for m in memories if m.status == "pending"),
        "belief_revisions": hygiene["belief_revisions"],
        "duplicates_merged": hygiene["duplicates_merged"],
        "decayed": hygiene["decayed"],
        "last_consolidation": last_run["completed_at"] if last_run else None,
        "total_contradictions_resolved": sum(
            r.get("contradictions_resolved", 0) for r in runs
        ) if runs else 0,
    }



# ─────────────────────────────────────────────────────────────────────────────
# Demo seed route
# ─────────────────────────────────────────────────────────────────────────────

class SeedRequest(BaseModel):
    force: bool = False


@app.post("/demo/seed")
async def seed_demo_route(req: SeedRequest):
    """
    Populate the store with a realistic 3-session workflow demo.
    Shows learning curve + contradiction in the UI immediately.
    Pass force=true to re-seed.
    """
    from api.demo_seed import seed_demo as _seed
    result = await _seed(store, force=req.force)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# MCP server mount (happens after app creation, needs store)
# ─────────────────────────────────────────────────────────────────────────────

# NOTE: MCP is mounted at startup via lifespan because it needs the store.
# We use a startup event to wire it up after the store is initialized.

@app.on_event("startup")
async def mount_mcp():
    """Mount the MCP server once the store is ready."""
    # Import here to avoid circular at module load
    try:
        from mcp.server import mount_mcp_on_app
        mount_mcp_on_app(app, store)
        log.info("MCP server mounted at /mcp")
    except ImportError:
        log.warning("MCP package not available — install with pip install qwen-agent[mcp]")
