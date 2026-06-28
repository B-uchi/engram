"""
engram/backend/tools/memory_tools.py

Native Qwen-Agent BaseTool implementations for all Engram memory operations.

Why BaseTool instead of MCP SSE:
  Qwen-Agent's function_list expects either a string (built-in tool name) or
  a dict with {"mcpServers": {"name": {"command": ..., "args": [...]}}} for
  stdio-based MCP servers. SSE-based MCP servers are not supported in this
  position. BaseTool is the correct pattern for custom tools.

  The MCP server in mcp/server.py is retained as a standalone server for
  external integrations and the hackathon architecture diagram — it's just
  not used as the agent's function_list.
"""

from __future__ import annotations

import json
from typing import Union, TYPE_CHECKING

import asyncio as _asyncio

from qwen_agent.tools.base import BaseTool, register_tool

# Enable nested loops where supported; uvloop raises ValueError (handled by _run).
try:
    import nest_asyncio
    nest_asyncio.apply()
except (ImportError, ValueError):
    pass

if TYPE_CHECKING:
    from memory.store import MemoryStore


# The MemoryStore's async engine is bound to the main event loop. The agent's
# (synchronous) tool calls run inside a worker thread (see ActiveAgent.run_session),
# so we bridge each coroutine back to that main loop — the store's DB connections
# must always be driven on the loop that created them, otherwise SQLAlchemy/aiosqlite
# raises "got Future attached to a different loop" (and nest_asyncio can't patch
# uvloop, which is what uvicorn uses).
_main_loop: "_asyncio.AbstractEventLoop | None" = None


def set_main_loop(loop: "_asyncio.AbstractEventLoop") -> None:
    global _main_loop
    _main_loop = loop


# Active session_id, injected by the agent each turn so tools don't rely on the
# LLM echoing a UUID. Tools fall back to this when session_id is omitted.
_current_session: "str | None" = None


def set_session(session_id: str) -> None:
    global _current_session
    _current_session = session_id


def _session_id(params: dict) -> str:
    return params.get("session_id") or _current_session or "unknown"


def _run(coro):
    """Run a store coroutine from a synchronous tool call, on the store's own loop."""
    loop = _main_loop
    if loop is not None and loop.is_running():
        try:
            current = _asyncio.get_running_loop()
        except RuntimeError:
            current = None
        # Expected path: we're in the agent worker thread, the main loop is free.
        if current is not loop:
            return _asyncio.run_coroutine_threadsafe(coro, loop).result(timeout=30)
    # Standalone context (eval harness, scripts) — no shared running loop.
    return _asyncio.run(coro)

# Module-level store reference — injected at startup by active.py
_store: "MemoryStore | None" = None


def set_store(store: "MemoryStore") -> None:
    global _store
    _store = store


def get_store() -> "MemoryStore":
    if _store is None:
        raise RuntimeError("Memory store not injected. Call set_store() at startup.")
    return _store


# ─────────────────────────────────────────────────────────────────────────────
# write_memory
# ─────────────────────────────────────────────────────────────────────────────

@register_tool("write_memory")
class WriteMemoryTool(BaseTool):
    description = (
        "Write a new memory to the Engram persistent memory store. "
        "Use memory_type='episodic' for session observations, "
        "'semantic' for learned facts and preferences, "
        "'procedural' for workflow steps and how-to knowledge. "
        "Always write important facts immediately after learning them."
    )
    parameters = [
        {"name": "content", "type": "string", "description": "The memory content to store", "required": True},
        {"name": "memory_type", "type": "string", "description": "One of: episodic, semantic, procedural", "required": True},
        {"name": "importance_score", "type": "number", "description": "Importance 0.0-1.0 (default 0.5). Use 0.9+ for critical facts.", "required": False},
        {"name": "tags", "type": "array", "description": "Keywords for retrieval (e.g. ['stripe', 'api', 'endpoint'])", "required": False},
        {"name": "summary", "type": "string", "description": "Optional shorter summary for context-packing", "required": False},
        {"name": "session_id", "type": "string", "description": "Optional — the current session is used automatically if omitted", "required": False},
    ]

    def call(self, params: Union[str, dict], **kwargs) -> str:
        from memory.schema import MemoryType
        if isinstance(params, str):
            params = json.loads(params)
        store = get_store()
        memory = _run(
            store.write_memory(
                session_id=_session_id(params),
                content=params["content"],
                memory_type=MemoryType(params["memory_type"]),
                importance_score=float(params.get("importance_score", 0.5)),
                tags=params.get("tags"),
                summary=params.get("summary"),
            )
        )
        return json.dumps({"ok": True, "memory_id": memory.id, "status": memory.status})


# ─────────────────────────────────────────────────────────────────────────────
# query_memory
# ─────────────────────────────────────────────────────────────────────────────

@register_tool("query_memory")
class QueryMemoryTool(BaseTool):
    description = (
        "Retrieve relevant memories from the persistent store for the current query. "
        "Uses multi-signal ranking (semantic + BM25 + entity overlap + recency + importance) "
        "and packs results into the token budget. "
        "Call this at the START of every session and before any major decision."
    )
    parameters = [
        {"name": "query", "type": "string", "description": "What to search for in memory", "required": True},
        {"name": "token_budget", "type": "integer", "description": "Max tokens to use for memory context (default 4000)", "required": False},
        {"name": "session_id", "type": "string", "description": "Optional — the current session is used automatically if omitted", "required": False},
    ]

    def call(self, params: Union[str, dict], **kwargs) -> str:
        if isinstance(params, str):
            params = json.loads(params)
        store = get_store()
        memories = _run(
            store.query_memory(
                query=params["query"],
                session_id=_session_id(params),
                token_budget=int(params.get("token_budget", 4000)),
            )
        )
        return json.dumps({"ok": True, "count": len(memories), "memories": memories})


# ─────────────────────────────────────────────────────────────────────────────
# get_playbook
# ─────────────────────────────────────────────────────────────────────────────

@register_tool("get_playbook")
class GetPlaybookTool(BaseTool):
    description = (
        "Retrieve the latest active version of a named playbook with all its steps. "
        "Call this at session start to check if a workflow already exists. "
        "If found, follow the steps in order and call record_step_result after each one."
    )
    parameters = [
        {"name": "name", "type": "string", "description": "The playbook name to retrieve", "required": True},
    ]

    def call(self, params: Union[str, dict], **kwargs) -> str:
        if isinstance(params, str):
            params = json.loads(params)
        store = get_store()
        playbook = _run(
            store.get_playbook(params["name"])
        )
        if playbook is None:
            return json.dumps({"ok": False, "error": f"No active playbook named '{params['name']}'"})
        return json.dumps({"ok": True, "playbook": playbook})


# ─────────────────────────────────────────────────────────────────────────────
# write_playbook
# ─────────────────────────────────────────────────────────────────────────────

@register_tool("write_playbook")
class WritePlaybookTool(BaseTool):
    description = (
        "Distill the current session into a structured, reusable playbook. "
        "Call this at the END of every completed workflow session — whether or not "
        "a playbook already existed. The new version incorporates improvements and corrections. "
        "Include ALL steps with titles, descriptions, tools used, expected outputs, and edge cases."
    )
    parameters = [
        {"name": "name", "type": "string", "description": "Unique name identifying this workflow type", "required": True},
        {"name": "description", "type": "string", "description": "What this playbook accomplishes", "required": True},
        {
            "name": "steps",
            "type": "array",
            "description": "Ordered list of workflow steps",
            "required": True,
        },
        {"name": "session_id", "type": "string", "description": "Optional — the current session is used automatically if omitted", "required": False},
    ]

    def call(self, params: Union[str, dict], **kwargs) -> str:
        if isinstance(params, str):
            params = json.loads(params)
        store = get_store()
        playbook = _run(
            store.write_playbook(
                name=params["name"],
                description=params["description"],
                session_id=_session_id(params),
                steps=params["steps"],
            )
        )
        return json.dumps({
            "ok": True,
            "playbook_id": playbook.id,
            "name": playbook.name,
            "version": playbook.version,
            "steps_written": len(params["steps"]),
        })


# ─────────────────────────────────────────────────────────────────────────────
# deprecate_memory
# ─────────────────────────────────────────────────────────────────────────────

@register_tool("deprecate_memory")
class DeprecateMemoryTool(BaseTool):
    description = (
        "Mark a memory as deprecated — it was superseded by new, more accurate information. "
        "Use this when you discover a stored fact is outdated or wrong. "
        "The memory is preserved for audit but removed from active retrieval. "
        "Always provide a clear reason explaining what changed."
    )
    parameters = [
        {"name": "memory_id", "type": "string", "description": "ID of the memory to deprecate", "required": True},
        {"name": "reason", "type": "string", "description": "Why this memory is outdated or wrong", "required": True},
        {"name": "superseded_by_id", "type": "string", "description": "ID of the newer memory that replaces this one", "required": False},
    ]

    def call(self, params: Union[str, dict], **kwargs) -> str:
        if isinstance(params, str):
            params = json.loads(params)
        store = get_store()
        memory = _run(
            store.deprecate_memory(
                memory_id=params["memory_id"],
                reason=params["reason"],
                superseded_by_id=params.get("superseded_by_id"),
            )
        )
        if memory is None:
            return json.dumps({
                "ok": False,
                "error": f"No memory with id '{params['memory_id']}'. "
                         f"Use the exact 'id' from a query_memory result.",
            })
        return json.dumps({"ok": True, "memory_id": memory.id, "status": memory.status})


# ─────────────────────────────────────────────────────────────────────────────
# update_memory
# ─────────────────────────────────────────────────────────────────────────────

@register_tool("update_memory")
class UpdateMemoryTool(BaseTool):
    description = "Update the content of an existing memory. Use for minor corrections that don't warrant creating a new memory."
    parameters = [
        {"name": "memory_id", "type": "string", "description": "ID of the memory to update", "required": True},
        {"name": "content", "type": "string", "description": "The corrected content", "required": True},
        {"name": "reason", "type": "string", "description": "Why the content is being updated", "required": True},
    ]

    def call(self, params: Union[str, dict], **kwargs) -> str:
        if isinstance(params, str):
            params = json.loads(params)
        store = get_store()
        memory = _run(
            store.update_memory(
                memory_id=params["memory_id"],
                content=params["content"],
                reason=params["reason"],
            )
        )
        if memory is None:
            return json.dumps({
                "ok": False,
                "error": f"No memory with id '{params['memory_id']}'. "
                         f"Use the exact 'id' from a query_memory result.",
            })
        return json.dumps({"ok": True, "memory_id": memory.id})


# ─────────────────────────────────────────────────────────────────────────────
# list_playbooks
# ─────────────────────────────────────────────────────────────────────────────

@register_tool("list_playbooks")
class ListPlaybooksTool(BaseTool):
    description = "List all active playbooks by name. Use at session start to discover what workflows already exist."
    parameters = []

    def call(self, params: Union[str, dict], **kwargs) -> str:
        store = get_store()
        playbooks = _run(store.list_playbooks())
        return json.dumps({"ok": True, "playbooks": playbooks, "count": len(playbooks)})


# ─────────────────────────────────────────────────────────────────────────────
# record_step_result
# ─────────────────────────────────────────────────────────────────────────────

@register_tool("record_step_result")
class RecordStepResultTool(BaseTool):
    description = (
        "Record the success or failure of a specific playbook step. "
        "Call after executing each step so the system can track reliability over time. "
        "Steps with high failure rates are flagged by the Forgetting Engine for review."
    )
    parameters = [
        {"name": "step_id", "type": "string", "description": "The PlaybookStep ID", "required": True},
        {"name": "success", "type": "boolean", "description": "True if the step succeeded", "required": True},
        {"name": "failure_reason", "type": "string", "description": "If failed, describe what went wrong", "required": False},
    ]

    def call(self, params: Union[str, dict], **kwargs) -> str:
        from sqlalchemy import select
        from memory.schema import PlaybookStep
        if isinstance(params, str):
            params = json.loads(params)
        store = get_store()

        async def _record():
            async with store._session() as db:
                result = await db.execute(
                    select(PlaybookStep).where(PlaybookStep.id == params["step_id"])
                )
                step = result.scalar_one_or_none()
                if not step:
                    return {"ok": False, "error": f"Step {params['step_id']} not found"}
                if params["success"]:
                    step.record_success()
                else:
                    step.record_failure()
                await db.commit()
                return {"ok": True, "step_id": step.id, "success_rate": step.success_rate}

        result = _run(_record())
        return json.dumps(result)


# ─────────────────────────────────────────────────────────────────────────────
# Tool name list — passed to function_list
# ─────────────────────────────────────────────────────────────────────────────

ENGRAM_TOOL_NAMES = [
    "write_memory",
    "query_memory",
    "get_playbook",
    "write_playbook",
    "deprecate_memory",
    "update_memory",
    "list_playbooks",
    "record_step_result",
]
