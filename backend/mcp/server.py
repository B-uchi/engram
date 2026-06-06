"""
engram/backend/mcp/server.py

Custom MCP server that exposes Engram's memory operations as tools
the Qwen-Agent can call natively via the MCP protocol.

Tools exposed:
  - write_memory          Write a new memory (episodic/semantic/procedural)
  - write_playbook        Distill a session into a structured playbook
  - query_memory          Multi-signal retrieval within token budget
  - get_playbook          Fetch a named playbook with all steps
  - deprecate_memory      Mark a memory as superseded
  - update_memory         Update memory content
  - list_playbooks        List all active playbooks
  - record_step_result    Mark a playbook step success/failure
"""

from __future__ import annotations

import json
from typing import Any, Optional

import structlog
from fastapi import FastAPI
from mcp.server import Server
from mcp.server.fastapi import create_mcp_router
from mcp.types import TextContent, Tool

from memory.store import MemoryStore
from memory.schema import MemoryType

log = structlog.get_logger(__name__)


def create_mcp_server(store: MemoryStore) -> Server:
    server = Server("engram-memory")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="write_memory",
                description=(
                    "Write a new memory to the Engram memory store. "
                    "Use memory_type='episodic' for session observations, "
                    "'semantic' for learned facts, 'procedural' for workflow knowledge."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string", "description": "Current session ID"},
                        "content": {"type": "string", "description": "The memory content to store"},
                        "memory_type": {
                            "type": "string",
                            "enum": ["episodic", "semantic", "procedural"],
                            "description": "Type of memory",
                        },
                        "importance_score": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "Importance weight 0.0–1.0 (default 0.5)",
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Keywords and entity tags for retrieval",
                        },
                        "summary": {
                            "type": "string",
                            "description": "Optional shorter summary for context-packing",
                        },
                    },
                    "required": ["session_id", "content", "memory_type"],
                },
            ),
            Tool(
                name="write_playbook",
                description=(
                    "Distill the current session into a structured, reusable playbook. "
                    "Call this at the end of a completed workflow session. "
                    "The playbook will be retrieved and executed in future sessions."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Unique name identifying this workflow",
                        },
                        "description": {
                            "type": "string",
                            "description": "What this playbook accomplishes",
                        },
                        "session_id": {"type": "string"},
                        "steps": {
                            "type": "array",
                            "description": "Ordered list of workflow steps",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "title": {"type": "string"},
                                    "description": {
                                        "type": "string",
                                        "description": "Detailed instruction for this step",
                                    },
                                    "tool_used": {
                                        "type": "string",
                                        "description": "Which tool/API this step invokes",
                                    },
                                    "expected_output": {
                                        "type": "string",
                                        "description": "What a successful execution produces",
                                    },
                                    "decision_point": {
                                        "type": "boolean",
                                        "description": "True if this step requires branching logic",
                                    },
                                    "edge_cases": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "Known failure modes and how to handle them",
                                    },
                                },
                                "required": ["title", "description"],
                            },
                        },
                    },
                    "required": ["name", "description", "session_id", "steps"],
                },
            ),
            Tool(
                name="query_memory",
                description=(
                    "Retrieve relevant memories for the current query. "
                    "Uses multi-signal ranking (semantic + BM25 + entity + recency + importance) "
                    "and packs results into the token budget. Always call this before responding "
                    "to understand what the agent already knows."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The query to find relevant memories for",
                        },
                        "session_id": {"type": "string"},
                        "token_budget": {
                            "type": "integer",
                            "description": "Maximum tokens to use for memory context (default 4000)",
                        },
                        "memory_types": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": ["episodic", "semantic", "procedural"],
                            },
                            "description": "Filter by memory type (omit for all types)",
                        },
                    },
                    "required": ["query", "session_id"],
                },
            ),
            Tool(
                name="get_playbook",
                description=(
                    "Retrieve the latest active version of a named playbook with all its steps. "
                    "Call this at the start of a session to check if a workflow already exists."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "The playbook name to retrieve",
                        },
                    },
                    "required": ["name"],
                },
            ),
            Tool(
                name="deprecate_memory",
                description=(
                    "Mark a memory as deprecated — it was superseded by new information. "
                    "Use this when you detect that a stored fact is now outdated or wrong. "
                    "The memory is kept for audit but removed from active retrieval."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "memory_id": {"type": "string", "description": "ID of memory to deprecate"},
                        "reason": {
                            "type": "string",
                            "description": "Why this memory is being deprecated",
                        },
                        "superseded_by_id": {
                            "type": "string",
                            "description": "ID of the new memory that replaces this one",
                        },
                    },
                    "required": ["memory_id", "reason"],
                },
            ),
            Tool(
                name="update_memory",
                description="Update the content of an existing memory record.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "memory_id": {"type": "string"},
                        "content": {"type": "string", "description": "The new content"},
                        "reason": {"type": "string", "description": "Why the content changed"},
                    },
                    "required": ["memory_id", "content", "reason"],
                },
            ),
            Tool(
                name="list_playbooks",
                description="List all active playbooks by name. Use to discover available workflows.",
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
            Tool(
                name="record_step_result",
                description=(
                    "Record the success or failure of a specific playbook step. "
                    "Call after each step execution so the system can track step reliability."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "step_id": {"type": "string", "description": "PlaybookStep ID"},
                        "success": {"type": "boolean"},
                        "failure_reason": {
                            "type": "string",
                            "description": "If failed, describe what went wrong",
                        },
                    },
                    "required": ["step_id", "success"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        try:
            result = await _dispatch(name, arguments, store)
            return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, default=str))]
        except Exception as e:
            log.error("MCP tool call failed", tool=name, error=str(e))
            return [TextContent(type="text", text=json.dumps({"error": str(e), "tool": name}))]

    return server


async def _dispatch(name: str, args: dict, store: MemoryStore) -> Any:
    if name == "write_memory":
        memory = await store.write_memory(
            session_id=args["session_id"],
            content=args["content"],
            memory_type=MemoryType(args["memory_type"]),
            importance_score=args.get("importance_score", 0.5),
            tags=args.get("tags"),
            summary=args.get("summary"),
        )
        return {
            "ok": True,
            "memory_id": memory.id,
            "status": memory.status,
            "memory_type": memory.memory_type,
        }

    elif name == "write_playbook":
        playbook = await store.write_playbook(
            name=args["name"],
            description=args["description"],
            steps=args["steps"],
            session_id=args["session_id"],
        )
        return {
            "ok": True,
            "playbook_id": playbook.id,
            "name": playbook.name,
            "version": playbook.version,
            "step_count": len(args["steps"]),
        }

    elif name == "query_memory":
        memories = await store.query_memory(
            query=args["query"],
            session_id=args["session_id"],
            token_budget=args.get("token_budget", 4000),
            memory_types=args.get("memory_types"),
        )
        return {"ok": True, "memories": memories, "count": len(memories)}

    elif name == "get_playbook":
        playbook = await store.get_playbook(args["name"])
        if playbook is None:
            return {"ok": False, "error": f"No active playbook named '{args['name']}'"}
        return {"ok": True, "playbook": playbook}

    elif name == "deprecate_memory":
        memory = await store.deprecate_memory(
            memory_id=args["memory_id"],
            reason=args["reason"],
            superseded_by_id=args.get("superseded_by_id"),
        )
        return {"ok": True, "memory_id": memory.id, "status": memory.status}

    elif name == "update_memory":
        memory = await store.update_memory(
            memory_id=args["memory_id"],
            content=args["content"],
            reason=args["reason"],
        )
        return {"ok": True, "memory_id": memory.id}

    elif name == "list_playbooks":
        playbooks = await store.list_playbooks()
        return {"ok": True, "playbooks": playbooks}

    elif name == "record_step_result":
        from sqlalchemy import select
        from memory.schema import PlaybookStep
        # Direct DB update for step metrics
        async with store._session() as db:
            result = await db.execute(
                select(PlaybookStep).where(PlaybookStep.id == args["step_id"])
            )
            step = result.scalar_one_or_none()
            if not step:
                return {"ok": False, "error": f"Step {args['step_id']} not found"}
            if args["success"]:
                step.record_success()
            else:
                step.record_failure()
            await db.commit()
        return {"ok": True, "step_id": args["step_id"], "success": args["success"]}

    else:
        raise ValueError(f"Unknown tool: {name}")


def mount_mcp_on_app(app: FastAPI, store: MemoryStore) -> None:
    """Mount the MCP server as a sub-router on the FastAPI app."""
    mcp_server = create_mcp_server(store)
    router = create_mcp_router(mcp_server)
    app.include_router(router, prefix="/mcp")
