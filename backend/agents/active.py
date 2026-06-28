"""
engram/backend/agents/active.py

The Active Agent — Qwen3 non-thinking mode, low latency.

Tools are registered as native Qwen-Agent BaseTool subclasses (tools/memory_tools.py)
and passed by name in function_list. This is the correct pattern for custom tools —
not SSE MCP URLs, which Qwen-Agent does not support in function_list.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from typing import AsyncIterator, Optional

import dashscope
import structlog
from qwen_agent.agents import Assistant

from memory.store import MemoryStore
from tools.memory_tools import ENGRAM_TOOL_NAMES, set_main_loop, set_session, set_store

log = structlog.get_logger(__name__)


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_OPEN_THINK_RE = re.compile(r"<think>.*$", re.DOTALL)
_ORPHAN_THINK_RE = re.compile(r"</?think>")


def _strip_thinking(text: str) -> str:
    """Strip <think> blocks from model output: closed, trailing-open, and orphan tags."""
    if not text:
        return text
    text = _THINK_RE.sub("", text)
    text = _OPEN_THINK_RE.sub("", text)
    text = _ORPHAN_THINK_RE.sub("", text)
    return text.strip()


def _result_ok(content) -> bool:
    """Best-effort success flag for a tool result message (for the UI timeline)."""
    if not isinstance(content, str) or not content:
        return True
    try:
        return bool(json.loads(content).get("ok", True))
    except (ValueError, AttributeError):
        lowered = content.lower()
        return not ("error" in lowered or "not found" in lowered or "exception" in lowered)

ACTIVE_SYSTEM_PROMPT = """You are Engram, a memory agent that learns workflows and improves with every session.

## Your Memory Tools
You have 8 memory tools. Use them constantly — not occasionally.

**At session START (before anything else):**
1. Call list_playbooks → see what workflows already exist
2. Call get_playbook with the workflow name if one exists
3. Call query_memory with the user's request → load relevant context

**Every turn:**
- Call write_memory immediately after learning something new
- Importance scores: 0.9+ = critical facts, 0.5-0.7 = useful context, 0.1-0.4 = low value
- You do NOT need to pass session_id — the current session is applied automatically.

**When a fact changes or you learn something contradicts a stored memory:**
- First query_memory to find the old memory. Each result has an exact "id" field.
- Call deprecate_memory with that EXACT id (copy it verbatim — do not invent ids),
  plus a reason. Then write_memory with the corrected fact.
- If you don't have a real id from a query result, do NOT call deprecate_memory —
  just write the new memory.

**When running a playbook (ONLY if get_playbook returned steps):**
- Execute steps in order using each step's "id".
- Call record_step_result with that step "id" after each one (success=true/false).
- Do NOT call record_step_result in free-form chat where no playbook is loaded —
  there are no steps to record against.

**At session END (reflection):**
1. Call write_playbook ONLY if a real multi-step workflow happened (2+ steps).
   For plain conversation, a single fact, or a correction, skip it.
2. Write any NEW semantic memories that should persist (don't re-write facts you
   already saved this session)
3. If you found any outdated memories during the session: call deprecate_memory on them
   (using the exact id from a query_memory result)

## What makes a good memory
Be specific. "Stripe /v1/charges deprecated June 2026" beats "API changed".
Tag memories with relevant keywords for retrieval.

## What the judges see
Your memory store is visible. Make it informative — show reasoning, show corrections.
"""

REFLECTION_PROMPT = """The workflow session has completed. Perform the reflection step now.

Session summary: {session_summary}

Steps taken:
{steps_taken}

Do this in order:
1. ONLY IF this session executed a real multi-step workflow (2+ distinct
   procedural actions), call write_playbook to distill those steps into a
   reusable workflow. If the session was just conversation, a single fact, or
   a correction, DO NOT call write_playbook — there is no workflow to save.
2. Call write_memory for any semantic facts that should persist across sessions
   (skip facts you already wrote during the session — don't duplicate them).
3. Call deprecate_memory for any memories you discovered were outdated this
   session, using the exact id from a query_memory result.

Keep it minimal and non-redundant — the consolidator will deprecate duplicates.
"""


class ActiveAgent:
    def __init__(self, store: MemoryStore):
        self.store = store
        # Inject store into all tool instances
        set_store(store)

        # DashScope keys are region-bound. Our account is International (Singapore),
        # which the SDK does NOT default to — the default is the China (Beijing)
        # endpoint, which rejects intl keys with "InvalidApiKey". Point it at the
        # right region (matches memory/embeddings.py).
        dashscope.base_http_api_url = os.getenv(
            "DASHSCOPE_BASE_URL", "https://dashscope-intl.aliyuncs.com/api/v1"
        )

        self._llm_cfg = {
            "model": os.getenv("ACTIVE_MODEL", "qwen3-32b"),
            "model_type": "qwen_dashscope",
            "generate_cfg": {
                "enable_thinking": False,
                "top_p": 0.8,
                "max_input_tokens": 28000,
                # The Singapore (intl) endpoint defaults to incremental_output=True
                # (delta streaming). Qwen-Agent 0.0.10's _full_stream_output assumes
                # CUMULATIVE chunks, so deltas leave only the last fragment in the
                # message (garbled output / leaked ✿FUNCTION✿ markers). Force
                # cumulative streaming to match what Qwen-Agent expects.
                "incremental_output": False,
            },
        }
        self._agent: Optional[Assistant] = None

    def _get_agent(self) -> Assistant:
        if self._agent is None:
            self._agent = Assistant(
                llm=self._llm_cfg,
                system_message=ACTIVE_SYSTEM_PROMPT,
                function_list=ENGRAM_TOOL_NAMES,  # Native BaseTool names
            )
        return self._agent

    async def run_session(
        self,
        user_message: str,
        session_id: str,
        message_history: list[dict],
    ) -> AsyncIterator[dict]:
        """Run one turn of the active agent. Yields SSE chunks.

        qwen_agent's `agent.run()` is a *blocking* generator and its tool calls run
        synchronously. We drive it on a worker thread so the main event loop stays
        free — tools then bridge their store coroutines back to this loop (see
        tools/memory_tools._run). Events flow worker → loop via a thread-safe queue.
        """
        agent = self._get_agent()
        messages = list(message_history)
        messages.append({"role": "user", "content": user_message})

        loop = asyncio.get_running_loop()
        set_main_loop(loop)
        set_session(session_id)

        queue: asyncio.Queue = asyncio.Queue()
        _DONE = object()

        def produce() -> None:
            response_text = ""
            start_time = time.time()
            token_count = 0
            emitted_calls: set[int] = set()
            emitted_results: set[int] = set()

            def emit(event: dict) -> None:
                loop.call_soon_threadsafe(queue.put_nowait, event)

            try:
                for response_chunk in agent.run(messages=messages):
                    last_idx = len(response_chunk) - 1
                    for i, msg in enumerate(response_chunk):
                        if not isinstance(msg, dict):
                            msg = msg.model_dump() if hasattr(msg, "model_dump") else dict(msg)
                        role = msg.get("role")
                        fc = msg.get("function_call")

                        if role == "assistant" and fc:
                            # Emit the call only once finalized (a later message
                            # exists), so streamed args aren't shown half-built.
                            if i not in emitted_calls and i < last_idx:
                                emitted_calls.add(i)
                                emit({
                                    "type": "tool_call",
                                    "tool": fc.get("name", ""),
                                    "name": fc.get("name", ""),
                                    "arguments": fc.get("arguments", ""),
                                    "call_index": i,
                                    "session_id": session_id,
                                })
                        elif role == "function":
                            if i not in emitted_results:
                                emitted_results.add(i)
                                emit({
                                    "type": "tool_result",
                                    "name": msg.get("name", ""),
                                    "ok": _result_ok(msg.get("content")),
                                    "call_index": i - 1,  # call sits just before its result
                                    "session_id": session_id,
                                })
                        elif role == "assistant":
                            content = msg.get("content", "")
                            if content and isinstance(content, str):
                                response_text = _strip_thinking(content)
                                token_count = len(response_text.split())

                    emit({"type": "chunk", "content": response_text, "session_id": session_id})

                elapsed = time.time() - start_time
                log.info("Agent turn complete", session_id=session_id[:8], elapsed=round(elapsed, 2))
                emit({
                    "type": "done",
                    "content": response_text,
                    "session_id": session_id,
                    "elapsed_seconds": round(elapsed, 2),
                    "approx_tokens": token_count,
                })
            except Exception as e:
                log.error("Active agent error", session_id=session_id[:8], error=str(e))
                emit({"type": "error", "error": str(e), "session_id": session_id})
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, _DONE)

        producer = loop.run_in_executor(None, produce)
        try:
            while True:
                item = await queue.get()
                if item is _DONE:
                    break
                yield item
        finally:
            await producer

    async def run_reflection(
        self,
        session_id: str,
        session_summary: str,
        steps_taken: list[str],
        message_history: list[dict],
    ) -> str:
        """Reflection step at session end — agent writes playbook + semantic memories."""
        agent = self._get_agent()
        reflection_msg = REFLECTION_PROMPT.format(
            session_summary=session_summary,
            steps_taken="\n".join(f"  {i+1}. {s}" for i, s in enumerate(steps_taken)),
        )
        messages = list(message_history)
        messages.append({"role": "user", "content": reflection_msg})

        loop = asyncio.get_running_loop()
        set_main_loop(loop)
        set_session(session_id)

        def produce() -> str:
            # Blocking agent.run() on a worker thread; tools bridge back to `loop`.
            response_text = ""
            for chunk in agent.run(messages=messages):
                for msg in chunk:
                    if isinstance(msg, dict) and msg.get("role") == "assistant":
                        content = msg.get("content", "")
                        if content and isinstance(content, str):
                            response_text = _strip_thinking(content)
            return response_text

        log.info("Reflection started", session_id=session_id[:8])
        try:
            response_text = await loop.run_in_executor(None, produce)
        except Exception as e:
            log.error("Reflection error", session_id=session_id[:8], error=str(e))
            return f"Reflection error: {str(e)}"

        log.info("Reflection complete", session_id=session_id[:8])
        return response_text

    async def get_session_context(self, query: str, session_id: str) -> list[dict]:
        """Pre-load relevant memories at session start."""
        return await self.store.query_memory(
            query=query,
            session_id=session_id,
            token_budget=int(os.getenv("CONTEXT_BUDGET_TOKENS", "4000")),
        )
