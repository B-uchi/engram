"""
engram/backend/memory/store.py

The memory store handles all persistence operations.
Retrieval is multi-signal: semantic (ChromaDB) + BM25 keyword + entity overlap
+ recency + importance — fused into a single relevance score, then budget-packed.

This is where the "context-window budgeted recall" mechanism lives.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from typing import Optional

import chromadb
import structlog
from rank_bm25 import BM25Okapi
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select, and_, update, func

from memory.schema import (
    Base,
    ConsolidationRun,
    Memory,
    MemoryDiff,
    MemoryStatus,
    MemoryTag,
    MemoryType,
    Playbook,
    PlaybookStep,
    Session,
    new_id,
    utcnow,
)

log = structlog.get_logger(__name__)

def count_tokens(text: str) -> int:
    """Offline estimator: ~4 chars/token for BPE languages. No network needed."""
    return max(1, len(text) // 4)


class MemoryStore:
    def __init__(self, db_path: str, chroma_path: str):
        self.db_path = db_path
        self.chroma_path = chroma_path
        self._engine = None
        self._session_factory = None
        self._chroma_client = None
        self._collection = None

    async def init(self) -> None:
        """Initialize DB and ChromaDB. Call once at startup."""
        engine = create_async_engine(
            f"sqlite+aiosqlite:///{self.db_path}",
            connect_args={"check_same_thread": False},
            echo=False,
        )
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        self._engine = engine
        self._session_factory = sessionmaker(
            engine, class_=AsyncSession, expire_on_commit=False
        )

        from memory.embeddings import DashScopeEmbeddingFunction
        self._embedding_fn = DashScopeEmbeddingFunction()
        self._chroma_client = chromadb.PersistentClient(path=self.chroma_path)
        self._collection = self._chroma_client.get_or_create_collection(
            name="engram_memories",
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

        log.info("MemoryStore initialized", db=self.db_path, chroma=self.chroma_path)

    def _session(self) -> AsyncSession:
        return self._session_factory()

    # -------------------------------------------------------------------------
    # WRITE OPERATIONS
    # -------------------------------------------------------------------------

    async def write_memory(
        self,
        session_id: str,
        content: str,
        memory_type: MemoryType,
        importance_score: float = 0.5,
        tags: Optional[list[str]] = None,
        summary: Optional[str] = None,
    ) -> Memory:
        """Write a new memory. Status starts as PENDING until consolidation."""
        memory = Memory(
            id=new_id(),
            session_id=session_id,
            content=content,
            summary=summary or content[:200],
            memory_type=memory_type,
            status=MemoryStatus.PENDING,
            importance_score=max(0.0, min(1.0, importance_score)),
            recency_score=1.0,
        )

        async with self._session() as db:
            db.add(memory)
            if tags:
                for tag in tags:
                    db.add(MemoryTag(memory_id=memory.id, tag=tag.lower()))
            await db.commit()
            await db.refresh(memory)

        # Add to ChromaDB for semantic search
        try:
            self._collection.add(
                ids=[memory.id],
                documents=[content],
                metadatas={
                    "memory_type": memory_type,
                    "session_id": session_id,
                    "status": MemoryStatus.PENDING,
                    "importance_score": importance_score,
                    "created_at": utcnow().isoformat(),
                },
            )
            async with self._session() as db:
                await db.execute(
                    update(Memory).where(Memory.id == memory.id).values(chroma_id=memory.id)
                )
                await db.commit()
        except Exception as e:
            log.warning("ChromaDB add failed (non-fatal)", error=str(e), memory_id=memory.id)

        log.info("Memory written", id=memory.id[:8], type=memory_type, importance=importance_score)
        return memory

    async def write_playbook(
        self,
        name: str,
        description: str,
        session_id: str,
        steps: list[dict],
    ) -> Playbook:
        """
        Write a structured playbook distilled from a session.
        steps: list of {title, description, tool_used, expected_output,
                         decision_point, edge_cases}
        """
        # First write the root memory record for the playbook
        memory = await self.write_memory(
            session_id=session_id,
            content=f"PLAYBOOK: {name}\n{description}",
            memory_type=MemoryType.PROCEDURAL,
            importance_score=0.9,  # Playbooks are high-importance
            tags=[name.lower().replace(" ", "_"), "playbook"],
        )

        async with self._session() as db:
            # Check if a playbook with this name already exists
            result = await db.execute(
                select(Playbook)
                .where(and_(Playbook.name == name, Playbook.status == MemoryStatus.ACTIVE))
                .order_by(Playbook.version.desc())
                .limit(1)
            )
            existing = result.scalar_one_or_none()
            version = (existing.version + 1) if existing else 1

            if existing:
                existing.status = MemoryStatus.DEPRECATED

            playbook = Playbook(
                id=new_id(),
                name=name,
                description=description,
                version=version,
                status=MemoryStatus.ACTIVE,
            )
            db.add(playbook)
            await db.flush()

            for i, step in enumerate(steps, 1):
                pb_step = PlaybookStep(
                    id=new_id(),
                    memory_id=memory.id,
                    playbook_id=playbook.id,
                    step_number=i,
                    title=step["title"],
                    description=step["description"],
                    tool_used=step.get("tool_used"),
                    expected_output=step.get("expected_output"),
                    decision_point=step.get("decision_point", False),
                    edge_cases=json.dumps(step.get("edge_cases", [])),
                    status=MemoryStatus.ACTIVE,
                )
                db.add(pb_step)

            await db.commit()
            await db.refresh(playbook)

        log.info("Playbook written", name=name, version=version, steps=len(steps))
        return playbook

    async def update_memory(self, memory_id: str, content: str, reason: str) -> Optional[Memory]:
        """Update memory content. Returns None if the memory doesn't exist."""
        async with self._session() as db:
            result = await db.execute(select(Memory).where(Memory.id == memory_id))
            memory = result.scalar_one_or_none()
            if not memory:
                return None
            memory.content = content
            memory.updated_at = utcnow()
            await db.commit()
            await db.refresh(memory)

        # Update ChromaDB
        try:
            self._collection.update(ids=[memory_id], documents=[content])
        except Exception as e:
            log.warning("ChromaDB update failed", error=str(e))

        return memory

    async def deprecate_memory(
        self,
        memory_id: str,
        reason: str,
        superseded_by_id: Optional[str] = None,
    ) -> Optional[Memory]:
        """Deprecate a memory (kept for audit, excluded from retrieval).

        Returns None if the memory doesn't exist.
        """
        async with self._session() as db:
            result = await db.execute(select(Memory).where(Memory.id == memory_id))
            memory = result.scalar_one_or_none()
            if not memory:
                return None
            memory.deprecate(reason=reason, superseded_by_id=superseded_by_id)
            await db.commit()
            await db.refresh(memory)

        # Update ChromaDB metadata
        try:
            self._collection.update(
                ids=[memory_id],
                metadatas={"status": MemoryStatus.DEPRECATED},
            )
        except Exception as e:
            log.warning("ChromaDB metadata update failed", error=str(e))

        log.info("Memory deprecated", id=memory_id[:8], reason=reason)
        return memory

    async def activate_pending_memories(self, session_id: str) -> int:
        """Move all PENDING memories from a session to ACTIVE after consolidation."""
        async with self._session() as db:
            result = await db.execute(
                select(Memory).where(
                    and_(
                        Memory.session_id == session_id,
                        Memory.status == MemoryStatus.PENDING,
                    )
                )
            )
            memories = result.scalars().all()
            for m in memories:
                m.activate()
            await db.commit()

        log.info("Memories activated", session_id=session_id[:8], count=len(memories))
        return len(memories)

    async def activate_memory(self, memory_id: str) -> None:
        """Promote a single memory to ACTIVE (used by the consolidator for merges)."""
        async with self._session() as db:
            result = await db.execute(select(Memory).where(Memory.id == memory_id))
            memory = result.scalar_one_or_none()
            if not memory:
                return
            memory.activate()
            await db.commit()

        try:
            self._collection.update(
                ids=[memory_id],
                metadatas={"status": MemoryStatus.ACTIVE},
            )
        except Exception as e:
            log.warning("ChromaDB status update failed", error=str(e))

    async def activate_all_pending_memories(self) -> int:
        """Promote every PENDING memory with real content to ACTIVE; archive the rest."""
        async with self._session() as db:
            result = await db.execute(
                select(Memory).where(Memory.status == MemoryStatus.PENDING)
            )
            memories = result.scalars().all()
            activated = 0
            for m in memories:
                if m.content and len(m.content.strip()) > 5:
                    m.activate()
                    activated += 1
                else:
                    m.archive()
            await db.commit()

        log.info("All pending memories processed", activated=activated, total=len(memories))
        return activated

    # -------------------------------------------------------------------------
    # RETRIEVAL — Multi-signal with budget packing
    # -------------------------------------------------------------------------

    async def query_memory(
        self,
        query: str,
        session_id: str,
        token_budget: int = 4000,
        memory_types: Optional[list[MemoryType]] = None,
        top_k_semantic: int = 20,
    ) -> list[dict]:
        """
        Retrieve memories ranked by fused relevance score.

        Scoring = (0.4 * semantic) + (0.25 * bm25) + (0.15 * entity_overlap)
                + (0.1 * recency) + (0.1 * importance)

        Results are packed into token_budget greedily (highest score first).
        This is the "context-window budgeted recall" mechanism.
        """
        # Step 1: Semantic candidates from ChromaDB
        semantic_results = self._collection.query(
            query_texts=[query],
            n_results=min(top_k_semantic, self._collection.count() or 1),
            where={"status": {"$in": [MemoryStatus.ACTIVE]}},
        )

        candidate_ids = semantic_results["ids"][0] if semantic_results["ids"] else []
        semantic_scores: dict[str, float] = {}
        if candidate_ids:
            distances = semantic_results["distances"][0]
            for cid, dist in zip(candidate_ids, distances):
                # ChromaDB cosine returns distance (0=identical), convert to similarity
                semantic_scores[cid] = max(0.0, 1.0 - dist)

        # Step 2: Load candidate memories from SQLite
        async with self._session() as db:
            if candidate_ids:
                result = await db.execute(
                    select(Memory).where(
                        and_(
                            Memory.id.in_(candidate_ids),
                            Memory.status == MemoryStatus.ACTIVE,
                        )
                    )
                )
            else:
                # Fallback: load recent active memories if semantic returns nothing
                result = await db.execute(
                    select(Memory)
                    .where(Memory.status == MemoryStatus.ACTIVE)
                    .order_by(Memory.created_at.desc())
                    .limit(top_k_semantic)
                )
            candidates = result.scalars().all()

            # Load tags for entity overlap
            tag_result = await db.execute(
                select(MemoryTag).where(
                    MemoryTag.memory_id.in_([m.id for m in candidates])
                )
            )
            tags_by_memory: dict[str, list[str]] = {}
            for tag in tag_result.scalars().all():
                tags_by_memory.setdefault(tag.memory_id, []).append(tag.tag)

        # Step 3: BM25 on candidate corpus
        corpus_docs = [m.content for m in candidates]
        tokenized_corpus = [doc.lower().split() for doc in corpus_docs]
        bm25 = BM25Okapi(tokenized_corpus) if tokenized_corpus else None
        query_tokens = query.lower().split()

        bm25_scores: dict[str, float] = {}
        if bm25 and candidates:
            raw = bm25.get_scores(query_tokens)
            max_bm25 = max(raw) if max(raw) > 0 else 1.0
            for m, score in zip(candidates, raw):
                bm25_scores[m.id] = score / max_bm25

        # Step 4: Entity overlap (query tokens vs memory tags)
        query_entities = set(query.lower().split())
        entity_scores: dict[str, float] = {}
        for m in candidates:
            mem_tags = set(tags_by_memory.get(m.id, []))
            overlap = len(query_entities & mem_tags)
            entity_scores[m.id] = min(1.0, overlap / max(len(query_entities), 1))

        # Step 5: Recency score (exponential decay from created_at)
        now = datetime.now(timezone.utc)
        recency_scores: dict[str, float] = {}
        for m in candidates:
            created = m.created_at
            if created.tzinfo is None:
                from datetime import timezone as _tz
                created = created.replace(tzinfo=_tz.utc)
            age_days = (now - created).total_seconds() / 86400
            recency_scores[m.id] = math.exp(-0.023 * age_days)  # half-life ~30 days

        # Step 6: Fuse scores
        scored: list[tuple[float, Memory]] = []
        for m in candidates:
            fused = (
                0.40 * semantic_scores.get(m.id, 0.0)
                + 0.25 * bm25_scores.get(m.id, 0.0)
                + 0.15 * entity_scores.get(m.id, 0.0)
                + 0.10 * recency_scores.get(m.id, 1.0)
                + 0.10 * m.importance_score
            )
            scored.append((fused, m))

        scored.sort(key=lambda x: x[0], reverse=True)

        # Step 7: Pack into token budget (greedy, highest score first)
        packed: list[dict] = []
        tokens_used = 0

        for score, memory in scored:
            content = memory.summary or memory.content
            tokens = count_tokens(content)
            if tokens_used + tokens > token_budget:
                continue  # Skip — doesn't fit; keep trying smaller ones
            tokens_used += tokens

            # Record access; keep recency_score in sync with the computed value.
            async with self._session() as db:
                result = await db.execute(select(Memory).where(Memory.id == memory.id))
                m = result.scalar_one_or_none()
                if m:
                    m.touch()
                    m.recency_score = recency_scores.get(memory.id, m.recency_score)
                    await db.commit()

            packed.append({
                "id": memory.id,
                "content": content,
                "memory_type": memory.memory_type,
                "importance_score": memory.importance_score,
                "relevance_score": round(score, 4),
                "status": memory.status,
                "created_at": memory.created_at.isoformat(),
            })

        log.info(
            "Memory query complete",
            query_preview=query[:50],
            candidates=len(candidates),
            packed=len(packed),
            tokens_used=tokens_used,
            budget=token_budget,
        )
        return packed

    async def get_playbook(self, name: str) -> Optional[dict]:
        """Retrieve the latest active version of a named playbook with all steps."""
        async with self._session() as db:
            result = await db.execute(
                select(Playbook)
                .where(and_(Playbook.name == name, Playbook.status == MemoryStatus.ACTIVE))
                .order_by(Playbook.version.desc())
                .limit(1)
            )
            playbook = result.scalar_one_or_none()
            if not playbook:
                return None

            steps_result = await db.execute(
                select(PlaybookStep)
                .where(
                    and_(
                        PlaybookStep.playbook_id == playbook.id,
                        PlaybookStep.status == MemoryStatus.ACTIVE,
                    )
                )
                .order_by(PlaybookStep.step_number)
            )
            steps = steps_result.scalars().all()

        return {
            "id": playbook.id,
            "name": playbook.name,
            "description": playbook.description,
            "version": playbook.version,
            "session_count": playbook.session_count,
            "avg_steps_taken": playbook.avg_steps_taken,
            "avg_tokens_used": playbook.avg_tokens_used,
            "steps": [
                {
                    "id": s.id,
                    "step_number": s.step_number,
                    "title": s.title,
                    "description": s.description,
                    "tool_used": s.tool_used,
                    "expected_output": s.expected_output,
                    "decision_point": s.decision_point,
                    "edge_cases": json.loads(s.edge_cases or "[]"),
                    "success_count": s.success_count,
                    "failure_count": s.failure_count,
                    "attempts": s.success_count + s.failure_count,
                    "success_rate": s.success_rate,
                    "status": s.status,
                }
                for s in steps
            ],
        }

    async def list_playbooks(self) -> list[dict]:
        """List all active playbooks (summary only)."""
        async with self._session() as db:
            result = await db.execute(
                select(Playbook)
                .where(Playbook.status == MemoryStatus.ACTIVE)
                .order_by(Playbook.updated_at.desc())
            )
            playbooks = result.scalars().all()

        return [
            {
                "id": p.id,
                "name": p.name,
                "description": p.description,
                "version": p.version,
                "session_count": p.session_count,
                "last_run_at": p.last_run_at.isoformat() if p.last_run_at else None,
            }
            for p in playbooks
        ]

    async def get_all_active_memories(self) -> list[Memory]:
        """Used by the consolidator to scan all memories due for review."""
        async with self._session() as db:
            result = await db.execute(
                select(Memory).where(
                    Memory.status.in_([MemoryStatus.ACTIVE, MemoryStatus.PENDING])
                )
            )
            return result.scalars().all()

    async def get_hygiene_counts(self) -> dict:
        """Cleanup counts: belief revisions, duplicates merged, decayed.

        Deprecation kind comes from the deprecated_reason prefix; anything not
        tagged REDUNDANT/MERGED counts as a belief revision.
        """
        async with self._session() as db:
            dep_rows = await db.execute(
                select(Memory.deprecated_reason).where(
                    Memory.status == MemoryStatus.DEPRECATED
                )
            )
            reasons = [(r or "") for r in dep_rows.scalars().all()]
            duplicates = sum(
                1 for r in reasons
                if r.startswith("REDUNDANT:") or r.startswith("MERGED:")
                or r.startswith("Merged into")
            )
            revisions = len(reasons) - duplicates

            arch = await db.execute(
                select(func.count())
                .select_from(Memory)
                .where(Memory.status == MemoryStatus.ARCHIVED)
            )
            decayed = int(arch.scalar_one() or 0)

        return {
            "belief_revisions": revisions,
            "duplicates_merged": duplicates,
            "decayed": decayed,
        }

    async def get_all_memories_for_graph(self) -> list[Memory]:
        """Active, pending, and deprecated memories for the graph (newest 200)."""
        async with self._session() as db:
            result = await db.execute(
                select(Memory)
                .where(
                    Memory.status.in_([
                        MemoryStatus.ACTIVE,
                        MemoryStatus.PENDING,
                        MemoryStatus.DEPRECATED,
                    ])
                )
                .order_by(Memory.created_at.desc())
                .limit(200)
            )
            return result.scalars().all()

    async def save_consolidation_run(self, run: ConsolidationRun) -> None:
        async with self._session() as db:
            db.add(run)
            await db.commit()

    async def get_recent_consolidation_runs(self, limit: int = 10) -> list[dict]:
        async with self._session() as db:
            result = await db.execute(
                select(ConsolidationRun)
                .order_by(ConsolidationRun.started_at.desc())
                .limit(limit)
            )
            runs = result.scalars().all()

        return [
            {
                "id": r.id,
                "started_at": r.started_at.isoformat(),
                "completed_at": r.completed_at.isoformat() if r.completed_at else None,
                "triggered_by": r.triggered_by,
                "memories_scanned": r.memories_scanned,
                "memories_activated": r.memories_activated,
                "memories_deprecated": r.memories_deprecated,
                "memories_archived": r.memories_archived,
                "contradictions_found": r.contradictions_found,
                "contradictions_resolved": r.contradictions_resolved,
                "diff_json": json.loads(r.diff_json) if r.diff_json else [],
                "error": r.error,
            }
            for r in runs
        ]

    async def create_session(self, playbook_id: Optional[str] = None) -> Session:
        session = Session(id=new_id(), playbook_id=playbook_id)
        async with self._session() as db:
            db.add(session)
            await db.commit()
            await db.refresh(session)
        return session

    async def end_session(
        self,
        session_id: str,
        task_completed: bool,
        steps_taken: int,
        tokens_used: int,
        messages: list[dict],
        playbook_used: bool = False,
    ) -> Session:
        async with self._session() as db:
            result = await db.execute(select(Session).where(Session.id == session_id))
            session = result.scalar_one_or_none()
            if not session:
                raise ValueError(f"Session {session_id} not found")
            session.end(task_completed, steps_taken, tokens_used)
            session.playbook_used = playbook_used
            session.messages_json = json.dumps(messages)
            await db.commit()
            await db.refresh(session)

        # Update playbook metrics if one was used
        if session.playbook_id:
            await self._update_playbook_metrics(session.playbook_id, steps_taken, tokens_used)

        return session

    async def _update_playbook_metrics(
        self, playbook_id: str, steps_taken: int, tokens_used: int
    ) -> None:
        async with self._session() as db:
            result = await db.execute(select(Playbook).where(Playbook.id == playbook_id))
            playbook = result.scalar_one_or_none()
            if not playbook:
                return
            n = playbook.session_count
            playbook.session_count = n + 1
            # Rolling average
            prev_steps = playbook.avg_steps_taken or steps_taken
            prev_tokens = playbook.avg_tokens_used or tokens_used
            playbook.avg_steps_taken = (prev_steps * n + steps_taken) / (n + 1)
            playbook.avg_tokens_used = (prev_tokens * n + tokens_used) / (n + 1)
            playbook.last_run_at = utcnow()
            await db.commit()

    async def get_session_metrics(self) -> list[dict]:
        """Return per-session metrics for the learning curve UI chart."""
        async with self._session() as db:
            result = await db.execute(
                select(Session)
                .where(Session.ended_at.isnot(None))
                .order_by(Session.started_at.asc())
            )
            sessions = result.scalars().all()

        return [
            {
                "session_number": i + 1,
                "session_id": s.id,
                "task_completed": s.task_completed,
                "steps_taken": s.steps_taken,
                "tokens_used": s.tokens_used,
                "playbook_used": s.playbook_used,
                "started_at": s.started_at.isoformat(),
            }
            for i, s in enumerate(sessions)
        ]
