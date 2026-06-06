"""
engram/backend/memory/schema.py

The canonical memory schema for Engram. Every memory written to the store
goes through this schema — no exceptions, no ad-hoc dicts.

Memory Types:
  - episodic:    What happened in a specific session (raw observations)
  - semantic:    General facts the agent has learned (user preferences, domain facts)
  - procedural:  How to do something (playbook steps, executable workflows)

Memory Status:
  - active:      Currently valid and retrievable
  - deprecated:  Superseded by a newer belief (kept for audit, not retrieved)
  - archived:    Too stale/low-value to surface, but not deleted
  - pending:     Written this session, not yet consolidated

Contradiction Resolution:
  When the consolidator detects a conflict between two memories,
  it writes a ConflictRecord, deprecates the loser, and updates the winner.
  The audit trail is never destroyed.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, relationship, sessionmaker


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    pass


class MemoryType(str, Enum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


class MemoryStatus(str, Enum):
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"
    PENDING = "pending"


class Memory(Base):
    """
    The core memory unit. Every fact, observation, and playbook step
    is a Memory row.
    """
    __tablename__ = "memories"

    id = Column(String(36), primary_key=True, default=new_id)
    session_id = Column(String(36), nullable=False, index=True)

    # Content
    content = Column(Text, nullable=False)
    summary = Column(Text, nullable=True)          # Compressed version for context packing
    memory_type = Column(String(20), nullable=False, default=MemoryType.EPISODIC)
    status = Column(String(20), nullable=False, default=MemoryStatus.PENDING, index=True)

    # Scoring — used by decay and retrieval ranking
    importance_score = Column(Float, nullable=False, default=0.5)   # 0.0–1.0, set at write time
    access_count = Column(Integer, nullable=False, default=0)
    last_accessed_at = Column(DateTime(timezone=True), nullable=True)
    recency_score = Column(Float, nullable=False, default=1.0)      # Decays over time

    # Provenance
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow)
    consolidated_at = Column(DateTime(timezone=True), nullable=True)  # Set by consolidator

    # Belief revision fields
    deprecated_at = Column(DateTime(timezone=True), nullable=True)
    deprecated_reason = Column(Text, nullable=True)
    superseded_by = Column(String(36), ForeignKey("memories.id"), nullable=True)

    # Relationships
    superseder = relationship("Memory", remote_side="Memory.id", foreign_keys=[superseded_by])
    playbook_steps = relationship("PlaybookStep", back_populates="memory", cascade="all, delete-orphan")
    tags = relationship("MemoryTag", back_populates="memory", cascade="all, delete-orphan")

    # ChromaDB vector ID (for semantic search)
    chroma_id = Column(String(36), nullable=True, unique=True)

    __table_args__ = (
        Index("ix_memories_type_status", "memory_type", "status"),
        Index("ix_memories_session_type", "session_id", "memory_type"),
    )

    def touch(self) -> None:
        """Record an access — used to update recency and access frequency."""
        self.access_count += 1
        self.last_accessed_at = utcnow()

    def deprecate(self, reason: str, superseded_by_id: Optional[str] = None) -> None:
        self.status = MemoryStatus.DEPRECATED
        self.deprecated_at = utcnow()
        self.deprecated_reason = reason
        if superseded_by_id:
            self.superseded_by = superseded_by_id

    def archive(self) -> None:
        self.status = MemoryStatus.ARCHIVED

    def activate(self) -> None:
        self.status = MemoryStatus.ACTIVE
        self.consolidated_at = utcnow()

    def __repr__(self) -> str:
        return f"<Memory id={self.id[:8]} type={self.memory_type} status={self.status}>"


class PlaybookStep(Base):
    """
    A single executable step within a procedural memory (playbook).
    Steps are individually scoreable and depreciatable — so when one
    step goes stale, only that step is deprecated, not the whole playbook.
    """
    __tablename__ = "playbook_steps"

    id = Column(String(36), primary_key=True, default=new_id)
    memory_id = Column(String(36), ForeignKey("memories.id"), nullable=False, index=True)
    playbook_id = Column(String(36), nullable=False, index=True)  # Groups steps into a workflow

    step_number = Column(Integer, nullable=False)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=False)
    tool_used = Column(String(255), nullable=True)           # Which MCP tool this step invokes
    expected_output = Column(Text, nullable=True)
    decision_point = Column(Boolean, nullable=False, default=False)  # Requires branching logic
    edge_cases = Column(Text, nullable=True)                 # JSON-encoded list of edge cases

    status = Column(String(20), nullable=False, default=MemoryStatus.ACTIVE, index=True)
    deprecated_at = Column(DateTime(timezone=True), nullable=True)
    deprecated_reason = Column(Text, nullable=True)

    success_count = Column(Integer, nullable=False, default=0)   # Times this step succeeded
    failure_count = Column(Integer, nullable=False, default=0)   # Times this step failed
    last_executed_at = Column(DateTime(timezone=True), nullable=True)

    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow)

    memory = relationship("Memory", back_populates="playbook_steps")

    def record_success(self) -> None:
        self.success_count += 1
        self.last_executed_at = utcnow()

    def record_failure(self) -> None:
        self.failure_count += 1
        self.last_executed_at = utcnow()

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 1.0

    def __repr__(self) -> str:
        return f"<PlaybookStep #{self.step_number} '{self.title}' status={self.status}>"


class Playbook(Base):
    """
    A named, versioned workflow — the top-level container for PlaybookSteps.
    Each time the consolidator rewrites a playbook, it bumps the version.
    Old versions are kept for diff display.
    """
    __tablename__ = "playbooks"

    id = Column(String(36), primary_key=True, default=new_id)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    version = Column(Integer, nullable=False, default=1)
    status = Column(String(20), nullable=False, default=MemoryStatus.ACTIVE, index=True)

    session_count = Column(Integer, nullable=False, default=0)    # Times this playbook was run
    avg_steps_taken = Column(Float, nullable=True)                # Tracks efficiency improvement
    avg_tokens_used = Column(Float, nullable=True)
    last_run_at = Column(DateTime(timezone=True), nullable=True)

    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow)

    __table_args__ = (
        UniqueConstraint("name", "version", name="uq_playbook_name_version"),
    )

    def __repr__(self) -> str:
        return f"<Playbook '{self.name}' v{self.version} status={self.status}>"


class ConsolidationRun(Base):
    """
    Audit record for every consolidation cycle the Forgetting Engine runs.
    Produces the memory diff visible in the UI.
    """
    __tablename__ = "consolidation_runs"

    id = Column(String(36), primary_key=True, default=new_id)
    started_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    triggered_by = Column(String(50), nullable=False, default="scheduler")  # scheduler | session_end | manual

    memories_scanned = Column(Integer, nullable=False, default=0)
    memories_activated = Column(Integer, nullable=False, default=0)
    memories_deprecated = Column(Integer, nullable=False, default=0)
    memories_archived = Column(Integer, nullable=False, default=0)
    contradictions_found = Column(Integer, nullable=False, default=0)
    contradictions_resolved = Column(Integer, nullable=False, default=0)
    playbooks_updated = Column(Integer, nullable=False, default=0)

    diff_json = Column(Text, nullable=True)   # Full structured diff for UI display
    error = Column(Text, nullable=True)       # If consolidation failed partway

    diffs = relationship("MemoryDiff", back_populates="run", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<ConsolidationRun id={self.id[:8]} deprecated={self.memories_deprecated}>"


class MemoryDiff(Base):
    """
    A single change record within a ConsolidationRun.
    The UI renders these as a live diff panel.
    """
    __tablename__ = "memory_diffs"

    id = Column(String(36), primary_key=True, default=new_id)
    run_id = Column(String(36), ForeignKey("consolidation_runs.id"), nullable=False, index=True)
    memory_id = Column(String(36), ForeignKey("memories.id"), nullable=False)

    action = Column(String(30), nullable=False)     # activated | deprecated | archived | updated | merged
    reason = Column(Text, nullable=False)
    before_content = Column(Text, nullable=True)
    after_content = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)

    run = relationship("ConsolidationRun", back_populates="diffs")

    def __repr__(self) -> str:
        return f"<MemoryDiff action={self.action} memory={self.memory_id[:8]}>"


class MemoryTag(Base):
    """Entity and keyword tags on memories — used for BM25 and entity-overlap retrieval."""
    __tablename__ = "memory_tags"

    id = Column(String(36), primary_key=True, default=new_id)
    memory_id = Column(String(36), ForeignKey("memories.id"), nullable=False, index=True)
    tag = Column(String(255), nullable=False, index=True)
    tag_type = Column(String(50), nullable=False, default="keyword")  # keyword | entity | topic

    memory = relationship("Memory", back_populates="tags")

    __table_args__ = (
        UniqueConstraint("memory_id", "tag", name="uq_memory_tag"),
    )


class Session(Base):
    """
    A single interaction session — groups memories and tracks per-session metrics
    for the learning curve eval.
    """
    __tablename__ = "sessions"

    id = Column(String(36), primary_key=True, default=new_id)
    playbook_id = Column(String(36), ForeignKey("playbooks.id"), nullable=True)

    started_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)
    ended_at = Column(DateTime(timezone=True), nullable=True)

    # Eval metrics — recorded at session end
    task_completed = Column(Boolean, nullable=True)
    steps_taken = Column(Integer, nullable=True)
    tokens_used = Column(Integer, nullable=True)
    playbook_used = Column(Boolean, nullable=False, default=False)
    contradictions_encountered = Column(Integer, nullable=False, default=0)

    messages_json = Column(Text, nullable=True)  # Full message history (JSON)

    def end(self, task_completed: bool, steps_taken: int, tokens_used: int) -> None:
        self.ended_at = utcnow()
        self.task_completed = task_completed
        self.steps_taken = steps_taken
        self.tokens_used = tokens_used

    def __repr__(self) -> str:
        return f"<Session id={self.id[:8]} completed={self.task_completed}>"


def init_db(db_path: str = "engram.db") -> sessionmaker:
    """Initialize the database and return a session factory."""
    engine = create_engine(
        f"sqlite+aiosqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)
