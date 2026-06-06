"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import {
  api,
  streamChat,
  type Memory,
  type PlaybookSummary,
  type ConsolidationRun,
  type SessionMetric,
  type SummaryMetrics,
} from "@/lib/api";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts";
import {
  Brain,
  Zap,
  GitBranch,
  RotateCcw,
  ChevronRight,
  Activity,
  Layers,
  AlertTriangle,
  CheckCircle,
  Clock,
  Cpu,
  Play,
  Square,
} from "lucide-react";
import clsx from "clsx";

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  streaming?: boolean;
}

interface SessionState {
  id: string;
  active: boolean;
  playbookUsed: boolean;
  playbookName?: string;
}

// ─────────────────────────────────────────────────────────────────────────────
// Sub-components
// ─────────────────────────────────────────────────────────────────────────────

function MemoryBadge({ type }: { type: string }) {
  const colors: Record<string, string> = {
    episodic: "bg-blue-500/20 text-blue-400 border-blue-500/30",
    semantic: "bg-emerald-500/20 text-emerald-400 border-emerald-500/30",
    procedural: "bg-violet-500/20 text-violet-400 border-violet-500/30",
  };
  return (
    <span
      className={clsx(
        "text-[10px] font-mono px-1.5 py-0.5 rounded border uppercase tracking-wider",
        colors[type] || "bg-gray-500/20 text-gray-400 border-gray-500/30"
      )}
    >
      {type}
    </span>
  );
}

function StatusDot({ status }: { status: string }) {
  const colors: Record<string, string> = {
    active: "bg-green-400",
    pending: "bg-amber-400 animate-pulse",
    deprecated: "bg-red-400",
    archived: "bg-gray-500",
  };
  return (
    <span
      className={clsx("inline-block w-1.5 h-1.5 rounded-full", colors[status] || "bg-gray-500")}
    />
  );
}

function MetricCard({
  label,
  value,
  sub,
  icon: Icon,
  accent = false,
}: {
  label: string;
  value: string | number;
  sub?: string;
  icon: any;
  accent?: boolean;
}) {
  return (
    <div
      className={clsx(
        "rounded-xl border p-4 flex flex-col gap-2",
        accent
          ? "bg-[#d4f000]/5 border-[#d4f000]/20"
          : "bg-[#13131a] border-[#2a2a3a]"
      )}
    >
      <div className="flex items-center justify-between">
        <span className="text-xs text-[#8888a0] font-syne uppercase tracking-widest">
          {label}
        </span>
        <Icon
          size={14}
          className={accent ? "text-[#d4f000]" : "text-[#555567]"}
        />
      </div>
      <div
        className={clsx(
          "text-2xl font-syne font-bold",
          accent ? "text-[#d4f000]" : "text-[#f0f0f4]"
        )}
      >
        {value}
      </div>
      {sub && <div className="text-xs text-[#555567]">{sub}</div>}
    </div>
  );
}

function DiffEntry({ entry }: { entry: any }) {
  const actionColors: Record<string, string> = {
    activated: "text-green-400",
    deprecated: "text-red-400",
    archived: "text-gray-400",
    merged: "text-violet-400",
    stale_step_flagged: "text-amber-400",
  };

  const actionIcons: Record<string, string> = {
    activated: "✓",
    deprecated: "✕",
    archived: "○",
    merged: "⊕",
    stale_step_flagged: "⚠",
  };

  return (
    <div className="flex gap-3 py-2 border-b border-[#1a1a24] last:border-0">
      <span
        className={clsx(
          "font-mono text-sm w-4 flex-shrink-0 mt-0.5",
          actionColors[entry.action] || "text-gray-400"
        )}
      >
        {actionIcons[entry.action] || "·"}
      </span>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 flex-wrap">
          <span
            className={clsx(
              "text-xs font-mono uppercase",
              actionColors[entry.action] || "text-gray-400"
            )}
          >
            {entry.action.replace(/_/g, " ")}
          </span>
          {entry.memory_type && <MemoryBadge type={entry.memory_type} />}
        </div>
        <p className="text-xs text-[#8888a0] mt-0.5 truncate">{entry.reason}</p>
        {entry.content_preview && (
          <p className="text-xs text-[#555567] font-mono mt-0.5 truncate">
            "{entry.content_preview}"
          </p>
        )}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Main Page
// ─────────────────────────────────────────────────────────────────────────────

export default function EngramDashboard() {
  // ── Chat state ──────────────────────────────────────────────────────────
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [session, setSession] = useState<SessionState | null>(null);
  const chatEndRef = useRef<HTMLDivElement>(null);

  // ── Memory & metrics state ──────────────────────────────────────────────
  const [memories, setMemories] = useState<Memory[]>([]);
  const [playbooks, setPlaybooks] = useState<PlaybookSummary[]>([]);
  const [sessionMetrics, setSessionMetrics] = useState<SessionMetric[]>([]);
  const [summary, setSummary] = useState<SummaryMetrics | null>(null);
  const [consolidationRuns, setConsolidationRuns] = useState<ConsolidationRun[]>([]);
  const [lastDiff, setLastDiff] = useState<any[]>([]);

  // ── UI state ────────────────────────────────────────────────────────────
  const [activeTab, setActiveTab] = useState<"chat" | "memories" | "metrics" | "consolidation">(
    "chat"
  );
  const [consolidating, setConsolidating] = useState(false);
  const [playbookInput, setPlaybookInput] = useState("");
  const [endingSession, setEndingSession] = useState(false);
  const [taskSummary, setTaskSummary] = useState("");

  // ── Auto-scroll chat ────────────────────────────────────────────────────
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // ── Polling for memory updates ──────────────────────────────────────────
  const refreshData = useCallback(async () => {
    try {
      const [mems, pbs, metrics, sum, hist] = await Promise.all([
        api.getAllMemories(),
        api.listPlaybooks(),
        api.getSessionMetrics(),
        api.getSummaryMetrics(),
        api.getConsolidationHistory(),
      ]);
      setMemories(mems.memories);
      setPlaybooks(pbs.playbooks);
      setSessionMetrics(metrics.sessions);
      setSummary(sum);
      setConsolidationRuns(hist.runs);
      if (hist.runs[0]?.diff_json?.length) {
        setLastDiff(hist.runs[0].diff_json);
      }
    } catch {
      // Backend not ready yet — silent
    }
  }, []);

  useEffect(() => {
    refreshData();
    const interval = setInterval(refreshData, 8000);
    return () => clearInterval(interval);
  }, [refreshData]);

  // ── Session management ──────────────────────────────────────────────────
  async function startSession() {
    try {
      const res = await api.startSession(playbookInput || undefined);
      setSession({
        id: res.session_id,
        active: true,
        playbookUsed: !!res.playbook,
        playbookName: playbookInput || undefined,
      });
      setMessages([
        {
          role: "assistant",
          content: res.playbook
            ? `Session started. I found an existing playbook for **${res.playbook.name}** (v${res.playbook.version}, ${res.playbook.steps?.length} steps). I'll follow it and track performance. Loaded ${res.initial_memories.length} relevant memories.`
            : `Session started. No existing playbook found${playbookInput ? ` for "${playbookInput}"` : ""}. I'll observe this workflow and create one when we're done. Loaded ${res.initial_memories.length} relevant memories.`,
        },
      ]);
    } catch (e: any) {
      alert(`Failed to start session: ${e.message}`);
    }
  }

  async function endSession() {
    if (!session || !taskSummary) return;
    setEndingSession(true);
    try {
      const res = await api.endSession(
        session.id,
        true,
        taskSummary,
        session.playbookUsed
      );
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: `Session ended. Reflection complete — ${res.steps_recorded} steps recorded. Playbook written/updated. Consolidation triggered.`,
        },
      ]);
      setSession(null);
      setTaskSummary("");
      await refreshData();
    } catch (e: any) {
      alert(`Failed to end session: ${e.message}`);
    } finally {
      setEndingSession(false);
    }
  }

  // ── Chat ────────────────────────────────────────────────────────────────
  async function sendMessage() {
    if (!session || !input.trim() || sending) return;
    const userMsg = input.trim();
    setInput("");
    setSending(true);

    setMessages((prev) => [
      ...prev,
      { role: "user", content: userMsg },
      { role: "assistant", content: "", streaming: true },
    ]);

    try {
      let finalContent = "";
      for await (const chunk of streamChat(session.id, userMsg)) {
        if (chunk.type === "chunk" || chunk.type === "done") {
          finalContent = chunk.content || "";
          setMessages((prev) => {
            const updated = [...prev];
            updated[updated.length - 1] = {
              role: "assistant",
              content: finalContent,
              streaming: chunk.type === "chunk",
            };
            return updated;
          });
        }
      }
    } catch (e: any) {
      setMessages((prev) => {
        const updated = [...prev];
        updated[updated.length - 1] = {
          role: "assistant",
          content: `Error: ${e.message}`,
        };
        return updated;
      });
    } finally {
      setSending(false);
      // Refresh memories after each turn
      const mems = await api.getAllMemories();
      setMemories(mems.memories);
    }
  }

  // ── Consolidation trigger ───────────────────────────────────────────────
  async function triggerConsolidation() {
    setConsolidating(true);
    try {
      await api.triggerConsolidation();
      await refreshData();
    } catch (e: any) {
      alert(`Consolidation failed: ${e.message}`);
    } finally {
      setConsolidating(false);
    }
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Render
  // ─────────────────────────────────────────────────────────────────────────

  return (
    <div className="min-h-screen bg-[#0d0d0f] font-syne text-[#f0f0f4] flex flex-col">
      {/* ── Header ── */}
      <header className="border-b border-[#2a2a3a] px-6 py-3 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-[#d4f000]/10 border border-[#d4f000]/20 flex items-center justify-center">
            <Brain size={16} className="text-[#d4f000]" />
          </div>
          <div>
            <h1 className="text-sm font-bold tracking-tight">ENGRAM</h1>
            <p className="text-[10px] text-[#555567] uppercase tracking-widest">
              Memory Agent · Track 1
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {/* Session status */}
          {session ? (
            <div className="flex items-center gap-2 px-3 py-1.5 bg-green-500/10 border border-green-500/20 rounded-lg">
              <span className="w-1.5 h-1.5 bg-green-400 rounded-full animate-pulse" />
              <span className="text-xs text-green-400 font-mono">
                SESSION {session.id.slice(0, 8).toUpperCase()}
              </span>
            </div>
          ) : (
            <div className="flex items-center gap-2 px-3 py-1.5 bg-[#1a1a24] border border-[#2a2a3a] rounded-lg">
              <span className="w-1.5 h-1.5 bg-[#555567] rounded-full" />
              <span className="text-xs text-[#555567] font-mono">NO SESSION</span>
            </div>
          )}

          {/* Memory count */}
          {summary && (
            <div className="px-3 py-1.5 bg-[#1a1a24] border border-[#2a2a3a] rounded-lg">
              <span className="text-xs text-[#8888a0] font-mono">
                {summary.active_memories} memories
              </span>
            </div>
          )}
        </div>
      </header>

      {/* ── Tab nav ── */}
      <nav className="border-b border-[#2a2a3a] px-6 flex gap-1 pt-1">
        {(
          [
            { id: "chat", label: "Chat", icon: Cpu },
            { id: "memories", label: "Memory Store", icon: Layers },
            { id: "metrics", label: "Learning Curve", icon: Activity },
            { id: "consolidation", label: "Forgetting Engine", icon: RotateCcw },
          ] as const
        ).map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            onClick={() => setActiveTab(id)}
            className={clsx(
              "flex items-center gap-1.5 px-4 py-2 text-xs uppercase tracking-widest transition-all border-b-2 -mb-px",
              activeTab === id
                ? "border-[#d4f000] text-[#d4f000]"
                : "border-transparent text-[#555567] hover:text-[#8888a0]"
            )}
          >
            <Icon size={12} />
            {label}
          </button>
        ))}
      </nav>

      {/* ── Main content ── */}
      <main className="flex-1 flex overflow-hidden">
        {/* ── CHAT TAB ── */}
        {activeTab === "chat" && (
          <div className="flex-1 flex flex-col">
            {/* Session controls */}
            <div className="border-b border-[#2a2a3a] bg-[#13131a] px-4 py-3 flex items-center gap-3">
              <input
                value={playbookInput}
                onChange={(e) => setPlaybookInput(e.target.value)}
                placeholder="Playbook name (optional — loads existing workflow)"
                className="flex-1 bg-[#1a1a24] border border-[#2a2a3a] rounded-lg px-3 py-1.5 text-xs text-[#f0f0f4] placeholder:text-[#555567] focus:outline-none focus:border-[#d4f000]/50 font-mono"
                disabled={!!session}
              />
              {!session ? (
                <button
                  onClick={startSession}
                  className="flex items-center gap-1.5 px-4 py-1.5 bg-[#d4f000] text-[#0d0d0f] rounded-lg text-xs font-bold hover:bg-[#c4e000] transition-colors"
                >
                  <Play size={11} /> Start Session
                </button>
              ) : (
                <button
                  onClick={() => {
                    const summary = prompt("Brief session summary (what was accomplished)?");
                    if (summary) {
                      setTaskSummary(summary);
                    }
                  }}
                  className="flex items-center gap-1.5 px-4 py-1.5 bg-red-500/20 border border-red-500/30 text-red-400 rounded-lg text-xs font-bold hover:bg-red-500/30 transition-colors"
                  disabled={endingSession}
                >
                  <Square size={11} /> End Session
                </button>
              )}
            </div>

            {/* End session confirmation */}
            {taskSummary && session && (
              <div className="border-b border-amber-500/20 bg-amber-500/5 px-4 py-2 flex items-center gap-3">
                <AlertTriangle size={12} className="text-amber-400 flex-shrink-0" />
                <span className="text-xs text-amber-400 flex-1">
                  Ending session: "{taskSummary}" — this will trigger reflection and playbook write.
                </span>
                <button
                  onClick={endSession}
                  disabled={endingSession}
                  className="text-xs px-3 py-1 bg-amber-500/20 border border-amber-500/30 text-amber-400 rounded hover:bg-amber-500/30 transition-colors disabled:opacity-50"
                >
                  {endingSession ? "Ending..." : "Confirm"}
                </button>
                <button
                  onClick={() => setTaskSummary("")}
                  className="text-xs text-[#555567] hover:text-[#8888a0]"
                >
                  Cancel
                </button>
              </div>
            )}

            {/* Messages */}
            <div className="flex-1 overflow-y-auto px-4 py-4 flex flex-col gap-4">
              {messages.length === 0 && (
                <div className="flex-1 flex flex-col items-center justify-center gap-4 text-center py-16">
                  <div className="w-16 h-16 rounded-2xl bg-[#d4f000]/5 border border-[#d4f000]/10 flex items-center justify-center">
                    <Brain size={28} className="text-[#d4f000]/40" />
                  </div>
                  <div>
                    <h2 className="text-lg font-bold text-[#f0f0f4] mb-1">
                      No active session
                    </h2>
                    <p className="text-sm text-[#555567] max-w-sm">
                      Start a session to begin. Engram will watch what you do,
                      write a playbook, and get faster every run.
                    </p>
                  </div>
                </div>
              )}

              {messages.map((msg, i) => (
                <div
                  key={i}
                  className={clsx(
                    "flex gap-3 max-w-3xl",
                    msg.role === "user" && "ml-auto flex-row-reverse"
                  )}
                >
                  <div
                    className={clsx(
                      "w-7 h-7 rounded-lg flex items-center justify-center flex-shrink-0 text-[10px] font-bold",
                      msg.role === "user"
                        ? "bg-[#2a2a3a] text-[#8888a0]"
                        : "bg-[#d4f000]/10 border border-[#d4f000]/20 text-[#d4f000]"
                    )}
                  >
                    {msg.role === "user" ? "U" : "E"}
                  </div>
                  <div
                    className={clsx(
                      "rounded-xl px-4 py-3 text-sm leading-relaxed max-w-[calc(100%-3rem)]",
                      msg.role === "user"
                        ? "bg-[#1a1a24] border border-[#2a2a3a] text-[#f0f0f4]"
                        : "bg-[#13131a] border border-[#2a2a3a] text-[#e0e0ec]",
                      msg.streaming && "streaming-cursor"
                    )}
                  >
                    {msg.content || (msg.streaming ? "" : "…")}
                  </div>
                </div>
              ))}
              <div ref={chatEndRef} />
            </div>

            {/* Input */}
            <div className="border-t border-[#2a2a3a] bg-[#13131a] px-4 py-3">
              <div className="flex gap-2">
                <input
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && sendMessage()}
                  placeholder={session ? "Message Engram..." : "Start a session first"}
                  disabled={!session || sending}
                  className="flex-1 bg-[#1a1a24] border border-[#2a2a3a] rounded-lg px-4 py-2.5 text-sm text-[#f0f0f4] placeholder:text-[#555567] focus:outline-none focus:border-[#d4f000]/40 disabled:opacity-40 font-syne"
                />
                <button
                  onClick={sendMessage}
                  disabled={!session || !input.trim() || sending}
                  className="px-4 py-2.5 bg-[#d4f000] text-[#0d0d0f] rounded-lg text-sm font-bold hover:bg-[#c4e000] transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
                >
                  <ChevronRight size={16} />
                </button>
              </div>
            </div>
          </div>
        )}

        {/* ── MEMORY STORE TAB ── */}
        {activeTab === "memories" && (
          <div className="flex-1 overflow-y-auto p-6">
            <div className="max-w-4xl mx-auto">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-sm font-bold uppercase tracking-widest text-[#8888a0]">
                  Memory Store
                </h2>
                <div className="flex gap-2">
                  {["episodic", "semantic", "procedural"].map((t) => (
                    <div key={t} className="flex items-center gap-1.5">
                      <span
                        className={clsx(
                          "w-2 h-2 rounded-full",
                          t === "episodic"
                            ? "bg-blue-400"
                            : t === "semantic"
                            ? "bg-emerald-400"
                            : "bg-violet-400"
                        )}
                      />
                      <span className="text-[10px] text-[#555567] uppercase">
                        {t}
                      </span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Memory list */}
              <div className="flex flex-col gap-2">
                {memories.length === 0 && (
                  <div className="text-center py-12 text-[#555567] text-sm">
                    No memories yet. Start a session.
                  </div>
                )}
                {memories.map((m) => (
                  <div
                    key={m.id}
                    className={clsx(
                      "rounded-xl border p-4 transition-all",
                      m.status === "active"
                        ? "bg-[#13131a] border-[#2a2a3a]"
                        : m.status === "deprecated"
                        ? "bg-red-500/5 border-red-500/20 opacity-60"
                        : "bg-[#0f0f14] border-[#1f1f2a] opacity-40"
                    )}
                  >
                    <div className="flex items-start gap-3">
                      <div className="flex flex-col items-center gap-1 mt-0.5">
                        <StatusDot status={m.status} />
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-1 flex-wrap">
                          <MemoryBadge type={m.memory_type} />
                          <span className="text-[10px] font-mono text-[#555567]">
                            {m.id.slice(0, 8)}
                          </span>
                          <span className="text-[10px] text-[#555567]">
                            importance {(m.importance_score * 100).toFixed(0)}%
                          </span>
                          <span className="text-[10px] text-[#555567]">
                            accessed {m.access_count}×
                          </span>
                        </div>
                        <p className="text-sm text-[#e0e0ec] leading-relaxed">
                          {m.content}
                        </p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>

              {/* Playbooks */}
              {playbooks.length > 0 && (
                <div className="mt-8">
                  <h3 className="text-xs font-bold uppercase tracking-widest text-[#8888a0] mb-4">
                    Playbooks
                  </h3>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                    {playbooks.map((pb) => (
                      <div
                        key={pb.id}
                        className="rounded-xl border border-[#2a2a3a] bg-[#13131a] p-4"
                      >
                        <div className="flex items-start justify-between gap-2">
                          <div>
                            <div className="flex items-center gap-2">
                              <GitBranch size={12} className="text-violet-400" />
                              <span className="text-sm font-bold text-[#f0f0f4]">
                                {pb.name}
                              </span>
                              <span className="text-[10px] font-mono text-[#555567]">
                                v{pb.version}
                              </span>
                            </div>
                            <p className="text-xs text-[#555567] mt-1 line-clamp-2">
                              {pb.description}
                            </p>
                          </div>
                        </div>
                        <div className="flex gap-3 mt-3 pt-3 border-t border-[#2a2a3a]">
                          <span className="text-xs text-[#555567]">
                            {pb.session_count} runs
                          </span>
                          {pb.last_run_at && (
                            <span className="text-xs text-[#555567]">
                              Last:{" "}
                              {new Date(pb.last_run_at).toLocaleDateString()}
                            </span>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* ── METRICS TAB ── */}
        {activeTab === "metrics" && (
          <div className="flex-1 overflow-y-auto p-6">
            <div className="max-w-4xl mx-auto">
              <h2 className="text-sm font-bold uppercase tracking-widest text-[#8888a0] mb-6">
                Learning Curve
              </h2>

              {/* Summary cards */}
              {summary && (
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-8">
                  <MetricCard
                    label="Sessions"
                    value={summary.total_sessions}
                    sub={`${summary.success_rate}% success`}
                    icon={Activity}
                  />
                  <MetricCard
                    label="Avg Steps"
                    value={summary.avg_steps}
                    sub="per session"
                    icon={GitBranch}
                  />
                  <MetricCard
                    label="Avg Tokens"
                    value={summary.avg_tokens.toLocaleString()}
                    sub="per session"
                    icon={Zap}
                  />
                  <MetricCard
                    label="Token Efficiency Gain"
                    value={
                      summary.token_efficiency_improvement_pct !== null
                        ? `${summary.token_efficiency_improvement_pct}%`
                        : "—"
                    }
                    sub="vs first sessions"
                    icon={CheckCircle}
                    accent={!!summary.token_efficiency_improvement_pct}
                  />
                </div>
              )}

              {/* Learning curve chart */}
              {sessionMetrics.length > 1 ? (
                <div className="bg-[#13131a] border border-[#2a2a3a] rounded-xl p-6 mb-6">
                  <h3 className="text-xs uppercase tracking-widest text-[#555567] mb-4">
                    Tokens Used Per Session
                  </h3>
                  <ResponsiveContainer width="100%" height={200}>
                    <LineChart data={sessionMetrics}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#2a2a3a" />
                      <XAxis
                        dataKey="session_number"
                        tick={{ fill: "#555567", fontSize: 10 }}
                        label={{
                          value: "Session",
                          position: "insideBottom",
                          fill: "#555567",
                          fontSize: 10,
                        }}
                      />
                      <YAxis tick={{ fill: "#555567", fontSize: 10 }} />
                      <Tooltip
                        contentStyle={{
                          background: "#13131a",
                          border: "1px solid #2a2a3a",
                          borderRadius: 8,
                          fontSize: 11,
                        }}
                      />
                      <Line
                        type="monotone"
                        dataKey="tokens_used"
                        stroke="#d4f000"
                        strokeWidth={2}
                        dot={{ fill: "#d4f000", r: 3 }}
                        name="Tokens Used"
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              ) : (
                <div className="bg-[#13131a] border border-[#2a2a3a] rounded-xl p-8 text-center text-[#555567] text-sm mb-6">
                  Complete 2+ sessions to see the learning curve.
                </div>
              )}

              {/* Steps chart */}
              {sessionMetrics.length > 1 && (
                <div className="bg-[#13131a] border border-[#2a2a3a] rounded-xl p-6">
                  <h3 className="text-xs uppercase tracking-widest text-[#555567] mb-4">
                    Steps Taken Per Session
                  </h3>
                  <ResponsiveContainer width="100%" height={160}>
                    <LineChart data={sessionMetrics}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#2a2a3a" />
                      <XAxis
                        dataKey="session_number"
                        tick={{ fill: "#555567", fontSize: 10 }}
                      />
                      <YAxis tick={{ fill: "#555567", fontSize: 10 }} />
                      <Tooltip
                        contentStyle={{
                          background: "#13131a",
                          border: "1px solid #2a2a3a",
                          borderRadius: 8,
                          fontSize: 11,
                        }}
                      />
                      <Line
                        type="monotone"
                        dataKey="steps_taken"
                        stroke="#3b82f6"
                        strokeWidth={2}
                        dot={{ fill: "#3b82f6", r: 3 }}
                        name="Steps Taken"
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              )}
            </div>
          </div>
        )}

        {/* ── CONSOLIDATION TAB ── */}
        {activeTab === "consolidation" && (
          <div className="flex-1 overflow-y-auto p-6">
            <div className="max-w-4xl mx-auto">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-sm font-bold uppercase tracking-widest text-[#8888a0]">
                  Forgetting Engine
                </h2>
                <button
                  onClick={triggerConsolidation}
                  disabled={consolidating}
                  className={clsx(
                    "flex items-center gap-1.5 px-4 py-2 rounded-lg text-xs font-bold transition-all border",
                    consolidating
                      ? "bg-[#1a1a24] border-[#2a2a3a] text-[#555567] cursor-wait"
                      : "bg-[#d4f000]/10 border-[#d4f000]/20 text-[#d4f000] hover:bg-[#d4f000]/15"
                  )}
                >
                  <RotateCcw
                    size={12}
                    className={consolidating ? "animate-spin" : ""}
                  />
                  {consolidating ? "Running..." : "Run Now"}
                </button>
              </div>

              {/* Latest diff */}
              {lastDiff.length > 0 && (
                <div className="bg-[#13131a] border border-[#2a2a3a] rounded-xl p-5 mb-6">
                  <h3 className="text-xs uppercase tracking-widest text-[#555567] mb-3">
                    Latest Memory Diff
                  </h3>
                  <div>
                    {lastDiff.map((entry, i) => (
                      <DiffEntry key={i} entry={entry} />
                    ))}
                  </div>
                </div>
              )}

              {/* Run history */}
              <div className="flex flex-col gap-3">
                {consolidationRuns.map((run) => (
                  <div
                    key={run.id}
                    className="bg-[#13131a] border border-[#2a2a3a] rounded-xl p-4"
                  >
                    <div className="flex items-start justify-between gap-4">
                      <div>
                        <div className="flex items-center gap-2 mb-1">
                          <span className="text-xs font-mono text-[#555567]">
                            {run.id.slice(0, 8)}
                          </span>
                          <span className="text-[10px] px-1.5 py-0.5 bg-[#1a1a24] border border-[#2a2a3a] rounded font-mono text-[#555567] uppercase">
                            {run.triggered_by}
                          </span>
                          {run.error && (
                            <span className="text-[10px] text-red-400">
                              ERROR
                            </span>
                          )}
                        </div>
                        <div className="flex gap-4 flex-wrap">
                          <span className="text-xs text-[#555567]">
                            <span className="text-green-400">
                              +{run.memories_activated}
                            </span>{" "}
                            activated
                          </span>
                          <span className="text-xs text-[#555567]">
                            <span className="text-red-400">
                              -{run.memories_deprecated}
                            </span>{" "}
                            deprecated
                          </span>
                          <span className="text-xs text-[#555567]">
                            <span className="text-gray-400">
                              {run.memories_archived}
                            </span>{" "}
                            archived
                          </span>
                          <span className="text-xs text-[#555567]">
                            <span className="text-[#d4f000]">
                              {run.contradictions_resolved}
                            </span>{" "}
                            contradictions resolved
                          </span>
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="text-[10px] text-[#555567]">
                          <Clock size={8} className="inline mr-1" />
                          {new Date(run.started_at).toLocaleTimeString()}
                        </div>
                        <div className="text-[10px] text-[#555567] mt-0.5">
                          {run.memories_scanned} scanned
                        </div>
                      </div>
                    </div>
                  </div>
                ))}

                {consolidationRuns.length === 0 && (
                  <div className="text-center py-12 text-[#555567] text-sm">
                    No consolidation runs yet. The engine runs automatically every 30 minutes,
                    or trigger it manually above.
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
