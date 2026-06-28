"use client";

import { useState, useEffect, useRef, useCallback, lazy, Suspense } from "react";
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
  Legend,
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
  Database,
  Sparkles,
} from "lucide-react";
import clsx from "clsx";
import ReactMarkdown from "react-markdown";
import GraphLegend from "@/components/GraphLegend";

// Lazy-load D3 graph (large bundle)
const MemoryGraph = lazy(() => import("@/components/MemoryGraph"));

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

interface ToolStep {
  callIndex: number;
  name: string;
  args?: string;
  status: "running" | "done" | "error";
}

interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  streaming?: boolean;
  tools?: ToolStep[];
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
    <span className={clsx("text-[10px] font-mono px-1.5 py-0.5 rounded border uppercase tracking-wider", colors[type] || "bg-gray-500/20 text-gray-400 border-gray-500/30")}>
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
  return <span className={clsx("inline-block w-1.5 h-1.5 rounded-full", colors[status] || "bg-gray-500")} />;
}

function MetricCard({ label, value, sub, icon: Icon, accent = false }: { label: string; value: string | number; sub?: string; icon: any; accent?: boolean }) {
  return (
    <div className={clsx("rounded-xl border p-4 flex flex-col gap-2", accent ? "bg-[#d4f000]/5 border-[#d4f000]/20" : "bg-[#13131a] border-[#2a2a3a]")}>
      <div className="flex items-center justify-between">
        <span className="text-xs text-[#8888a0] font-syne uppercase tracking-widest">{label}</span>
        <Icon size={14} className={accent ? "text-[#d4f000]" : "text-[#555567]"} />
      </div>
      <div className={clsx("text-2xl font-syne font-bold", accent ? "text-[#d4f000]" : "text-[#f0f0f4]")}>{value}</div>
      {sub && <div className="text-xs text-[#555567]">{sub}</div>}
    </div>
  );
}

function DiffEntry({ entry }: { entry: any }) {
  const actionColors: Record<string, string> = {
    activated: "text-green-400", deprecated: "text-red-400",
    deduplicated: "text-blue-400",
    archived: "text-gray-400", merged: "text-violet-400",
    stale_step_flagged: "text-amber-400",
  };
  const actionIcons: Record<string, string> = {
    activated: "✓", deprecated: "✕", deduplicated: "⧉", archived: "○", merged: "⊕", stale_step_flagged: "⚠",
  };
  const actionLabels: Record<string, string> = {
    deprecated: "belief revised", deduplicated: "duplicate merged",
  };
  // Treat a duplicate-flagged deprecation as its own "deduplicated" kind.
  const reason: string = entry.reason || "";
  const isDuplicate = /merged into|redundant|duplicate/i.test(reason);
  const kind = entry.action === "deprecated" && isDuplicate ? "deduplicated" : entry.action;
  return (
    <div className="flex gap-3 py-2 border-b border-[#1a1a24] last:border-0">
      <span className={clsx("font-mono text-sm w-4 flex-shrink-0 mt-0.5", actionColors[kind] || "text-gray-400")}>
        {actionIcons[kind] || "·"}
      </span>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 flex-wrap">
          <span className={clsx("text-xs font-mono uppercase", actionColors[kind] || "text-gray-400")}>
            {(actionLabels[kind] || kind).replace(/_/g, " ")}
          </span>
          {entry.memory_type && <MemoryBadge type={entry.memory_type} />}
        </div>
        <p className="text-xs text-[#8888a0] mt-0.5 truncate">{entry.reason}</p>
        {entry.content_preview && (
          <p className="text-xs text-[#555567] font-mono mt-0.5 truncate">"{entry.content_preview}"</p>
        )}
      </div>
    </div>
  );
}

const TOOL_LABELS: Record<string, string> = {
  list_playbooks: "List playbooks",
  get_playbook: "Get playbook",
  write_playbook: "Write playbook",
  query_memory: "Query memory",
  write_memory: "Write memory",
  update_memory: "Update memory",
  deprecate_memory: "Deprecate memory",
  record_step_result: "Record step result",
};

function ToolTimeline({ tools }: { tools: ToolStep[] }) {
  return (
    <div className="flex flex-col gap-1 rounded-lg border border-[#2a2a3a] bg-[#0f0f14] px-3 py-2">
      <span className="text-[10px] uppercase tracking-widest text-[#555567] mb-0.5">Tool calls</span>
      {tools.map((t, i) => (
        <div key={i} className="flex items-center gap-2 text-xs font-mono">
          <span
            className={clsx(
              "flex items-center justify-center w-3.5 h-3.5",
              t.status === "running" ? "text-[#d4f000]" : t.status === "error" ? "text-red-400" : "text-green-400"
            )}
          >
            {t.status === "running" ? (
              <RotateCcw size={11} className="animate-spin" />
            ) : t.status === "error" ? (
              <AlertTriangle size={11} />
            ) : (
              <CheckCircle size={11} />
            )}
          </span>
          <span className={clsx(t.status === "error" ? "text-red-400/80" : "text-[#8888a0]")}>
            {t.status === "running" && <span className="text-[#d4f000]">calling </span>}
            {TOOL_LABELS[t.name] || t.name.replace(/_/g, " ")}
            {t.status === "running" && "…"}
          </span>
          {t.status === "error" && <span className="text-red-400/60 text-[10px]">failed</span>}
        </div>
      ))}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Main Page
// ─────────────────────────────────────────────────────────────────────────────

export default function EngramDashboard() {
  // ── State ──────────────────────────────────────────────────────────────
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [session, setSession] = useState<SessionState | null>(null);
  const chatEndRef = useRef<HTMLDivElement>(null);

  const [memories, setMemories] = useState<Memory[]>([]);
  const [playbooks, setPlaybooks] = useState<PlaybookSummary[]>([]);
  const [sessionMetrics, setSessionMetrics] = useState<SessionMetric[]>([]);
  const [summary, setSummary] = useState<SummaryMetrics | null>(null);
  const [consolidationRuns, setConsolidationRuns] = useState<ConsolidationRun[]>([]);
  const [lastDiff, setLastDiff] = useState<any[]>([]);

  const [activeTab, setActiveTab] = useState<"chat" | "memories" | "metrics" | "consolidation">("chat");
  const [consolidating, setConsolidating] = useState(false);
  const [playbookInput, setPlaybookInput] = useState("");
  const [taskSummary, setTaskSummary] = useState("");
  const [endingSession, setEndingSession] = useState(false);
  const [seeding, setSeeding] = useState(false);
  const [graphView, setGraphView] = useState<"graph" | "list">("graph");

  // List-view filters + pagination
  const [memStatusFilter, setMemStatusFilter] = useState<"all" | "active" | "pending" | "deprecated" | "archived">("all");
  const [memTypeFilter, setMemTypeFilter] = useState<"all" | "episodic" | "semantic" | "procedural">("all");
  const [memPage, setMemPage] = useState(0);
  const MEM_PAGE_SIZE = 10;
  const [selectedMemory, setSelectedMemory] = useState<Memory | null>(null);
  const [expandedMems, setExpandedMems] = useState<Set<string>>(new Set());
  const toggleExpand = (id: string) =>
    setExpandedMems(prev => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });

  useEffect(() => { chatEndRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages]);

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
      if (hist.runs[0]?.diff_json?.length) setLastDiff(hist.runs[0].diff_json);
    } catch { /* backend not ready */ }
  }, []);

  useEffect(() => {
    refreshData();
    const i = setInterval(refreshData, 8000);
    return () => clearInterval(i);
  }, [refreshData]);

  // ── Session ────────────────────────────────────────────────────────────
  async function startSession() {
    try {
      const res = await api.startSession(playbookInput || undefined);
      setSession({ id: res.session_id, active: true, playbookUsed: !!res.playbook, playbookName: playbookInput || undefined });
      setMessages([{
        role: "assistant",
        content: res.playbook
          ? `Session started. Found playbook **${res.playbook.name}** v${res.playbook.version} (${res.playbook.steps?.length} steps). Loaded ${res.initial_memories.length} relevant memories.`
          : `Session started. No playbook found${playbookInput ? ` for "${playbookInput}"` : ""}. I'll observe this workflow and create one. Loaded ${res.initial_memories.length} memories.`,
      }]);
    } catch (e: any) { alert(`Failed to start session: ${e.message}`); }
  }

  async function endSession() {
    if (!session || !taskSummary) return;
    setEndingSession(true);
    try {
      const res = await api.endSession(session.id, true, taskSummary, session.playbookUsed);
      setMessages(prev => [...prev, { role: "assistant", content: `Session ended. ${res.steps_recorded} steps recorded. Playbook written. Consolidation triggered.` }]);
      setSession(null); setTaskSummary("");
      await refreshData();
    } catch (e: any) { alert(`Failed to end session: ${e.message}`); }
    finally { setEndingSession(false); }
  }

  // ── Chat ────────────────────────────────────────────────────────────────
  async function sendMessage() {
    if (!session || !input.trim() || sending) return;
    const userMsg = input.trim();
    setInput(""); setSending(true);
    setMessages(prev => [...prev, { role: "user", content: userMsg }, { role: "assistant", content: "", streaming: true, tools: [] }]);
    try {
      let final = "";
      let tools: ToolStep[] = [];
      const update = (streaming: boolean) =>
        setMessages(prev => { const u = [...prev]; u[u.length - 1] = { role: "assistant", content: final, streaming, tools: [...tools] }; return u; });

      for await (const chunk of streamChat(session.id, userMsg)) {
        if (chunk.type === "chunk" || chunk.type === "done") {
          final = chunk.content || "";
          update(chunk.type === "chunk");
        } else if (chunk.type === "tool_call") {
          tools = [...tools, { callIndex: chunk.call_index ?? tools.length, name: chunk.tool || chunk.name || "tool", args: chunk.arguments, status: "running" }];
          update(true);
        } else if (chunk.type === "tool_result") {
          tools = tools.map(t => t.callIndex === chunk.call_index ? { ...t, status: chunk.ok === false ? "error" : "done" } : t);
          update(true);
        } else if (chunk.type === "error") {
          final = `Error: ${chunk.error || "unknown error"}`;
          update(false);
        }
      }
    } catch (e: any) {
      setMessages(prev => { const u = [...prev]; u[u.length - 1] = { ...u[u.length - 1], role: "assistant", content: `Error: ${e.message}`, streaming: false }; return u; });
    } finally {
      setSending(false);
      const mems = await api.getAllMemories();
      setMemories(mems.memories);
    }
  }

  // ── Consolidation ───────────────────────────────────────────────────────
  async function triggerConsolidation() {
    setConsolidating(true);
    try { await api.triggerConsolidation(); await refreshData(); }
    catch (e: any) { alert(`Consolidation failed: ${e.message}`); }
    finally { setConsolidating(false); }
  }

  // ── Demo seed ───────────────────────────────────────────────────────────
  async function seedDemo() {
    setSeeding(true);
    try {
      const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}/demo/seed`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ force: false }),
      });
      const data = await res.json();
      if (data.ok) {
        await refreshData();
        setActiveTab("memories");
      } else {
        alert(data.reason || "Seed failed");
      }
    } catch (e: any) { alert(`Seed failed: ${e.message}`); }
    finally { setSeeding(false); }
  }

  // ── Memory counts by status ─────────────────────────────────────────────
  const memStats = {
    active: memories.filter(m => m.status === "active").length,
    pending: memories.filter(m => m.status === "pending").length,
    deprecated: memories.filter(m => m.status === "deprecated").length,
    archived: memories.filter(m => m.status === "archived").length,
  };

  // ── List view: filtered + paginated memories ─────────────────────────────
  const filteredMemories = memories.filter(m =>
    (memStatusFilter === "all" || m.status === memStatusFilter) &&
    (memTypeFilter === "all" || m.memory_type === memTypeFilter)
  );
  const memPageCount = Math.max(1, Math.ceil(filteredMemories.length / MEM_PAGE_SIZE));
  const clampedMemPage = Math.min(memPage, memPageCount - 1);
  const pagedMemories = filteredMemories.slice(
    clampedMemPage * MEM_PAGE_SIZE,
    (clampedMemPage + 1) * MEM_PAGE_SIZE
  );
  // Reset to first page whenever the filters change.
  useEffect(() => { setMemPage(0); }, [memStatusFilter, memTypeFilter]);

  // ─────────────────────────────────────────────────────────────────────────
  // Render
  // ─────────────────────────────────────────────────────────────────────────

  return (
    <div className="h-screen overflow-hidden bg-[#0d0d0f] font-syne text-[#f0f0f4] flex flex-col">

      {/* ── Header ── */}
      <header className="flex-shrink-0 border-b border-[#2a2a3a] px-6 py-3 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-[#d4f000]/10 border border-[#d4f000]/20 flex items-center justify-center">
            <Brain size={16} className="text-[#d4f000]" />
          </div>
          <div>
            <h1 className="text-sm font-bold tracking-tight">ENGRAM</h1>
            <p className="text-[10px] text-[#555567] uppercase tracking-widest">Memory Agent · Track 1</p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {/* Demo seed button */}
          {memories.length === 0 && (
            <button
              onClick={seedDemo}
              disabled={seeding}
              className="flex items-center gap-1.5 px-3 py-1.5 bg-[#d4f000]/10 border border-[#d4f000]/20 text-[#d4f000] rounded-lg text-xs font-bold hover:bg-[#d4f000]/15 transition-colors disabled:opacity-50"
            >
              <Sparkles size={11} className={seeding ? "animate-spin" : ""} />
              {seeding ? "Seeding…" : "Load Demo"}
            </button>
          )}

          {session ? (
            <div className="flex items-center gap-2 px-3 py-1.5 bg-green-500/10 border border-green-500/20 rounded-lg">
              <span className="w-1.5 h-1.5 bg-green-400 rounded-full animate-pulse" />
              <span className="text-xs text-green-400 font-mono">SESSION {session.id.slice(0, 8).toUpperCase()}</span>
            </div>
          ) : (
            <div className="flex items-center gap-2 px-3 py-1.5 bg-[#1a1a24] border border-[#2a2a3a] rounded-lg">
              <span className="w-1.5 h-1.5 bg-[#555567] rounded-full" />
              <span className="text-xs text-[#555567] font-mono">NO SESSION</span>
            </div>
          )}

          {summary && (
            <div className="hidden sm:flex items-center gap-3 px-3 py-1.5 bg-[#1a1a24] border border-[#2a2a3a] rounded-lg">
              <span className="flex items-center gap-1"><span className="w-1.5 h-1.5 rounded-full bg-green-400" /><span className="text-[10px] text-[#8888a0] font-mono">{memStats.active}</span></span>
              <span className="flex items-center gap-1"><span className="w-1.5 h-1.5 rounded-full bg-red-400" /><span className="text-[10px] text-[#8888a0] font-mono">{memStats.deprecated}</span></span>
              <span className="flex items-center gap-1"><span className="w-1.5 h-1.5 rounded-full bg-gray-500" /><span className="text-[10px] text-[#8888a0] font-mono">{memStats.archived}</span></span>
            </div>
          )}
        </div>
      </header>

      {/* ── Tab nav ── */}
      <nav className="flex-shrink-0 border-b border-[#2a2a3a] px-6 flex gap-1 pt-1">
        {([
          { id: "chat", label: "Chat", icon: Cpu },
          { id: "memories", label: "Memory Graph", icon: Layers },
          { id: "metrics", label: "Learning Curve", icon: Activity },
          { id: "consolidation", label: "Forgetting Engine", icon: RotateCcw },
        ] as const).map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            onClick={() => setActiveTab(id)}
            className={clsx(
              "flex items-center gap-1.5 px-4 py-2 text-xs uppercase tracking-widest transition-all border-b-2 -mb-px",
              activeTab === id ? "border-[#d4f000] text-[#d4f000]" : "border-transparent text-[#555567] hover:text-[#8888a0]"
            )}
          >
            <Icon size={12} />{label}
          </button>
        ))}
      </nav>

      {/* ── Main ── */}
      <main className="flex-1 flex overflow-hidden">

        {/* ── CHAT TAB ── */}
        {activeTab === "chat" && (
          <div className="flex-1 flex flex-col">
            <div className="border-b border-[#2a2a3a] bg-[#13131a] px-4 py-3 flex items-center gap-3">
              <input
                value={playbookInput}
                onChange={e => setPlaybookInput(e.target.value)}
                placeholder="Playbook name (optional — loads existing workflow)"
                className="flex-1 bg-[#1a1a24] border border-[#2a2a3a] rounded-lg px-3 py-1.5 text-xs text-[#f0f0f4] placeholder:text-[#555567] focus:outline-none focus:border-[#d4f000]/50 font-mono"
                disabled={!!session}
              />
              {!session ? (
                <button onClick={startSession} className="flex items-center gap-1.5 px-4 py-1.5 bg-[#d4f000] text-[#0d0d0f] rounded-lg text-xs font-bold hover:bg-[#c4e000] transition-colors">
                  <Play size={11} /> Start Session
                </button>
              ) : (
                <button
                  onClick={() => { const s = prompt("Brief session summary?"); if (s) setTaskSummary(s); }}
                  className="flex items-center gap-1.5 px-4 py-1.5 bg-red-500/20 border border-red-500/30 text-red-400 rounded-lg text-xs font-bold hover:bg-red-500/30 transition-colors"
                  disabled={endingSession}
                >
                  <Square size={11} /> End Session
                </button>
              )}
            </div>

            {taskSummary && session && (
              <div className="border-b border-amber-500/20 bg-amber-500/5 px-4 py-2 flex items-center gap-3">
                <AlertTriangle size={12} className="text-amber-400 flex-shrink-0" />
                <span className="text-xs text-amber-400 flex-1">Ending: "{taskSummary}"</span>
                <button onClick={endSession} disabled={endingSession} className="text-xs px-3 py-1 bg-amber-500/20 border border-amber-500/30 text-amber-400 rounded hover:bg-amber-500/30 transition-colors disabled:opacity-50">
                  {endingSession ? "Ending…" : "Confirm"}
                </button>
                <button onClick={() => setTaskSummary("")} className="text-xs text-[#555567] hover:text-[#8888a0]">Cancel</button>
              </div>
            )}

            <div className="flex-1 overflow-y-auto px-4 py-4 flex flex-col gap-4">
              {messages.length === 0 && (
                <div className="flex-1 flex flex-col items-center justify-center gap-4 text-center py-16">
                  <div className="w-16 h-16 rounded-2xl bg-[#d4f000]/5 border border-[#d4f000]/10 flex items-center justify-center">
                    <Brain size={28} className="text-[#d4f000]/40" />
                  </div>
                  <div>
                    <h2 className="text-lg font-bold text-[#f0f0f4] mb-1">No active session</h2>
                    <p className="text-sm text-[#555567] max-w-sm">Start a session to begin. Engram watches, learns, and self-corrects.</p>
                  </div>
                  {memories.length === 0 && (
                    <button onClick={seedDemo} disabled={seeding} className="flex items-center gap-2 px-4 py-2 bg-[#1a1a24] border border-[#2a2a3a] text-[#8888a0] rounded-lg text-xs hover:border-[#d4f000]/30 hover:text-[#d4f000] transition-colors">
                      <Sparkles size={12} />{seeding ? "Loading demo…" : "Load demo data to explore the UI"}
                    </button>
                  )}
                </div>
              )}
              {messages.map((msg, i) => (
                <div key={i} className={clsx("flex gap-3 max-w-3xl", msg.role === "user" && "ml-auto flex-row-reverse")}>
                  <div className={clsx("w-7 h-7 rounded-lg flex items-center justify-center flex-shrink-0 text-[10px] font-bold", msg.role === "user" ? "bg-[#2a2a3a] text-[#8888a0]" : "bg-[#d4f000]/10 border border-[#d4f000]/20 text-[#d4f000]")}>
                    {msg.role === "user" ? "U" : "E"}
                  </div>
                  <div className={clsx("rounded-xl px-4 py-3 text-sm leading-relaxed max-w-[calc(100%-3rem)] flex flex-col gap-2", msg.role === "user" ? "bg-[#1a1a24] border border-[#2a2a3a]" : "bg-[#13131a] border border-[#2a2a3a]")}>
                    {msg.tools && msg.tools.length > 0 && <ToolTimeline tools={msg.tools} />}
                    {msg.role === "assistant" && msg.content ? (
                      <div className={clsx("prose prose-invert prose-sm max-w-none prose-p:my-1 prose-headings:my-2 prose-ul:my-1 prose-ol:my-1 prose-li:my-0 prose-pre:bg-[#1a1a24] prose-pre:border prose-pre:border-[#2a2a3a] prose-code:text-[#d4f000] prose-code:bg-[#1a1a24] prose-code:px-1 prose-code:rounded prose-code:before:content-none prose-code:after:content-none prose-a:text-[#d4f000]", msg.streaming && "streaming-cursor")}>
                        <ReactMarkdown>{msg.content}</ReactMarkdown>
                      </div>
                    ) : msg.content ? (
                      <div className="whitespace-pre-wrap">{msg.content}</div>
                    ) : null}
                    {!msg.content && msg.streaming && (!msg.tools || msg.tools.length === 0) && (
                      <div className="flex items-center gap-1.5 text-[#555567] text-xs">
                        <span className="w-1.5 h-1.5 bg-[#d4f000] rounded-full animate-pulse" />
                        <span className="w-1.5 h-1.5 bg-[#d4f000] rounded-full animate-pulse [animation-delay:150ms]" />
                        <span className="w-1.5 h-1.5 bg-[#d4f000] rounded-full animate-pulse [animation-delay:300ms]" />
                        <span className="ml-1">thinking…</span>
                      </div>
                    )}
                    {!msg.content && !msg.streaming && (!msg.tools || msg.tools.length === 0) && (
                      <span>…</span>
                    )}
                  </div>
                </div>
              ))}
              <div ref={chatEndRef} />
            </div>

            <div className="border-t border-[#2a2a3a] bg-[#13131a] px-4 py-3">
              <div className="flex gap-2">
                <input
                  value={input}
                  onChange={e => setInput(e.target.value)}
                  onKeyDown={e => e.key === "Enter" && !e.shiftKey && sendMessage()}
                  placeholder={session ? "Message Engram…" : "Start a session first"}
                  disabled={!session || sending}
                  className="flex-1 bg-[#1a1a24] border border-[#2a2a3a] rounded-lg px-4 py-2.5 text-sm text-[#f0f0f4] placeholder:text-[#555567] focus:outline-none focus:border-[#d4f000]/40 disabled:opacity-40"
                />
                <button onClick={sendMessage} disabled={!session || !input.trim() || sending} className="px-4 py-2.5 bg-[#d4f000] text-[#0d0d0f] rounded-lg text-sm font-bold hover:bg-[#c4e000] transition-colors disabled:opacity-30 disabled:cursor-not-allowed">
                  <ChevronRight size={16} />
                </button>
              </div>
            </div>
          </div>
        )}

        {/* ── MEMORY GRAPH TAB ── */}
        {activeTab === "memories" && (
          <div className="flex-1 flex flex-col overflow-hidden">
            {/* Tab header */}
            <div className="border-b border-[#2a2a3a] px-6 py-3 flex items-center justify-between flex-shrink-0">
              <div className="flex items-center gap-4">
                <h2 className="text-xs font-bold uppercase tracking-widest text-[#8888a0]">Memory Graph</h2>
                <div className="flex gap-1">
                  {(["graph", "list"] as const).map(v => (
                    <button key={v} onClick={() => setGraphView(v)} className={clsx("px-3 py-1 rounded text-[10px] uppercase tracking-widest transition-colors", graphView === v ? "bg-[#d4f000]/10 text-[#d4f000] border border-[#d4f000]/20" : "text-[#555567] hover:text-[#8888a0]")}>
                      {v}
                    </button>
                  ))}
                </div>
              </div>
              <div className="flex gap-3 text-[10px] font-mono">
                <span className="text-green-400">{memStats.active} active</span>
                <span className="text-amber-400">{memStats.pending} pending</span>
                <span className="text-red-400">{memStats.deprecated} deprecated</span>
                <span className="text-gray-400">{memStats.archived} archived</span>
              </div>
            </div>

            {/* Graph or list */}
            {graphView === "graph" ? (
              <div className="flex-1 flex flex-col overflow-hidden">
                {memories.length === 0 ? (
                  <div className="flex-1 flex flex-col items-center justify-center gap-4">
                    <Database size={36} className="text-[#2a2a3a]" />
                    <div className="text-center">
                      <p className="text-sm text-[#555567]">No memories yet.</p>
                      <button onClick={seedDemo} disabled={seeding} className="mt-3 flex items-center gap-2 mx-auto px-4 py-2 bg-[#1a1a24] border border-[#2a2a3a] text-[#8888a0] rounded-lg text-xs hover:border-[#d4f000]/30 hover:text-[#d4f000] transition-colors">
                        <Sparkles size={11} />{seeding ? "Loading…" : "Load Demo Data"}
                      </button>
                    </div>
                  </div>
                ) : (
                  <>
                    <div className="flex-1 p-4 overflow-hidden relative">
                      <Suspense fallback={<div className="w-full h-full flex items-center justify-center text-[#555567] text-sm">Loading graph…</div>}>
                        <MemoryGraph memories={memories} onSelect={setSelectedMemory} />
                      </Suspense>

                      {selectedMemory && (
                        <div className="absolute top-6 right-6 w-80 max-h-[calc(100%-3rem)] overflow-y-auto rounded-xl border border-[#2a2a3a] bg-[#13131a]/95 backdrop-blur p-4 shadow-2xl">
                          <div className="flex items-center gap-2 mb-3">
                            <StatusDot status={selectedMemory.status} />
                            <MemoryBadge type={selectedMemory.memory_type} />
                            <span className="text-[10px] font-mono text-[#555567]">{selectedMemory.id.slice(0, 8)}</span>
                            <button onClick={() => setSelectedMemory(null)} className="ml-auto text-[#555567] hover:text-[#f0f0f4] text-sm leading-none">✕</button>
                          </div>
                          <p className="text-sm text-[#e0e0ec] leading-relaxed whitespace-pre-wrap break-words">{selectedMemory.content}</p>
                          {selectedMemory.deprecated_reason && (
                            <p className="mt-3 text-xs text-red-400/80 border-t border-[#2a2a3a] pt-2">
                              <span className="uppercase tracking-wider text-[10px] text-red-400/60">Deprecated</span><br />
                              {selectedMemory.deprecated_reason}
                            </p>
                          )}
                          <div className="mt-3 pt-2 border-t border-[#2a2a3a] grid grid-cols-2 gap-1 text-[10px] text-[#555567]">
                            <span>importance</span><span className="text-[#8888a0]">{(selectedMemory.importance_score * 100).toFixed(0)}%</span>
                            <span>recency</span><span className="text-[#8888a0]">{(selectedMemory.recency_score * 100).toFixed(0)}%</span>
                            <span>accessed</span><span className="text-[#8888a0]">{selectedMemory.access_count}×</span>
                            <span>created</span><span className="text-[#8888a0]">{new Date(selectedMemory.created_at).toLocaleString()}</span>
                          </div>
                        </div>
                      )}
                    </div>
                    <GraphLegend />
                  </>
                )}
              </div>
            ) : (
              <div className="flex-1 overflow-y-auto p-6">
                <div className="max-w-3xl mx-auto flex flex-col gap-2">
                  {/* Filters */}
                  <div className="flex flex-wrap items-center gap-2 mb-2">
                    <div className="flex items-center gap-1">
                      <span className="text-[10px] text-[#555567] uppercase tracking-widest mr-1">Status</span>
                      {(["all", "active", "pending", "deprecated", "archived"] as const).map(s => (
                        <button
                          key={s}
                          onClick={() => setMemStatusFilter(s)}
                          className={clsx(
                            "px-2 py-0.5 rounded text-[10px] uppercase tracking-wider border transition-colors",
                            memStatusFilter === s
                              ? "bg-[#d4f000]/10 text-[#d4f000] border-[#d4f000]/20"
                              : "text-[#555567] border-transparent hover:text-[#8888a0]"
                          )}
                        >
                          {s}
                        </button>
                      ))}
                    </div>
                    <div className="flex items-center gap-1 sm:ml-2">
                      <span className="text-[10px] text-[#555567] uppercase tracking-widest mr-1">Type</span>
                      {(["all", "episodic", "semantic", "procedural"] as const).map(t => (
                        <button
                          key={t}
                          onClick={() => setMemTypeFilter(t)}
                          className={clsx(
                            "px-2 py-0.5 rounded text-[10px] uppercase tracking-wider border transition-colors",
                            memTypeFilter === t
                              ? "bg-[#d4f000]/10 text-[#d4f000] border-[#d4f000]/20"
                              : "text-[#555567] border-transparent hover:text-[#8888a0]"
                          )}
                        >
                          {t}
                        </button>
                      ))}
                    </div>
                    <span className="text-[10px] text-[#555567] ml-auto font-mono">
                      {filteredMemories.length} result{filteredMemories.length === 1 ? "" : "s"}
                    </span>
                  </div>

                  {filteredMemories.length === 0 && (
                    <div className="text-center py-12 text-[#555567] text-sm">
                      {memories.length === 0 ? "No memories yet." : "No memories match these filters."}
                    </div>
                  )}
                  {pagedMemories.map(m => (
                    <div key={m.id} className={clsx("rounded-xl border p-4", m.status === "active" ? "bg-[#13131a] border-[#2a2a3a]" : m.status === "deprecated" ? "bg-red-500/5 border-red-500/20 opacity-60" : "bg-[#0f0f14] border-[#1f1f2a] opacity-40")}>
                      <div className="flex items-start gap-3">
                        <StatusDot status={m.status} />
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2 mb-1 flex-wrap">
                            <MemoryBadge type={m.memory_type} />
                            <span className="text-[10px] font-mono text-[#555567]">{m.id.slice(0, 8)}</span>
                            <span className="text-[10px] text-[#555567]">imp {(m.importance_score * 100).toFixed(0)}%</span>
                            <span className="text-[10px] text-[#555567]">{m.access_count}× accessed</span>
                          </div>
                          <p className={clsx("text-sm text-[#e0e0ec] leading-relaxed whitespace-pre-wrap break-words", !expandedMems.has(m.id) && "line-clamp-2")}>{m.content}</p>
                          {m.content.length > 140 && (
                            <button
                              onClick={() => toggleExpand(m.id)}
                              className="mt-1 text-[10px] uppercase tracking-widest text-[#555567] hover:text-[#d4f000] transition-colors"
                            >
                              {expandedMems.has(m.id) ? "Show less" : "Show more"}
                            </button>
                          )}
                        </div>
                      </div>
                    </div>
                  ))}

                  {memPageCount > 1 && (
                    <div className="flex items-center justify-center gap-3 mt-3">
                      <button
                        onClick={() => setMemPage(p => Math.max(0, p - 1))}
                        disabled={clampedMemPage === 0}
                        className="px-3 py-1 rounded text-[10px] uppercase tracking-widest border border-[#2a2a3a] text-[#8888a0] hover:text-[#d4f000] hover:border-[#d4f000]/30 transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
                      >
                        Prev
                      </button>
                      <span className="text-[10px] text-[#555567] font-mono">
                        {clampedMemPage + 1} / {memPageCount}
                      </span>
                      <button
                        onClick={() => setMemPage(p => Math.min(memPageCount - 1, p + 1))}
                        disabled={clampedMemPage >= memPageCount - 1}
                        className="px-3 py-1 rounded text-[10px] uppercase tracking-widest border border-[#2a2a3a] text-[#8888a0] hover:text-[#d4f000] hover:border-[#d4f000]/30 transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
                      >
                        Next
                      </button>
                    </div>
                  )}

                  {playbooks.length > 0 && (
                    <div className="mt-6">
                      <h3 className="text-xs font-bold uppercase tracking-widest text-[#8888a0] mb-4">Playbooks</h3>
                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                        {playbooks.map(pb => (
                          <div key={pb.id} className="rounded-xl border border-[#2a2a3a] bg-[#13131a] p-4">
                            <div className="flex items-center gap-2 mb-1">
                              <GitBranch size={12} className="text-violet-400" />
                              <span className="text-sm font-bold">{pb.name}</span>
                              <span className="text-[10px] font-mono text-[#555567]">v{pb.version}</span>
                            </div>
                            <p className="text-xs text-[#555567] line-clamp-2">{pb.description}</p>
                            <div className="flex gap-3 mt-2 pt-2 border-t border-[#2a2a3a] text-xs text-[#555567]">
                              <span>{pb.session_count} runs</span>
                              {pb.last_run_at && <span>Last {new Date(pb.last_run_at).toLocaleDateString()}</span>}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        )}

        {/* ── METRICS TAB ── */}
        {activeTab === "metrics" && (
          <div className="flex-1 overflow-y-auto p-6">
            <div className="max-w-4xl mx-auto">
              <h2 className="text-xs font-bold uppercase tracking-widest text-[#8888a0] mb-6">Learning Curve</h2>

              {summary && (
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-8">
                  <MetricCard label="Sessions" value={summary.total_sessions} sub={`${summary.success_rate}% success`} icon={Activity} />
                  <MetricCard label="Avg Steps" value={summary.avg_steps} sub="per session" icon={GitBranch} />
                  <MetricCard label="Avg Tokens" value={summary.avg_tokens.toLocaleString()} sub="per session" icon={Zap} />
                  <MetricCard
                    label="Token Efficiency"
                    value={summary.token_efficiency_improvement_pct !== null ? `${summary.token_efficiency_improvement_pct}%` : "—"}
                    sub="improvement vs first sessions"
                    icon={CheckCircle}
                    accent={!!summary.token_efficiency_improvement_pct}
                  />
                </div>
              )}

              {sessionMetrics.length > 1 ? (
                <>
                  <div className="bg-[#13131a] border border-[#2a2a3a] rounded-xl p-6 mb-4">
                    <h3 className="text-xs uppercase tracking-widest text-[#555567] mb-4">Tokens Used Per Session</h3>
                    <ResponsiveContainer width="100%" height={200}>
                      <LineChart data={sessionMetrics}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#2a2a3a" />
                        <XAxis dataKey="session_number" tick={{ fill: "#555567", fontSize: 10 }} />
                        <YAxis tick={{ fill: "#555567", fontSize: 10 }} />
                        <Tooltip contentStyle={{ background: "#13131a", border: "1px solid #2a2a3a", borderRadius: 8, fontSize: 11 }} />
                        <Line type="monotone" dataKey="tokens_used" stroke="#d4f000" strokeWidth={2} dot={{ fill: "#d4f000", r: 4 }} name="Tokens Used" />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                  <div className="bg-[#13131a] border border-[#2a2a3a] rounded-xl p-6">
                    <h3 className="text-xs uppercase tracking-widest text-[#555567] mb-4">Steps Taken Per Session</h3>
                    <ResponsiveContainer width="100%" height={160}>
                      <LineChart data={sessionMetrics}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#2a2a3a" />
                        <XAxis dataKey="session_number" tick={{ fill: "#555567", fontSize: 10 }} />
                        <YAxis tick={{ fill: "#555567", fontSize: 10 }} />
                        <Tooltip contentStyle={{ background: "#13131a", border: "1px solid #2a2a3a", borderRadius: 8, fontSize: 11 }} />
                        <Line type="monotone" dataKey="steps_taken" stroke="#3b82f6" strokeWidth={2} dot={{ fill: "#3b82f6", r: 4 }} name="Steps" />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </>
              ) : (
                <div className="bg-[#13131a] border border-[#2a2a3a] rounded-xl p-8 text-center text-[#555567] text-sm">
                  Complete 2+ sessions to see the learning curve — or{" "}
                  <button onClick={seedDemo} className="text-[#d4f000] hover:underline">{seeding ? "loading…" : "load demo data"}</button>.
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
                <h2 className="text-xs font-bold uppercase tracking-widest text-[#8888a0]">Forgetting Engine</h2>
                <button
                  onClick={triggerConsolidation}
                  disabled={consolidating}
                  className={clsx("flex items-center gap-1.5 px-4 py-2 rounded-lg text-xs font-bold transition-all border", consolidating ? "bg-[#1a1a24] border-[#2a2a3a] text-[#555567] cursor-wait" : "bg-[#d4f000]/10 border-[#d4f000]/20 text-[#d4f000] hover:bg-[#d4f000]/15")}
                >
                  <RotateCcw size={12} className={consolidating ? "animate-spin" : ""} />
                  {consolidating ? "Running…" : "Run Now"}
                </button>
              </div>

              {summary && (
                <div className="grid grid-cols-3 gap-3 mb-6">
                  <MetricCard label="Belief Revisions" value={summary.belief_revisions} sub="facts corrected" icon={RotateCcw} accent={summary.belief_revisions > 0} />
                  <MetricCard label="Duplicates Merged" value={summary.duplicates_merged} sub="consolidated" icon={Layers} />
                  <MetricCard label="Decayed" value={summary.decayed} sub="archived from disuse" icon={Clock} />
                </div>
              )}

              {lastDiff.length > 0 && (
                <div className="bg-[#13131a] border border-[#2a2a3a] rounded-xl p-5 mb-6">
                  <h3 className="text-xs uppercase tracking-widest text-[#555567] mb-3">Latest Memory Diff</h3>
                  {lastDiff.map((entry, i) => <DiffEntry key={i} entry={entry} />)}
                </div>
              )}

              <div className="flex flex-col gap-3">
                {consolidationRuns.map(run => (
                  <div key={run.id} className="bg-[#13131a] border border-[#2a2a3a] rounded-xl p-4">
                    <div className="flex items-start justify-between gap-4">
                      <div>
                        <div className="flex items-center gap-2 mb-1">
                          <span className="text-xs font-mono text-[#555567]">{run.id.slice(0, 8)}</span>
                          <span className="text-[10px] px-1.5 py-0.5 bg-[#1a1a24] border border-[#2a2a3a] rounded font-mono text-[#555567] uppercase">{run.triggered_by}</span>
                          {run.error && <span className="text-[10px] text-red-400">ERROR</span>}
                        </div>
                        <div className="flex gap-4 flex-wrap text-xs text-[#555567]">
                          <span><span className="text-green-400">+{run.memories_activated}</span> activated</span>
                          <span><span className="text-red-400">-{run.memories_deprecated}</span> deprecated</span>
                          <span><span className="text-gray-400">{run.memories_archived}</span> archived</span>
                          <span><span className="text-[#d4f000]">{run.contradictions_resolved}</span> contradictions</span>
                        </div>
                      </div>
                      <div className="text-right text-[10px] text-[#555567]">
                        <Clock size={8} className="inline mr-1" />
                        {new Date(run.started_at).toLocaleTimeString()}
                        <div className="mt-0.5">{run.memories_scanned} scanned</div>
                      </div>
                    </div>
                  </div>
                ))}
                {consolidationRuns.length === 0 && (
                  <div className="text-center py-12 text-[#555567] text-sm">
                    No runs yet. Engine runs every 30 min automatically, or trigger manually above.
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
