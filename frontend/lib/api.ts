const BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

async function req<T>(
  path: string,
  opts: RequestInit = {}
): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...opts.headers },
    ...opts,
  });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`API error ${res.status}: ${err}`);
  }
  return res.json();
}

export const api = {
  // ── Sessions ──────────────────────────────────────────────────────────────
  startSession: (playbookName?: string) =>
    req<{ session_id: string; playbook: any; initial_memories: any[] }>(
      "/sessions/start",
      {
        method: "POST",
        body: JSON.stringify({ playbook_name: playbookName }),
      }
    ),

  endSession: (
    sessionId: string,
    taskCompleted: boolean,
    sessionSummary: string,
    playbookUsed: boolean
  ) =>
    req<{ ok: boolean; reflection: string; steps_recorded: number }>(
      `/sessions/${sessionId}/end`,
      {
        method: "POST",
        body: JSON.stringify({
          session_id: sessionId,
          task_completed: taskCompleted,
          session_summary: sessionSummary,
          playbook_used: playbookUsed,
        }),
      }
    ),

  // ── Memory ────────────────────────────────────────────────────────────────
  getAllMemories: () =>
    req<{ memories: Memory[] }>("/memory/all"),

  queryMemory: (query: string, sessionId: string, tokenBudget = 4000) =>
    req<{ memories: Memory[]; count: number }>("/memory/query", {
      method: "POST",
      body: JSON.stringify({ query, session_id: sessionId, token_budget: tokenBudget }),
    }),

  // ── Playbooks ─────────────────────────────────────────────────────────────
  listPlaybooks: () =>
    req<{ playbooks: PlaybookSummary[] }>("/playbooks"),

  getPlaybook: (name: string) =>
    req<Playbook>(`/playbooks/${encodeURIComponent(name)}`),

  // ── Consolidation ─────────────────────────────────────────────────────────
  triggerConsolidation: () =>
    req<ConsolidationResult>("/consolidation/trigger", { method: "POST" }),

  getConsolidationHistory: () =>
    req<{ runs: ConsolidationRun[] }>("/consolidation/history"),

  // ── Metrics ───────────────────────────────────────────────────────────────
  getSessionMetrics: () =>
    req<{ sessions: SessionMetric[] }>("/metrics/sessions"),

  getSummaryMetrics: () =>
    req<SummaryMetrics>("/metrics/summary"),
};

// ── Types ──────────────────────────────────────────────────────────────────

export interface Memory {
  id: string;
  content: string;
  memory_type: "episodic" | "semantic" | "procedural";
  status: "active" | "pending" | "deprecated" | "archived";
  importance_score: number;
  recency_score: number;
  access_count: number;
  relevance_score?: number;
  superseded_by?: string | null;
  deprecated_reason?: string | null;
  created_at: string;
}

export interface PlaybookSummary {
  id: string;
  name: string;
  description: string;
  version: number;
  session_count: number;
  last_run_at: string | null;
}

export interface PlaybookStep {
  id: string;
  step_number: number;
  title: string;
  description: string;
  tool_used?: string;
  expected_output?: string;
  decision_point: boolean;
  edge_cases: string[];
  success_rate: number;
  status: string;
}

export interface Playbook extends PlaybookSummary {
  steps: PlaybookStep[];
}

export interface ConsolidationResult {
  run_id: string;
  completed_at: string;
  memories_activated: number;
  memories_deprecated: number;
  memories_archived: number;
  contradictions_resolved: number;
  error?: string;
}

export interface ConsolidationRun {
  id: string;
  started_at: string;
  completed_at: string | null;
  triggered_by: string;
  memories_scanned: number;
  memories_activated: number;
  memories_deprecated: number;
  memories_archived: number;
  contradictions_found: number;
  contradictions_resolved: number;
  diff_json: DiffEntry[];
  error?: string;
}

export interface DiffEntry {
  action: string;
  memory_id?: string;
  memory_type?: string;
  reason: string;
  content_preview?: string;
  superseded_by?: string;
  merged_content?: string;
  decay_score?: number;
  age_days?: number;
}

export interface SessionMetric {
  session_number: number;
  session_id: string;
  task_completed: boolean;
  steps_taken: number;
  tokens_used: number;
  playbook_used: boolean;
  started_at: string;
}

export interface SummaryMetrics {
  total_sessions: number;
  completed_sessions: number;
  success_rate: number;
  avg_steps: number;
  avg_tokens: number;
  token_efficiency_improvement_pct: number | null;
  active_memories: number;
  pending_memories: number;
  belief_revisions: number;
  duplicates_merged: number;
  decayed: number;
  last_consolidation: string | null;
  total_contradictions_resolved: number;
}

// ── SSE chat stream helper ─────────────────────────────────────────────────

export interface ChatStreamEvent {
  type: "chunk" | "done" | "error" | "tool_call" | "tool_result";
  content?: string;
  error?: string;
  tool?: string;
  name?: string;
  arguments?: string;
  ok?: boolean;
  call_index?: number;
}

export async function* streamChat(
  sessionId: string,
  message: string
): AsyncGenerator<ChatStreamEvent> {
  const res = await fetch(`${BASE}/sessions/${sessionId}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, message }),
  });

  if (!res.ok || !res.body) {
    throw new Error(`Chat stream failed: ${res.status}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    const lines = buffer.split("\n\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      if (line.startsWith("data: ")) {
        try {
          yield JSON.parse(line.slice(6));
        } catch {
          // Malformed chunk — skip
        }
      }
    }
  }
}
