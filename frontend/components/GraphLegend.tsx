"use client";

export default function GraphLegend() {
  const statuses = [
    { color: "#22c55e", label: "Active" },
    { color: "#f59e0b", label: "Pending" },
    { color: "#ef4444", label: "Deprecated" },
    { color: "#4b5563", label: "Archived" },
  ];
  const types = [
    { color: "#60a5fa", label: "Episodic" },
    { color: "#34d399", label: "Semantic" },
    { color: "#a78bfa", label: "Procedural" },
  ];

  return (
    <div className="flex-shrink-0 flex flex-wrap gap-x-6 gap-y-2 px-4 py-2 border-t border-[#1a1a24]">
      <div className="flex items-center gap-3 flex-wrap">
        <span className="text-[10px] text-[#555567] uppercase tracking-widest">Status</span>
        {statuses.map((s) => (
          <div key={s.label} className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full" style={{ backgroundColor: s.color }} />
            <span className="text-[10px] text-[#8888a0]">{s.label}</span>
          </div>
        ))}
      </div>
      <div className="flex items-center gap-3 flex-wrap">
        <span className="text-[10px] text-[#555567] uppercase tracking-widest">Type</span>
        {types.map((t) => (
          <div key={t.label} className="flex items-center gap-1.5">
            <span
              className="w-3 h-3 rounded-sm border"
              style={{ borderColor: t.color, backgroundColor: t.color + "22" }}
            />
            <span className="text-[10px] text-[#8888a0]">{t.label}</span>
          </div>
        ))}
      </div>
      <div className="flex items-center gap-3 flex-wrap">
        <span className="text-[10px] text-[#555567] uppercase tracking-widest">Edges</span>
        <div className="flex items-center gap-1.5">
          <svg width="20" height="8">
            <line x1="0" y1="4" x2="20" y2="4" stroke="#5b5b7a" strokeWidth="1.5" opacity="0.8" />
          </svg>
          <span className="text-[10px] text-[#8888a0]">Session chain</span>
        </div>
        <div className="flex items-center gap-1.5">
          <svg width="20" height="8">
            <line x1="0" y1="4" x2="20" y2="4" stroke="#ef4444" strokeWidth="1.5" strokeDasharray="4 3" opacity="0.7" />
          </svg>
          <span className="text-[10px] text-[#8888a0]">Supersedes (deprecated)</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="text-[10px] text-[#d4f000]">◉</span>
          <span className="text-[10px] text-[#8888a0]">Node size = importance</span>
        </div>
      </div>
    </div>
  );
}
