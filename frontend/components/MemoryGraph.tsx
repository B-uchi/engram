"use client";

import { useEffect, useRef, useCallback, useState, useMemo } from "react";
import * as d3 from "d3";
import type { Memory } from "@/lib/api";

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

interface GraphNode extends d3.SimulationNodeDatum {
  id: string;
  content: string;
  memory_type: string;
  status: string;
  importance_score: number;
  recency_score: number;
  access_count: number;
  created_at: string;
  superseded_by?: string | null;
  isNew?: boolean;
}

interface GraphLink extends d3.SimulationLinkDatum<GraphNode> {
  source: string | GraphNode;
  target: string | GraphNode;
  type: "supersedes" | "session";
}

interface Props {
  memories: Memory[];
  width?: number;
  height?: number;
  responsive?: boolean;
  onSelect?: (m: Memory) => void;
}

// ─────────────────────────────────────────────────────────────────────────────
// Color tokens — must match tailwind config exactly
// ─────────────────────────────────────────────────────────────────────────────

const STATUS_COLOR: Record<string, string> = {
  active: "#22c55e",
  pending: "#f59e0b",
  deprecated: "#ef4444",
  archived: "#4b5563",
};

const TYPE_COLOR: Record<string, string> = {
  episodic: "#60a5fa",
  semantic: "#34d399",
  procedural: "#a78bfa",
};

const STATUS_GLOW: Record<string, string> = {
  active: "rgba(34,197,94,0.35)",
  pending: "rgba(245,158,11,0.35)",
  deprecated: "rgba(239,68,68,0.35)",
  archived: "rgba(75,85,99,0.2)",
};

// ─────────────────────────────────────────────────────────────────────────────
// MemoryGraph
// ─────────────────────────────────────────────────────────────────────────────

export default function MemoryGraph({ memories, width: propWidth = 900, height: propHeight = 480, responsive = true, onSelect }: Props) {
  const svgRef = useRef<SVGSVGElement>(null);
  const onSelectRef = useRef(onSelect);
  onSelectRef.current = onSelect;
  const containerRef = useRef<HTMLDivElement>(null);
  const simRef = useRef<d3.Simulation<GraphNode, GraphLink> | null>(null);
  const prevIdsRef = useRef<Set<string>>(new Set());

  // Responsive dimensions
  const [dimensions, setDimensions] = useState({ width: propWidth, height: propHeight });
  const width = dimensions.width;
  const height = dimensions.height;

  // Only rebuild the simulation when content actually changes, not on every
  // 8s poll (which hands us a fresh array reference). Keyed by a content signature.
  const memoriesRef = useRef(memories);
  memoriesRef.current = memories;
  const signature = useMemo(
    () =>
      memories
        .map((m) => `${m.id}:${m.status}:${m.importance_score}:${m.superseded_by ?? ""}`)
        .sort()
        .join("|"),
    [memories]
  );

  // Track container size for responsive mode
  useEffect(() => {
    if (!responsive || !containerRef.current) return;

    const resizeObserver = new ResizeObserver(() => {
      if (containerRef.current) {
        const rect = containerRef.current.getBoundingClientRect();
        setDimensions({
          width: Math.max(300, rect.width),
          height: Math.max(200, rect.height),
        });
      }
    });

    resizeObserver.observe(containerRef.current);
    return () => resizeObserver.disconnect();
  }, [responsive]);

  const buildGraph = useCallback(
    (memories: Memory[]) => {
      const nodes: GraphNode[] = memories.map((m) => ({
        id: m.id,
        content: m.content,
        memory_type: m.memory_type,
        status: m.status,
        importance_score: m.importance_score,
        recency_score: m.recency_score,
        access_count: m.access_count,
        created_at: m.created_at,
        superseded_by: m.superseded_by,
        isNew: !prevIdsRef.current.has(m.id),
      }));

      const nodeIds = new Set(nodes.map((n) => n.id));
      const links: GraphLink[] = [];

      // Supersession edges from real superseded_by data (deprecated → replacement).
      for (const n of nodes) {
        if (n.superseded_by && nodeIds.has(n.superseded_by)) {
          links.push({ source: n.id, target: n.superseded_by, type: "supersedes" });
        }
      }

      // Temporal chain — link memories written within 2 min (same session).
      const superPairs = new Set(links.map((l) => `${l.source}->${l.target}`));
      const sorted = [...nodes].sort(
        (a, b) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime()
      );
      for (let i = 0; i < sorted.length - 1; i++) {
        const a = sorted[i];
        const b = sorted[i + 1];
        const tA = new Date(a.created_at).getTime();
        const tB = new Date(b.created_at).getTime();
        if (
          tB - tA < 120_000 &&
          !superPairs.has(`${a.id}->${b.id}`) &&
          !superPairs.has(`${b.id}->${a.id}`)
        ) {
          links.push({ source: a.id, target: b.id, type: "session" });
        }
      }

      return { nodes, links };
    },
    []
  );

  useEffect(() => {
    const memories = memoriesRef.current;
    if (!svgRef.current || memories.length === 0) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const { nodes, links } = buildGraph(memories);

    // ── Defs: glows + arrowheads ──────────────────────────────────────────
    const defs = svg.append("defs");

    // Glow filters per status
    Object.entries(STATUS_GLOW).forEach(([status, color]) => {
      const filter = defs
        .append("filter")
        .attr("id", `glow-${status}`)
        .attr("x", "-50%")
        .attr("y", "-50%")
        .attr("width", "200%")
        .attr("height", "200%");
      filter
        .append("feGaussianBlur")
        .attr("stdDeviation", "4")
        .attr("result", "coloredBlur");
      const merge = filter.append("feMerge");
      merge.append("feMergeNode").attr("in", "coloredBlur");
      merge.append("feMergeNode").attr("in", "SourceGraphic");
    });

    // Arrowhead for supersedes links
    defs
      .append("marker")
      .attr("id", "arrow-supersedes")
      .attr("viewBox", "0 -5 10 10")
      .attr("refX", 18)
      .attr("refY", 0)
      .attr("markerWidth", 6)
      .attr("markerHeight", 6)
      .attr("orient", "auto")
      .append("path")
      .attr("d", "M0,-5L10,0L0,5")
      .attr("fill", "#ef4444")
      .attr("opacity", 0.7);

    // ── Background ────────────────────────────────────────────────────────
    svg
      .append("rect")
      .attr("width", width)
      .attr("height", height)
      .attr("fill", "#0d0d0f")
      .attr("rx", 12);

    // Subtle grid
    const gridGroup = svg.append("g").attr("opacity", 0.04);
    for (let x = 0; x < width; x += 40) {
      gridGroup
        .append("line")
        .attr("x1", x)
        .attr("y1", 0)
        .attr("x2", x)
        .attr("y2", height)
        .attr("stroke", "#ffffff")
        .attr("stroke-width", 0.5);
    }
    for (let y = 0; y < height; y += 40) {
      gridGroup
        .append("line")
        .attr("x1", 0)
        .attr("y1", y)
        .attr("x2", width)
        .attr("y2", y)
        .attr("stroke", "#ffffff")
        .attr("stroke-width", 0.5);
    }

    // ── Clip path ─────────────────────────────────────────────────────────
    defs
      .append("clipPath")
      .attr("id", "graph-clip")
      .append("rect")
      .attr("width", width)
      .attr("height", height)
      .attr("rx", 12);

    const container = svg.append("g").attr("clip-path", "url(#graph-clip)");

    // ── Zoom ──────────────────────────────────────────────────────────────
    const zoom = d3
      .zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.3, 3])
      .on("zoom", (event) => {
        zoomGroup.attr("transform", event.transform);
      });
    svg.call(zoom);

    const zoomGroup = container.append("g");

    // ── Links ─────────────────────────────────────────────────────────────
    const linkGroup = zoomGroup.append("g");
    const linkSel = linkGroup
      .selectAll<SVGLineElement, GraphLink>("line")
      .data(links)
      .join("line")
      .attr("stroke", (d) =>
        d.type === "supersedes" ? "#ef4444" : "#5b5b7a"
      )
      .attr("stroke-width", (d) => (d.type === "supersedes" ? 1.5 : 1.25))
      .attr("stroke-opacity", (d) => (d.type === "supersedes" ? 0.6 : 0.6))
      .attr("stroke-dasharray", (d) =>
        d.type === "supersedes" ? "4 3" : "none"
      )
      .attr("marker-end", (d) =>
        d.type === "supersedes" ? "url(#arrow-supersedes)" : null
      );

    // ── Nodes ─────────────────────────────────────────────────────────────
    const nodeGroup = zoomGroup.append("g");
    const nodeSel = nodeGroup
      .selectAll<SVGGElement, GraphNode>("g")
      .data(nodes, (d) => d.id)
      .join("g")
      .attr("cursor", "pointer")
      .call(
        d3
          .drag<SVGGElement, GraphNode>()
          .on("start", (event, d) => {
            if (!event.active) simRef.current?.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
          })
          .on("drag", (event, d) => {
            d.fx = event.x;
            d.fy = event.y;
          })
          .on("end", (event, d) => {
            if (!event.active) simRef.current?.alphaTarget(0);
            d.fx = null;
            d.fy = null;
          })
      );

    // Node radius scales with importance
    const rScale = d3.scaleLinear().domain([0, 1]).range([7, 18]);

    // Outer glow ring
    nodeSel
      .append("circle")
      .attr("r", (d) => rScale(d.importance_score) + 5)
      .attr("fill", "none")
      .attr("stroke", (d) => STATUS_COLOR[d.status] || "#555")
      .attr("stroke-width", 1)
      .attr("stroke-opacity", (d) => (d.status === "active" ? 0.3 : 0.1))
      .attr("filter", (d) => `url(#glow-${d.status})`);

    // Main node circle
    nodeSel
      .append("circle")
      .attr("r", (d) => rScale(d.importance_score))
      .attr("fill", (d) => {
        const base = TYPE_COLOR[d.memory_type] || "#888";
        return d.status === "deprecated" || d.status === "archived"
          ? "#1a1a24"
          : base + "22";
      })
      .attr("stroke", (d) =>
        d.status === "deprecated" || d.status === "archived"
          ? STATUS_COLOR[d.status]
          : TYPE_COLOR[d.memory_type] || "#888"
      )
      .attr("stroke-width", (d) =>
        d.status === "active" || d.status === "pending" ? 2 : 1.5
      )
      .attr("stroke-opacity", (d) =>
        d.status === "archived" ? 0.35 : 0.85
      );

    // Type indicator dot (centre)
    nodeSel
      .append("circle")
      .attr("r", 2.5)
      .attr("fill", (d) =>
        d.status === "deprecated" || d.status === "archived"
          ? STATUS_COLOR[d.status]
          : TYPE_COLOR[d.memory_type] || "#888"
      )
      .attr("opacity", (d) => (d.status === "archived" ? 0.4 : 0.9));

    // Repeating amber pulse for PENDING nodes.
    function pendingPulse(sel: d3.Selection<SVGCircleElement, GraphNode, any, unknown>) {
      sel
        .attr("r", (d) => rScale(d.importance_score))
        .attr("stroke-opacity", 0.8)
        .transition()
        .duration(1400)
        .ease(d3.easeSinInOut)
        .attr("r", (d) => rScale(d.importance_score) + 10)
        .attr("stroke-opacity", 0)
        .on("end", function () {
          pendingPulse(d3.select<SVGCircleElement, GraphNode>(this));
        });
    }
    pendingPulse(
      nodeSel
        .filter((d) => d.status === "pending")
        .append("circle")
        .attr("fill", "none")
        .attr("stroke", STATUS_COLOR.pending)
        .attr("stroke-width", 1.5) as d3.Selection<SVGCircleElement, GraphNode, any, unknown>
    );

    // "New memory" pulse ring — for nodes that just appeared
    nodeSel
      .filter((d) => d.isNew === true)
      .append("circle")
      .attr("r", (d) => rScale(d.importance_score))
      .attr("fill", "none")
      .attr("stroke", "#d4f000")
      .attr("stroke-width", 2)
      .attr("stroke-opacity", 0.9)
      .attr("class", "pulse-ring")
      .transition()
      .duration(1800)
      .ease(d3.easeCubicOut)
      .attr("r", (d) => rScale(d.importance_score) + 16)
      .attr("stroke-opacity", 0)
      .remove();

    // ── Tooltip ───────────────────────────────────────────────────────────
    const tooltip = d3
      .select("body")
      .selectAll<HTMLDivElement, unknown>("#engram-graph-tooltip")
      .data([null])
      .join("div")
      .attr("id", "engram-graph-tooltip")
      .style("position", "fixed")
      .style("pointer-events", "none")
      .style("background", "#13131a")
      .style("border", "1px solid #2a2a3a")
      .style("border-radius", "10px")
      .style("padding", "10px 14px")
      .style("font-family", "Syne, sans-serif")
      .style("font-size", "11px")
      .style("color", "#e0e0ec")
      .style("max-width", "260px")
      .style("line-height", "1.5")
      .style("z-index", "9999")
      .style("opacity", "0")
      .style("transition", "opacity 0.15s");

    nodeSel
      .on("mouseenter", (event, d) => {
        const typeColor = TYPE_COLOR[d.memory_type] || "#888";
        const statusColor = STATUS_COLOR[d.status] || "#888";
        tooltip
          .html(
            `<div style="display:flex;gap:6px;align-items:center;margin-bottom:6px">
              <span style="width:8px;height:8px;border-radius:50%;background:${statusColor};display:inline-block;flex-shrink:0"></span>
              <span style="font-weight:700;color:${typeColor};text-transform:uppercase;font-size:9px;letter-spacing:0.1em">${d.memory_type}</span>
              <span style="color:#555567;font-size:9px;margin-left:auto">${d.status}</span>
            </div>
            <div style="color:#f0f0f4;margin-bottom:8px;font-size:11px">${d.content.slice(0, 240)}${d.content.length > 240 ? "…" : ""}</div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:4px 12px;color:#555567;font-size:10px">
              <span>importance</span><span style="color:#8888a0">${(d.importance_score * 100).toFixed(0)}%</span>
              <span>recency</span><span style="color:#8888a0">${(d.recency_score * 100).toFixed(0)}%</span>
              <span>accessed</span><span style="color:#8888a0">${d.access_count}×</span>
              <span>id</span><span style="color:#8888a0;font-family:monospace">${d.id.slice(0, 8)}</span>
            </div>
            ${d.content.length > 240 ? '<div style="color:#d4f000;font-size:9px;margin-top:6px">click node for full detail →</div>' : ""}`
          )
          .style("opacity", "1");
      })
      .on("mousemove", (event) => {
        tooltip
          .style("left", event.clientX + 14 + "px")
          .style("top", event.clientY - 10 + "px");
      })
      .on("mouseleave", () => {
        tooltip.style("opacity", "0");
      })
      .on("click", (event, d) => {
        const full = memoriesRef.current.find((m) => m.id === d.id);
        if (full) onSelectRef.current?.(full);
      });

    // ── Simulation ────────────────────────────────────────────────────────
    const sim = d3
      .forceSimulation<GraphNode>(nodes)
      .force(
        "link",
        d3
          .forceLink<GraphNode, GraphLink>(links)
          .id((d) => d.id)
          .distance((d) => (d.type === "supersedes" ? 90 : 60))
          .strength((d) => (d.type === "supersedes" ? 0.4 : 0.2))
      )
      .force("charge", d3.forceManyBody().strength(-220))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collision", d3.forceCollide<GraphNode>().radius((d) => rScale(d.importance_score) + 12))
      .force("x", d3.forceX(width / 2).strength(0.04))
      .force("y", d3.forceY(height / 2).strength(0.04))
      .alphaDecay(0.025)
      .on("tick", () => {
        linkSel
          .attr("x1", (d) => (d.source as GraphNode).x ?? 0)
          .attr("y1", (d) => (d.source as GraphNode).y ?? 0)
          .attr("x2", (d) => (d.target as GraphNode).x ?? 0)
          .attr("y2", (d) => (d.target as GraphNode).y ?? 0);

        nodeSel.attr("transform", (d) => `translate(${d.x ?? 0},${d.y ?? 0})`);
      });

    simRef.current = sim;

    // Update previous IDs
    prevIdsRef.current = new Set(memories.map((m) => m.id));

    return () => {
      sim.stop();
      tooltip.remove();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [signature, width, height, buildGraph]);

  return (
    <div ref={containerRef} style={{ width: "100%", height: "100%", display: "flex" }}>
      <svg
        ref={svgRef}
        width={width}
        height={height}
        viewBox={`0 0 ${width} ${height}`}
        style={{ borderRadius: 12, display: "block", width: "100%", height: "100%", flex: 1 }}
      />
    </div>
  );
}
