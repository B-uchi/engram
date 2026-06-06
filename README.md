# Engram

> A memory agent that learns workflows, gets faster at them, and self-corrects when they go stale.

Engram is a submission for the **Global AI Hackathon Series with Qwen Cloud** — Track 1: MemoryAgent.

It combines two memory mechanisms no existing agent demo tackles together:
- **Procedural memory**: watches a workflow once, distills it into a structured playbook, executes it faster and with fewer tokens each subsequent session
- **The Forgetting Engine**: a background consolidator that detects contradictions in playbooks, decays stale memories, and updates the playbook — so the agent doesn't just get faster, it stays *correct* as the world changes

## Architecture

```
User
  ↓
Next.js Frontend (UI + API Routes)
  ↓
FastAPI Backend (Qwen-Agent framework)
  ├── Active Agent (Qwen3, non-thinking, low latency)
  │     ├── Retrieves playbook from skills/ store
  │     ├── Executes workflow steps via MCP tools
  │     └── Writes episodic memories post-session
  │
  ├── Consolidator (Qwen3, thinking-mode, async)
  │     ├── Contradiction Detector
  │     ├── Decay Scorer
  │     ├── Belief Revision Engine
  │     └── Memory Diff Logger
  │
  └── MCP Server (custom)
        ├── write_memory
        ├── update_memory
        ├── deprecate_memory
        ├── query_memory (semantic + BM25 + recency + importance)
        └── get_playbook
  ↓
Memory Store (SQLite + ChromaDB)
  ├── Episodic memories
  ├── Procedural memories (playbooks)
  └── Semantic memories
  ↓
Alibaba Cloud ECS
```

## Setup

### Prerequisites
- Python 3.10+
- Node.js 18+
- A DashScope API key (from [Alibaba Cloud Model Studio](https://dashscope.aliyun.com))

### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Add your DASHSCOPE_API_KEY to .env
uvicorn api.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
cp .env.local.example .env.local
# NEXT_PUBLIC_API_URL=http://localhost:8000
npm run dev
```

Open [http://localhost:3000](http://localhost:3000)

## Tracks & Requirements

- **Track**: Track 1 — MemoryAgent
- **Models**: Qwen3-32B (active agent, non-thinking), Qwen3-32B (consolidator, thinking mode)
- **Infrastructure**: Alibaba Cloud ECS
- **Memory**: SQLite (structured) + ChromaDB (semantic vectors)
- **Protocol**: Custom MCP server for all memory operations

## License

MIT — see [LICENSE](./LICENSE)
