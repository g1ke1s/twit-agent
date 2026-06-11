# Voice Agent v2

A content pipeline that researches a topic, takes a contrarian angle on it, writes in your voice, and puts a human in the loop before anything goes out.

**Backend:** FastAPI + LangGraph · **Frontend:** Next.js 14 · **Deployed:** Railway + Vercel

---

## Pipeline

```
input_router → researcher → analyst → angle_finder → writer → critic
                                                                  ↓ score < 6.5 → back to writer (max 2 iterations)
                                                                  ↓ score ≥ 6.5
                                                        validator_gate  ← HUMAN: approve / edit / reject
                                                                  ↓ approved
                                                        formatter → distributor
```

| Node | What it does | Model |
|---|---|---|
| `input_router` | Normalise input, detect type (tweet URL / web URL / file / instruction) | — |
| `researcher` | Concurrent Tavily + Exa + HN + arXiv search → Jina enrichment → MiniLM rerank → 6 sources, ≤6k tokens | llama-3.3-70b (query expansion) |
| `analyst` | Extract source-grounded claims with surprise scores, produce thesis. Speculative-input detection prevents hypothesis→fact promotion | llama-3.3-70b |
| `angle_finder` | Enumerate obvious takes, generate contrarian thesis + second-order consequence. Validates chosen angle for entity-level concreteness and overlap with clichés | llama-3.3-70b |
| `writer` | Writes draft arguing the chosen thesis as a through-line. Per-tweet concreteness scrub: abstract tweets replaced with verified claims. Hook promotion: most-specific tweet moved to position 1 | Gemini 2.0 Flash → Mistral fallback |
| `critic` | Three LLM axes (hook strength, insight density, clarity) + syntactic concreteness (numbers + proper nouns per tweet) + Tavily originality search with MiniLM cosine similarity + FAISS voice match | llama-3.3-70b |
| `validator_gate` | LangGraph interrupt. Human sees draft, critic scores, sources, trace | — |
| `formatter` | Thread JSON, copy-ready plaintext, LinkedIn and newsletter variants | — |
| `distributor` | Persist to Supabase, update FAISS voice index from approved output | — |

### LLM fallback chain

Most nodes use `call_llama_stack`: Groq key1 → Groq key2 → Cerebras → OpenRouter. On 429, skip — no sleep, just next provider.

Writer iteration 1 uses Gemini 2.0 Flash with an immediate Mistral fallback if Gemini returns empty. Writer iteration 2+ uses Mistral directly.

### Critic verdict logic

```
concreteness < 5.0  → REWRITE  (named-entity veto)
originality  < 6.0  → REWRITE  (semantic similarity veto — too many matching web results)
average      < 6.5  → REWRITE
iteration    ≥ 2    → APPROVE  (fail-safe cap)
```

---

## Input types

| Type | How to trigger |
|---|---|
| `instruction` | Plain text topic or question (default) |
| `tweet_url` | `https://x.com/...` or `https://twitter.com/...` |
| `web_url` | Any other URL — fetched via Jina Reader |
| `file_upload` | Upload via UI; PDF, DOCX, TXT, CSV, JSON |
| `voice_archive` | Upload writing samples to build your FAISS voice profile |

## Output types

`x_thread` · `quote_rt` · `essay` · `analysis` · `strategic_narrative`

---

## Stack

| Layer | Technology |
|---|---|
| Pipeline | [LangGraph](https://github.com/langchain-ai/langgraph) `StateGraph`, `MemorySaver`, `interrupt_before=["validator_gate"]` |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local) — source reranking, originality, voice match |
| Vector index | FAISS (local, per user) |
| Persistence | Supabase (runs, voice\_profiles, feedback) |
| API | FastAPI, SSE via `StreamingResponse` |
| Frontend | Next.js 14 App Router, Tailwind CSS |
| Backend deploy | Railway (Nixpacks, `uvicorn api.main:app --host 0.0.0.0 --port $PORT`) |
| Frontend deploy | Vercel |

---

## Quick start

```bash
git clone https://github.com/g1ke1s/twit-agent
cd twit-agent

# Backend
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
mkdir -p memory/faiss

cp .env.example .env          # fill in API keys (see below)
uvicorn api.main:app --reload --port 8000

# Frontend (new terminal)
cd frontend
cp .env.local.example .env.local   # or create: NEXT_PUBLIC_API_URL=http://localhost:8000
npm install && npm run dev
# → http://localhost:3000
```

### Environment variables

**Required:**

| Variable | Where to get it |
|---|---|
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) |
| `GEMINI_API_KEY` | [aistudio.google.com](https://aistudio.google.com) |
| `MISTRAL_API_KEY` | [console.mistral.ai](https://console.mistral.ai) |

**Optional (more coverage, better quality):**

| Variable | Purpose |
|---|---|
| `GROQ_API_KEY_2` | Second Groq key — llama-stack fallback |
| `GEMINI_API_KEY_2` | Second Gemini key — writer fallback |
| `CEREBRA_KEY` | Cerebras — llama-stack provider 3 |
| `OPENROUTER_API_KEY` | OpenRouter — llama-stack provider 4 |
| `TAVILY_API_KEY` | Web search (originality check + researcher) |
| `EXA_API_KEY` | Semantic search (researcher) |
| `SUPABASE_URL` | Run persistence |
| `SUPABASE_SERVICE_KEY` | Run persistence |

**Frontend (`frontend/.env.local`):**
```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

---

## API

All generation responses are Server-Sent Events.

```bash
# Generate content
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "raw_input": "Why Zapier is losing to Claude API integrations",
    "output_type": "x_thread",
    "user_id": "alice"
  }'

# SSE event types: start | node_start | node_complete | awaiting_validation | complete | error

# Poll run state (for validator UI)
curl http://localhost:8000/api/runs/{run_id}

# Submit validation decision
curl -X POST http://localhost:8000/api/validate/{run_id} \
  -H "Content-Type: application/json" \
  -d '{"decision": "approve"}'
  # decision: "approve" | "edit" | "reject"
  # optional: "rejection_note": "...", "human_edits": "..."

# Upload content file
curl -X POST http://localhost:8000/api/upload/content-file \
  -F "file=@report.pdf"

# Build voice profile
curl -X POST http://localhost:8000/api/memory/voice-archive \
  -F "user_id=alice" \
  -F "file=@my_tweets.txt"

# Health check
curl http://localhost:8000/health
```

---

## Deployment

**Backend → Railway:**
- Push to GitHub; connect repo in Railway
- Set all environment variables in Railway → Variables
- For cross-origin frontend: set `CORS_ORIGINS=https://your-app.vercel.app` (or `*` to allow all)
- `railway.toml` is already configured

**Frontend → Vercel:**
- Connect repo in Vercel
- Add environment variable: `NEXT_PUBLIC_API_URL=https://your-backend.up.railway.app`
- Redeploy after setting the variable (Next.js bakes `NEXT_PUBLIC_*` at build time)

---

## Project structure

```
agent/
  graph.py              LangGraph graph definition + compile
  schemas.py            All Pydantic models (AgentState, InputContext, CritiqueResult, ...)
  llm.py                Multi-provider LLM helpers (llama-stack, Gemini, Mistral)
  tools.py              Tavily, Exa, HN, arXiv, Jina wrappers
  nodes/
    input_router.py     Input normalisation + type detection
    researcher.py       Multi-source search + rerank
    analyst.py          Claim extraction + thesis
    angle_finder.py     Contrarian angle selection
    writer.py           Draft generation + concreteness enforcement
    critic.py           Scoring + originality + voice match
    validator_gate.py   LangGraph interrupt node
    formatter.py        Platform-native output formatting
    distributor.py      Supabase persist + FAISS update

api/
  main.py               FastAPI app, CORS, startup key check
  routers/
    generate.py         POST /api/generate — SSE stream
    validate.py         GET /api/runs/{id}, POST /api/validate/{id}
    memory.py           File upload, voice archive endpoints

memory/
  embedder.py           sentence-transformers + FAISS utilities
  store.py              Supabase client (runs, voice_profiles, feedback)

frontend/src/
  app/page.jsx          Main page — SSE event handler
  components/
    GeneratePanel.jsx   Input form + output type selector
    StageTracker.jsx    Pipeline progress + critic scorecard
    ValidatorPanel.jsx  Draft review, approve/edit/reject
    ThreadPreview.jsx   Tweet card rendering
    SharedOutput.jsx    Public share page output
```

---

## Adding a new output type

1. Add to `OutputType` enum — `agent/schemas.py`
2. Add writer branch in the `else` block — `agent/nodes/writer.py`
3. Add formatter branch — `agent/nodes/formatter.py`
4. Add to `OUTPUT_TYPES` array — `frontend/src/components/GeneratePanel.jsx`
