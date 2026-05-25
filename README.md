# Voice Agent v2

Agentic content pipeline: research → analyse → write → critique → **human validates** → format → share.
Zero hallucination tolerance. Voice-matched output. Human always in the loop.

---

## What's new in v2

- **Beautiful UI** — Stage tracker (not a log dump). All raw events go to browser console.
- **File upload** — PDF, DOCX, TXT, CSV, JSON → extracted text → pipeline input.
- **Voice archive upload** — same file types for building your voice profile.
- **Share links** — after approval, one click creates a public `/share/{id}` URL.
- **Robust prompts** — all 9 system prompts rewritten with explicit scoring rubrics,
  architectural requirements, and hard prohibitions.

---

## Architecture

```
INPUT
  ├── Text / URL  →  Jina fetch or tweet scrape
  ├── File Upload →  PDF/DOCX/TXT extracted → pipeline
  └── Voice Archive → FAISS voice profile

PIPELINE (LangGraph, interrupt_before=validator_gate)
  [0] Input Router   → normalise + detect type
  [1] Voice Loader   → FAISS profile cache / build
  [2] Researcher     → Tavily + Exa + HN + arXiv (concurrent)
  [3] Analyst        → DeepSeek R1 (contrarian) + Llama 70B (systems)
  [4] Hook Generator → 12 hooks → top 4 scored by FAISS voice sim
  [5] Writer         → Gemini Flash + Qwen 72B + Llama 70B → MoA fusion
  [6] Critic         → Gemini (voice) + Mistral (density) + Llama 8B (facts)
        ↓ score < 7.5 → back to [5] (max 2 iterations)
        ↓ score ≥ 7.5 → INTERRUPT
  [7] *** HUMAN VALIDATOR GATE *** ← approve / edit / reject
  [8] Formatter      → thread JSON / markdown + LinkedIn + newsletter
  [9] Distributor    → Supabase persist + FAISS voice update

SHARE LINK → /api/share/{run_id} → POST → /share/{share_id} public page
```

---

## Quick start

```bash
# 1. Clone
git clone https://github.com/yourname/voice-agent
cd voice-agent

# 2. Environment
cp .env.example .env
# Fill in all API keys (see .env.example)

# 3. Supabase tables
# Open Supabase SQL editor → paste supabase_schema.sql → Run

# 4. Backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
mkdir -p memory/faiss
uvicorn api.main:app --reload --port 8000

# 5. Frontend (new terminal)
cd frontend
npm install
npm run dev
# → http://localhost:3000
```

---

## File upload — supported formats

| Format | Content extraction |
|--------|-------------------|
| `.pdf`  | PyMuPDF — text per page, splits on blank lines |
| `.docx` | python-docx — paragraph by paragraph |
| `.txt`  | Blank-line separated posts/paragraphs |
| `.csv`  | Treated as plain text |
| `.json` | Array → each item; or dict with `posts`/`tweets`/`texts` key |

Max file size: **10 MB** per upload.

**For content generation** (`/api/upload/content-file`):
```bash
curl -X POST http://localhost:8000/api/upload/content-file \
  -F "file=@my_report.pdf"
# → { filename, text, paragraphs, chars }
# Then pass text as uploaded_content in /api/generate
```

**For voice archive** (`/api/memory/voice-archive`):
```bash
curl -X POST http://localhost:8000/api/memory/voice-archive \
  -F "user_id=alice" \
  -F "file=@my_tweets_archive.txt"
# → { user_id, samples_processed, style_summary, ... }
```

---

## Generate content — curl examples

```bash
# Thread from topic
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"raw_input":"Why a16z is wrong about AI replacing engineers","output_type":"x_thread","user_id":"alice"}'

# Quote RT from tweet URL
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"raw_input":"https://x.com/naval/status/123","output_type":"quote_rt","user_id":"alice"}'

# Essay from uploaded file
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"raw_input":"Analyse key findings","input_type":"file_upload","output_type":"essay","uploaded_content":"<text from file>","user_id":"alice"}'

# All responses are Server-Sent Events. Consume with EventSource.
# All raw events are console.log'd in the browser — check DevTools > Console.
```

---

## Human validator gate

1. Pipeline runs autonomously through all 9 layers.
2. After critique passes (≥ 7.5) or hits iteration cap (2), graph **pauses**.
3. Frontend receives `awaiting_validation` SSE event → stage tracker shows "Human Review".
4. Click **Review Draft →** in the stage tracker → `/validate/{runId}`.
5. Validator page shows:
   - Full draft (tweet cards or essay)
   - Critic scorecard (4 scores with progress bars)
   - Research sources + credibility scores (collapsible)
   - Agent reasoning trace (collapsible)
6. Three decisions:
   - **Approve** → formatter runs, share link created
   - **Edit** → inline per-tweet or full essay editor, diff saved to memory
   - **Reject** → feedback note injected into writer system prompt, re-runs from Layer 5

---

## Share links

After approval, the validator page automatically creates a share link:

```bash
# Manual creation
curl -X POST http://localhost:8000/api/share/{run_id}
# → { share_id, share_url }

# Public view
https://your-app.vercel.app/share/{share_id}
```

The share page shows:
- The final output (thread cards or essay)
- Tab switcher: Original / LinkedIn / Newsletter variants
- Copy button for each format
- The shareable URL itself for forwarding

Share links are **read-only** and **require no login**. They use Supabase Row Level Security — `shared_runs` table has a public SELECT policy.

---

## Rate limit table

| Provider       | Model                   | Limit   | Sleep  |
|----------------|-------------------------|---------|--------|
| Gemini Flash   | gemini-2.0-flash-exp    | 15 RPM  | 500ms  |
| Groq           | llama-3.3-70b-versatile | 30 RPM  | 300ms  |
| Mistral        | mistral-small-latest    | 5 RPM   | 2000ms |
| OpenRouter     | deepseek/deepseek-r1    | 10 RPM  | 1000ms |
| OpenRouter     | qwen/qwen-2.5-72b       | 10 RPM  | 1000ms |
| Tavily         | search                  | 5 RPS   | 500ms  |
| Exa.ai         | search                  | 10 RPM  | 1000ms |
| Jina Reader    | r.jina.ai               | —       | 300ms  |

---

## Debugging

**All pipeline events are logged to browser console.** Open DevTools → Console to see:
- Every SSE event with full payload
- File parse results (chars, paragraphs)
- Voice profile build status
- Validation decisions
- Share link creation

**Backend logs** to stdout at DEBUG level. On Railway: Deployments → Logs.

---

## Deployment

**Backend → Railway:**
```
# railway.toml
[build]
  nixpacksVersion = "1"

[deploy]
  startCommand = "uvicorn api.main:app --host 0.0.0.0 --port $PORT"

railway up
# Set all env vars in Railway dashboard
```

**Frontend → Vercel:**
```
cd frontend
vercel deploy --prod
# Set NEXT_PUBLIC_API_URL=https://your-backend.up.railway.app
```

**Share link for task managers (Notion, Linear, Slack, etc.):**
After any approved run, copy the `/share/{id}` URL and paste it anywhere.
No login required. Works in Notion embeds, Slack previews, email links.

---

## How to add a new output type

1. Add to `OutputType` enum in `agent/schemas.py`
2. Add detection keywords in `agent/nodes/input_router.py` → `OUTPUT_KEYWORDS`
3. Create `prompts/writer_{type}.txt` (follow the structured format with sections)
4. Update `_build_writer_prompt()` in `agent/nodes/writer.py`
5. Update `formatter_node()` in `agent/nodes/formatter.py`
6. Add to `OUTPUT_TYPES` array in `frontend/src/components/GeneratePanel.jsx`
