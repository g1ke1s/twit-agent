"use client";
import { useState } from "react";
import ThreadPreview from "./ThreadPreview";
import { apiJson, getApiBase } from "@/lib/api";

// ─── Score bar ───────────────────────────────────────────────────────────────
function ScoreBar({ label, value }) {
  const pct   = value ? ((value - 1) / 9) * 100 : 0;
  const color = value >= 8 ? "bg-signal-green" : value >= 6 ? "bg-signal-amber" : "bg-signal-red";
  const textColor = value >= 8 ? "text-signal-green" : value >= 6 ? "text-signal-amber" : "text-signal-red";
  return (
    <div className="space-y-1.5">
      <div className="flex justify-between">
        <span className="text-xs text-ink-400">{label}</span>
        <span className={`text-xs font-mono font-bold ${textColor}`}>{value?.toFixed(1)}</span>
      </div>
      <div className="h-1 bg-ink-700 rounded-full overflow-hidden">
        <div className={`h-full rounded-full ${color} transition-all duration-700`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

// ─── Inline tweet editor ──────────────────────────────────────────────────────
function TweetEditor({ tweets, onSave }) {
  const [texts, setTexts] = useState(tweets.map(t => t.text || ""));

  function update(i, val) {
    const next = [...texts]; next[i] = val; setTexts(next);
  }

  return (
    <div className="space-y-3">
      {texts.map((text, i) => {
        const len = text.length;
        return (
          <div key={i} className="space-y-1">
            <div className="flex items-center justify-between">
              <span className="text-xs font-mono text-ink-500">{i + 1}/</span>
              <span className={`text-xs font-mono ${len > 280 ? "text-signal-red" : "text-ink-500"}`}>{len}/280</span>
            </div>
            <textarea
              rows={3}
              className={`va-input w-full px-3 py-2 text-sm resize-none ${len > 280 ? "border-signal-red/60" : ""}`}
              value={text}
              onChange={e => update(i, e.target.value)}
            />
          </div>
        );
      })}
      <button
        onClick={() => onSave(texts.join("\n"))}
        className="w-full py-2.5 bg-signal-amber/20 hover:bg-signal-amber/30 border border-signal-amber/30 text-signal-amber font-semibold text-sm rounded-xl transition-colors"
      >
        Save Edits →
      </button>
    </div>
  );
}

// ─── Essay editor ─────────────────────────────────────────────────────────────
function EssayEditor({ text, onSave }) {
  const [val, setVal] = useState(text || "");
  return (
    <div className="space-y-3">
      <textarea
        rows={16}
        className="va-input w-full px-3 py-2 text-sm resize-y"
        value={val}
        onChange={e => setVal(e.target.value)}
      />
      <button
        onClick={() => onSave(val)}
        className="w-full py-2.5 bg-signal-amber/20 hover:bg-signal-amber/30 border border-signal-amber/30 text-signal-amber font-semibold text-sm rounded-xl transition-colors"
      >
        Save Edits →
      </button>
    </div>
  );
}

// ─── Main component ───────────────────────────────────────────────────────────
export default function ValidatorPanel({ runId, runData }) {
  const [mode, setMode]               = useState(null); // null | "edit" | "reject"
  const [rejectionNote, setNote]      = useState("");
  const [submitting, setSubmitting]   = useState(false);
  const [result, setResult]           = useState(null);
  const [shareLink, setShareLink]     = useState(null);
  const [error, setError]             = useState(null);
  const [sourcesOpen, setSourcesOpen] = useState(false);
  const [traceOpen, setTraceOpen]     = useState(false);

  const draft   = runData?.draft;
  const critique = runData?.critique;
  const sources  = runData?.sources || [];
  const trace    = runData?.trace   || [];
  const isThread = Array.isArray(draft);

  async function submit({ decision, human_edits, rejection_note }) {
    setSubmitting(true);
    setError(null);
    console.log("[Voice Agent] Submitting validation:", { decision, human_edits: !!human_edits, rejection_note });
    try {
      const data = await apiJson(`/api/validate/${runId}`, {
        method: "POST",
        body: JSON.stringify({ decision, human_edits, rejection_note }),
      });
      setResult(data);

      // After rejection: poll until new draft is ready then reload
      if (data.decision === "reject") {
        const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
        let attempts = 0;
        const poll = setInterval(async () => {
          attempts++;
          try {
            const r = await fetch(`${API}/api/runs/${runId}`);
            const s = await r.json();
            if (s.awaiting_validation) {
              clearInterval(poll);
              window.location.reload();
            }
          } catch {}
          if (attempts > 60) clearInterval(poll); // give up after 2 min
        }, 2000);
      }

      // Create share link if approved/edited
      if (decision !== "reject") {
        try {
          const share = await apiJson(`/api/share/${runId}`, { method: "POST" });
          setShareLink(share.share_url);
          console.log("[Voice Agent] Share link created:", share.share_url);
        } catch (e) {
          console.warn("[Voice Agent] Share link creation failed:", e.message);
        }
      }
    } catch (e) {
      setError(e.message);
      console.error("[Voice Agent] Validation error:", e);
    } finally {
      setSubmitting(false);
    }
  }

  // ── Result state ─────────────────────────────────────────────────────────
  if (result) {
    const isApproved = result.decision !== "reject";
    return (
      <div className="max-w-xl mx-auto py-16 text-center space-y-6 animate-fade-in">
        <div className={`w-16 h-16 rounded-full mx-auto flex items-center justify-center text-2xl ${
          result.decision === "approve" ? "bg-signal-green/15 text-signal-green" :
          result.decision === "edit"    ? "bg-signal-amber/15 text-signal-amber" :
          "bg-signal-red/15 text-signal-red"
        }`}>
          {result.decision === "approve" ? "✓" : result.decision === "edit" ? "✎" : "↺"}
        </div>
        <div>
          <h2 className="text-xl font-semibold text-ink-100">
            {result.decision === "approve" ? "Approved" :
             result.decision === "edit"    ? "Edits saved" :
             "Rejected — re-running with your feedback"}
          </h2>
          <p className="text-ink-400 text-sm mt-2">
            {result.decision === "approve" && "Sending to formatter…"}
            {result.decision === "edit"    && "Applying your edits and formatting…"}
            {result.decision === "reject"  && "Writer is incorporating your feedback. The new draft will appear here shortly."}
          </p>
        </div>

        {result.decision === "reject" && (
          <div className="animate-pulse text-ink-500 text-sm">Polling for new draft…</div>
        )}

        {isApproved && shareLink && (
          <div className="glass rounded-xl p-4 text-left space-y-2">
            <p className="text-xs font-semibold text-ink-400 uppercase tracking-wide">Share link</p>
            <div className="flex items-center gap-2">
              <code className="flex-1 text-xs text-signal-teal font-mono bg-ink-800/60 px-3 py-2 rounded-lg truncate">
                {typeof window !== "undefined" ? window.location.origin : ""}{shareLink}
              </code>
              <button
                onClick={() => {
                  const url = `${window.location.origin}${shareLink}`;
                  navigator.clipboard?.writeText(url);
                }}
                className="px-3 py-2 bg-ink-700 hover:bg-ink-600 text-ink-200 text-xs rounded-lg transition-colors"
              >
                Copy
              </button>
            </div>
            <p className="text-ink-500 text-xs">Anyone with this link can view the final output.</p>
          </div>
        )}

        <a href="/" className="inline-block text-signal-blue text-sm hover:underline">
          ← New generation
        </a>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-[1fr_380px] gap-6">

      {/* ── LEFT: Draft preview ───────────────────────────────────────────── */}
      <div className="space-y-4">
        <div className="glass rounded-2xl overflow-hidden">
          <div className="px-5 py-4 border-b border-ink-800/60 flex items-center justify-between">
            <h2 className="text-sm font-semibold text-ink-100">Draft</h2>
            {critique && (
              <span className={`stage-pill text-[10px] ${
                critique.average_score >= 7.5
                  ? "bg-signal-green/10 text-signal-green border border-signal-green/20"
                  : "bg-signal-amber/10 text-signal-amber border border-signal-amber/20"
              }`}>
                avg {critique.average_score?.toFixed(1)}
              </span>
            )}
          </div>
          <div className="p-5">
            {mode === "edit" ? (
              isThread ? (
                <TweetEditor tweets={draft} onSave={human_edits => submit({ decision: "edit", human_edits })} />
              ) : (
                <EssayEditor text={draft} onSave={human_edits => submit({ decision: "edit", human_edits })} />
              )
            ) : (
              <ThreadPreview draft={draft} />
            )}
          </div>
        </div>

        {/* Sources */}
        {sources.length > 0 && (
          <div className="glass rounded-2xl overflow-hidden">
            <button
              onClick={() => setSourcesOpen(v => !v)}
              className="w-full px-5 py-4 flex items-center justify-between text-sm text-ink-300 hover:text-ink-100 transition-colors"
            >
              <span className="font-medium">Sources ({sources.length})</span>
              <span className="text-ink-500">{sourcesOpen ? "▲" : "▼"}</span>
            </button>
            {sourcesOpen && (
              <div className="px-5 pb-5 space-y-2 animate-fade-in">
                {sources.map((s, i) => (
                  <div key={i} className="flex items-start justify-between gap-3">
                    <a href={s.url} target="_blank" rel="noreferrer"
                      className="text-signal-blue hover:underline text-xs truncate flex-1">
                      {s.title || s.url}
                    </a>
                    <span className="text-ink-500 text-xs font-mono shrink-0">{s.credibility_score?.toFixed(1)}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Trace */}
        {trace.length > 0 && (
          <div className="glass rounded-2xl overflow-hidden">
            <button
              onClick={() => setTraceOpen(v => !v)}
              className="w-full px-5 py-4 flex items-center justify-between text-sm text-ink-300 hover:text-ink-100 transition-colors"
            >
              <span className="font-medium">Agent reasoning ({trace.length} events)</span>
              <span className="text-ink-500">{traceOpen ? "▲" : "▼"}</span>
            </button>
            {traceOpen && (
              <div className="px-5 pb-5 space-y-1.5 max-h-48 overflow-y-auto animate-fade-in">
                {trace.map((t, i) => (
                  <div key={i} className="text-xs font-mono text-ink-500 flex gap-2">
                    <span className="text-ink-400 shrink-0">[{t.node}]</span>
                    <span>{t.model}</span>
                    <span className="text-ink-600">— {t.duration_ms?.toFixed(0)}ms</span>
                    {t.detail && <span className="text-ink-600 truncate">— {t.detail}</span>}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>

      {/* ── RIGHT: Decision panel ─────────────────────────────────────────── */}
      <div className="space-y-4 lg:sticky lg:top-20 self-start">

        {/* Scorecard */}
        {critique && (
          <div className="glass rounded-2xl p-5 space-y-3">
            <h3 className="text-xs font-semibold text-ink-400 uppercase tracking-wide">Critic Scorecard</h3>
            <ScoreBar label="Hook Strength"    value={critique.hook_strength} />
            <ScoreBar label="Voice Match"      value={critique.voice_match} />
            <ScoreBar label="Insight Density"  value={critique.insight_density} />
            <ScoreBar label="Clarity"          value={critique.clarity} />
            <div className="pt-2 border-t border-ink-800 flex justify-between items-center">
              <span className="text-xs text-ink-500">Overall</span>
              <span className={`text-lg font-bold font-mono ${
                critique.average_score >= 7.5 ? "text-signal-green" : "text-signal-amber"
              }`}>
                {critique.average_score?.toFixed(1)}/10
              </span>
            </div>
            {/* Flags */}
            {(critique.hallucination_flags?.length > 0 || critique.cliche_flags?.length > 0) && (
              <div className="pt-2 border-t border-ink-800 space-y-1">
                {critique.hallucination_flags?.slice(0,3).map((f,i) => (
                  <p key={i} className="text-xs text-signal-red/80 flex gap-1.5 items-start">
                    <span className="shrink-0">⚠</span>
                    <span className="break-words">{typeof f === "object" ? f.claim : f}</span>
                  </p>
                ))}
                {critique.cliche_flags?.slice(0,3).map((f,i) => (
                  <p key={i} className="text-xs text-signal-amber/70 flex gap-1.5 items-start">
                    <span className="shrink-0">◈</span>
                    <span className="break-words">{typeof f === "object" ? f.phrase : f}</span>
                  </p>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Decision */}
        <div className="glass rounded-2xl p-5 space-y-3">
          <h3 className="text-xs font-semibold text-ink-400 uppercase tracking-wide">Your Decision</h3>

          {/* APPROVE */}
          <div className="rounded-xl border border-signal-green/20 bg-signal-green/5 p-4 space-y-2">
            <div>
              <p className="text-signal-green font-semibold text-sm">Approve</p>
              <p className="text-ink-500 text-xs mt-0.5">Looks good — format and save</p>
            </div>
            <button
              disabled={submitting || mode === "edit" || mode === "reject"}
              onClick={() => submit({ decision: "approve" })}
              className="w-full py-2.5 bg-signal-green/20 hover:bg-signal-green/30 border border-signal-green/30 text-signal-green font-semibold text-sm rounded-lg transition-colors disabled:opacity-40"
            >
              {submitting ? "…" : "Approve →"}
            </button>
          </div>

          {/* EDIT */}
          <div className="rounded-xl border border-signal-amber/20 bg-signal-amber/5 p-4 space-y-2">
            <div>
              <p className="text-signal-amber font-semibold text-sm">Edit</p>
              <p className="text-ink-500 text-xs mt-0.5">Good idea, refine the wording</p>
            </div>
            {mode !== "edit" ? (
              <button
                onClick={() => setMode("edit")}
                className="w-full py-2.5 bg-signal-amber/15 hover:bg-signal-amber/25 border border-signal-amber/25 text-signal-amber font-semibold text-sm rounded-lg transition-colors"
              >
                Open Editor
              </button>
            ) : (
              <button
                onClick={() => setMode(null)}
                className="w-full py-2 text-xs text-ink-400 hover:text-ink-200 transition-colors"
              >
                Cancel edit
              </button>
            )}
          </div>

          {/* REJECT */}
          <div className="rounded-xl border border-signal-red/20 bg-signal-red/5 p-4 space-y-2">
            <div>
              <p className="text-signal-red font-semibold text-sm">Reject + Feedback</p>
              <p className="text-ink-500 text-xs mt-0.5">Re-generate with your direction</p>
            </div>
            {mode !== "reject" ? (
              <button
                onClick={() => setMode("reject")}
                className="w-full py-2.5 bg-signal-red/15 hover:bg-signal-red/25 border border-signal-red/25 text-signal-red font-semibold text-sm rounded-lg transition-colors"
              >
                Reject
              </button>
            ) : (
              <div className="space-y-2">
                <textarea
                  rows={4}
                  className="va-input w-full px-3 py-2 text-sm resize-none"
                  placeholder="e.g. Too academic. Write as if explaining to a smart friend. Focus on implications for founders, not researchers."
                  value={rejectionNote}
                  onChange={e => setNote(e.target.value)}
                />
                <button
                  disabled={submitting || !rejectionNote.trim()}
                  onClick={() => submit({ decision: "reject", rejection_note: rejectionNote })}
                  className="w-full py-2.5 bg-signal-red/20 hover:bg-signal-red/30 border border-signal-red/30 text-signal-red font-semibold text-sm rounded-lg transition-colors disabled:opacity-40"
                >
                  {submitting ? "Submitting…" : "Re-generate →"}
                </button>
                <button onClick={() => setMode(null)} className="w-full py-1 text-xs text-ink-500 hover:text-ink-300 transition-colors">
                  Cancel
                </button>
              </div>
            )}
          </div>

          {error && (
            <p className="text-signal-red text-xs">{error}</p>
          )}
        </div>
      </div>
    </div>
  );
}