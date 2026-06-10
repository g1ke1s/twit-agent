"use client";
import { useState, useRef } from "react";
import { getApiBase } from "@/lib/api";

const OUTPUT_TYPES = [
  { value: "x_thread",          label: "X Thread",            desc: "6–10 tweets, data-grounded" },
  { value: "quote_rt",          label: "Quote RT",            desc: "Insightful reply to a tweet" },
  { value: "essay",             label: "Essay",               desc: "800–3000 words, researched" },
  { value: "analysis",          label: "Analysis",            desc: "Structured tech report" },
  { value: "strategic_narrative", label: "Strategic Narrative", desc: "Socratic, question-driven" },
];

const TONES = [
  { value: "analytical",   label: "Analytical" },
  { value: "provocative",  label: "Provocative" },
  { value: "educational",  label: "Educational" },
];

const INPUT_TABS = [
  { id: "text",   label: "Text / URL" },
  { id: "file",   label: "File Upload" },
  { id: "voice",  label: "Voice Archive" },
];

export default function GeneratePanel({ onEvent, phase, runId, onReset }) {
  const [activeTab, setActiveTab]         = useState("text");
  const [input, setInput]                 = useState("");
  const [outputType, setOutputType]       = useState("x_thread");
  const [tone, setTone]                   = useState("analytical");
  const [instruction, setInstruction]     = useState("");
  const [userId, setUserId]               = useState("default");
  const [uploadedText, setUploadedText]   = useState("");
  const [uploadedName, setUploadedName]   = useState("");
  const [voiceFile, setVoiceFile]         = useState(null);
  const [voiceResult, setVoiceResult]     = useState(null);
  const [uploading, setUploading]         = useState(false);
  const [uploadError, setUploadError]     = useState("");
  const [running, setRunning]             = useState(false);
  const [error, setError]                 = useState("");
  const fileRef  = useRef(null);
  const voiceRef = useRef(null);

  const isRunning = phase === "running";
  const isAwaiting = phase === "awaiting";
  const isDone = phase === "done" || phase === "error";
  const canGenerate = phase === "idle";

  // ---------------------------------------------------------------------------
  // File upload for content
  // ---------------------------------------------------------------------------
  async function handleContentFile(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploading(true);
    setUploadError("");
    const fd = new FormData();
    fd.append("file", file);
    try {
      const res = await fetch(`${getApiBase()}/api/upload/content-file`, { method: "POST", body: fd });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setUploadedText(data.text);
      setUploadedName(data.filename);
      console.log("[Voice Agent] File parsed:", data.filename, `${data.chars} chars, ${data.paragraphs} paragraphs`);
    } catch (err) {
      setUploadError(err.message);
    } finally {
      setUploading(false);
    }
  }

  // ---------------------------------------------------------------------------
  // Voice archive upload
  // ---------------------------------------------------------------------------
  async function handleVoiceUpload() {
    if (!voiceFile || !userId) return;
    setUploading(true);
    setUploadError("");
    const fd = new FormData();
    fd.append("user_id", userId);
    fd.append("file", voiceFile);
    try {
      const res = await fetch(`${getApiBase()}/api/memory/voice-archive`, { method: "POST", body: fd });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setVoiceResult(data);
      console.log("[Voice Agent] Voice profile built:", data);
    } catch (err) {
      setUploadError(err.message);
    } finally {
      setUploading(false);
    }
  }

  // ---------------------------------------------------------------------------
  // SSE generate
  // ---------------------------------------------------------------------------
  async function handleGenerate() {
    const rawInput = activeTab === "file"
      ? (instruction || uploadedName || "Analyse this document")
      : input.trim();

    if (!rawInput && !uploadedText) return;
    setRunning(true);
    setError("");

    const body = {
      raw_input: rawInput || "Analyse the uploaded content",
      output_type: outputType,
      user_instruction: instruction || undefined,
      target_tone: tone,
      user_id: userId,
      ...(uploadedText ? { uploaded_content: uploadedText, input_type: "file_upload" } : {}),
    };

    console.log("[Voice Agent] Starting generation:", body);

    try {
      const res = await fetch(`${getApiBase()}/api/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);

      const reader  = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer    = "";
      let evtType   = null;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";
        for (const line of lines) {
          if (line.startsWith("event: "))     evtType = line.slice(7).trim();
          else if (line.startsWith("data: ") && evtType) {
            try { onEvent({ type: evtType, data: JSON.parse(line.slice(6)) }); } catch {}
            evtType = null;
          }
        }
      }
    } catch (err) {
      setError(err.message);
      onEvent({ type: "error", data: { error: err.message } });
    } finally {
      setRunning(false);
    }
  }

  if (isRunning || isAwaiting || isDone) {
    return (
      <div className="glass rounded-2xl p-6 flex items-center justify-between gap-4">
        <div>
          <p className="text-ink-100 font-medium text-sm">
            {isRunning  && "Pipeline running…"}
            {isAwaiting && "Draft ready for your review"}
            {phase === "done"  && "Run complete"}
            {phase === "error" && "Pipeline error — check logs"}
          </p>
          {runId && (
            <p className="text-ink-500 text-xs font-mono mt-0.5">{runId.slice(0,8)}…</p>
          )}
        </div>
        <div className="flex items-center gap-3">
          {isAwaiting && (
            <a
              href={`/validate/${runId}`}
              className="px-4 py-2 bg-signal-amber hover:bg-amber-400 text-ink-950 font-semibold text-sm rounded-lg transition-colors"
            >
              Review Draft →
            </a>
          )}
          {isDone && (
            <button
              onClick={onReset}
              className="px-4 py-2 bg-ink-700 hover:bg-ink-600 text-ink-100 text-sm rounded-lg transition-colors"
            >
              New run
            </button>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="glass rounded-2xl overflow-hidden">
      {/* Tab bar */}
      <div className="flex border-b border-ink-800/60">
        {INPUT_TABS.map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`px-5 py-3.5 text-xs font-semibold tracking-wide uppercase transition-colors relative ${
              activeTab === tab.id
                ? "text-ink-100"
                : "text-ink-500 hover:text-ink-300"
            }`}
          >
            {tab.label}
            {activeTab === tab.id && (
              <span className="absolute bottom-0 left-0 right-0 h-0.5 bg-signal-blue" />
            )}
          </button>
        ))}
      </div>

      <div className="p-6 space-y-5">
        {/* ── TEXT / URL TAB ── */}
        {activeTab === "text" && (
          <div>
            <label className="block text-xs font-semibold text-ink-400 uppercase tracking-wide mb-2">
              Input — URL, tweet, or topic
            </label>
            <textarea
              className="va-input w-full px-4 py-3 text-sm resize-none leading-relaxed"
              rows={4}
              placeholder={`https://x.com/... — Twitter/X URL to analyse\nhttps://... — any web article via Jina\nWrite a thread about AI replacing engineers`}
              value={input}
              onChange={e => setInput(e.target.value)}
            />
          </div>
        )}

        {/* ── FILE UPLOAD TAB ── */}
        {activeTab === "file" && (
          <div className="space-y-4">
            <label className="block text-xs font-semibold text-ink-400 uppercase tracking-wide mb-2">
              Upload document — PDF, DOCX, TXT, CSV, JSON
            </label>
            {!uploadedText ? (
              <div
                onClick={() => fileRef.current?.click()}
                className="border-2 border-dashed border-ink-700 hover:border-signal-blue/50 rounded-xl p-10 text-center cursor-pointer transition-colors group"
              >
                <div className="text-2xl mb-2">📄</div>
                <p className="text-ink-300 text-sm font-medium group-hover:text-ink-100 transition-colors">
                  {uploading ? "Parsing file…" : "Drop a file or click to browse"}
                </p>
                <p className="text-ink-500 text-xs mt-1">PDF · DOCX · TXT · CSV · JSON · max 10MB</p>
                <input ref={fileRef} type="file" className="hidden"
                  accept=".pdf,.docx,.txt,.csv,.json"
                  onChange={handleContentFile}
                />
              </div>
            ) : (
              <div className="bg-ink-800/60 rounded-xl p-4 flex items-start justify-between gap-3">
                <div>
                  <p className="text-ink-100 text-sm font-medium">{uploadedName}</p>
                  <p className="text-ink-400 text-xs mt-0.5">{uploadedText.length.toLocaleString()} characters extracted</p>
                </div>
                <button
                  onClick={() => { setUploadedText(""); setUploadedName(""); }}
                  className="text-ink-500 hover:text-signal-red text-xs transition-colors"
                >
                  ✕ Remove
                </button>
              </div>
            )}
            <div>
              <label className="block text-xs font-semibold text-ink-400 uppercase tracking-wide mb-2">
                Instruction for this document
              </label>
              <input
                className="va-input w-full px-4 py-2.5 text-sm"
                placeholder="e.g. Write a thread on the key findings. Focus on implications for founders."
                value={instruction}
                onChange={e => setInstruction(e.target.value)}
              />
            </div>
            {uploadError && <p className="text-signal-red text-xs">{uploadError}</p>}
          </div>
        )}

        {/* ── VOICE ARCHIVE TAB ── */}
        {activeTab === "voice" && (
          <div className="space-y-4">
            <p className="text-ink-400 text-sm leading-relaxed">
              Upload a collection of your own writing. The system will extract your voice
              fingerprint and use it to write content that sounds exactly like you.
            </p>
            <div>
              <label className="block text-xs font-semibold text-ink-400 uppercase tracking-wide mb-2">
                User ID
              </label>
              <input
                className="va-input w-full px-4 py-2.5 text-sm font-mono"
                value={userId}
                onChange={e => setUserId(e.target.value)}
              />
            </div>
            <div>
              <label className="block text-xs font-semibold text-ink-400 uppercase tracking-wide mb-2">
                Writing archive — PDF, DOCX, TXT, JSON
              </label>
              {!voiceFile ? (
                <div
                  onClick={() => voiceRef.current?.click()}
                  className="border-2 border-dashed border-ink-700 hover:border-signal-teal/50 rounded-xl p-8 text-center cursor-pointer transition-colors group"
                >
                  <div className="text-2xl mb-2">🎙️</div>
                  <p className="text-ink-300 text-sm font-medium group-hover:text-ink-100 transition-colors">
                    Upload your writing samples
                  </p>
                  <p className="text-ink-500 text-xs mt-1">One post per blank-line separator · 50+ samples recommended</p>
                  <input ref={voiceRef} type="file" className="hidden"
                    accept=".pdf,.docx,.txt,.csv,.json"
                    onChange={e => setVoiceFile(e.target.files?.[0] || null)}
                  />
                </div>
              ) : (
                <div className="bg-ink-800/60 rounded-xl p-4 flex items-center justify-between">
                  <p className="text-ink-100 text-sm">{voiceFile.name}</p>
                  <button onClick={() => setVoiceFile(null)} className="text-ink-500 hover:text-signal-red text-xs transition-colors">✕</button>
                </div>
              )}
            </div>
            {voiceFile && !voiceResult && (
              <button
                onClick={handleVoiceUpload}
                disabled={uploading}
                className="w-full py-2.5 bg-signal-teal/20 hover:bg-signal-teal/30 border border-signal-teal/30 text-signal-teal font-medium text-sm rounded-xl transition-colors disabled:opacity-50"
              >
                {uploading ? "Building voice profile…" : "Build Voice Profile →"}
              </button>
            )}
            {voiceResult && (
              <div className="bg-signal-teal/10 border border-signal-teal/20 rounded-xl p-4 space-y-2 animate-fade-in">
                <p className="text-signal-teal text-sm font-semibold">✓ Voice profile built</p>
                <p className="text-ink-400 text-xs">{voiceResult.samples_processed} samples · avg sentence {voiceResult.avg_sentence_length} words</p>
                <p className="text-ink-300 text-xs italic leading-relaxed">"{voiceResult.style_summary}"</p>
              </div>
            )}
            {uploadError && <p className="text-signal-red text-xs">{uploadError}</p>}
            <p className="text-ink-600 text-xs">After uploading, switch to the Text / URL tab to generate content.</p>
          </div>
        )}

        {/* ── SHARED CONTROLS ── */}
        {activeTab !== "voice" && (
          <>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-xs font-semibold text-ink-400 uppercase tracking-wide mb-2">Output type</label>
                <div className="grid grid-cols-1 gap-1.5">
                  {OUTPUT_TYPES.map(t => (
                    <button
                      key={t.value}
                      onClick={() => setOutputType(t.value)}
                      className={`text-left px-3 py-2 rounded-lg text-sm transition-all ${
                        outputType === t.value
                          ? "bg-signal-blue/15 border border-signal-blue/30 text-ink-100"
                          : "bg-ink-800/40 border border-transparent text-ink-400 hover:text-ink-200 hover:bg-ink-800"
                      }`}
                    >
                      <span className="font-medium">{t.label}</span>
                      <span className="text-ink-500 text-xs ml-2">{t.desc}</span>
                    </button>
                  ))}
                </div>
              </div>

              <div className="space-y-4">
                <div>
                  <label className="block text-xs font-semibold text-ink-400 uppercase tracking-wide mb-2">Tone</label>
                  <div className="flex gap-2">
                    {TONES.map(t => (
                      <button
                        key={t.value}
                        onClick={() => setTone(t.value)}
                        className={`flex-1 py-2 rounded-lg text-xs font-medium transition-all ${
                          tone === t.value
                            ? "bg-signal-teal/20 border border-signal-teal/30 text-signal-teal"
                            : "bg-ink-800/40 border border-transparent text-ink-400 hover:text-ink-200"
                        }`}
                      >
                        {t.label}
                      </button>
                    ))}
                  </div>
                </div>

                <div>
                  <label className="block text-xs font-semibold text-ink-400 uppercase tracking-wide mb-2">User ID</label>
                  <input
                    className="va-input w-full px-3 py-2 text-sm font-mono"
                    value={userId}
                    onChange={e => setUserId(e.target.value)}
                  />
                </div>

                <div>
                  <label className="block text-xs font-semibold text-ink-400 uppercase tracking-wide mb-2">Extra instruction</label>
                  <textarea
                    className="va-input w-full px-3 py-2 text-sm resize-none"
                    rows={3}
                    placeholder="Focus on implications for Series A founders…"
                    value={instruction}
                    onChange={e => setInstruction(e.target.value)}
                  />
                </div>
              </div>
            </div>

            {error && <p className="text-signal-red text-xs">{error}</p>}

            <button
              onClick={handleGenerate}
              disabled={running || (activeTab === "text" && !input.trim()) || (activeTab === "file" && !uploadedText)}
              className="w-full py-3 bg-signal-blue hover:bg-blue-400 disabled:bg-ink-700 disabled:text-ink-500 text-white font-semibold text-sm rounded-xl transition-all duration-200 tracking-wide"
            >
              {running ? "Streaming…" : "Run Pipeline →"}
            </button>
          </>
        )}
      </div>
    </div>
  );
}
