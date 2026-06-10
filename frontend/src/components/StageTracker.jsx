"use client";
import { useMemo } from "react";
import Link from "next/link";

// Pipeline stages — must match backend PIPELINE_NODES in api/routers/generate.py
const STAGES = [
  { id: "input_router",   label: "Parsing Input",    icon: "⊙",  color: "text-ink-300" },
  { id: "researcher",     label: "Researching",      icon: "⬡",  color: "text-signal-blue" },
  { id: "analyst",        label: "Analysing Claims", icon: "◎",  color: "text-signal-purple" },
  { id: "angle_finder",   label: "Finding Angle",    icon: "◉",  color: "text-signal-amber" },
  { id: "writer",         label: "Writing",          icon: "✦",  color: "text-signal-blue" },
  { id: "critic",         label: "Critiquing",       icon: "◈",  color: "text-signal-red" },
  { id: "validator_gate", label: "Human Review",     icon: "⊛",  color: "text-signal-amber" },
  { id: "formatter",      label: "Formatting",       icon: "◻",  color: "text-signal-teal" },
  { id: "distributor",    label: "Saving",           icon: "◆",  color: "text-signal-green" },
];

function getStageStatus(stageId, completedNodes, activeNode, phase) {
  if (completedNodes.has(stageId)) return "done";
  if (activeNode === stageId)       return "active";
  if (phase === "awaiting" && stageId === "validator_gate") return "waiting";
  return "pending";
}

function ScoreChip({ label, value }) {
  const color =
    value >= 8 ? "text-signal-green bg-signal-green/10" :
    value >= 6 ? "text-signal-amber bg-signal-amber/10" :
    "text-signal-red bg-signal-red/10";
  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-xs font-mono font-medium ${color}`}>
      {label} {value?.toFixed(1)}
    </span>
  );
}

export default function StageTracker({ events, phase, runId }) {
  const { completedNodes, activeNode, critique, iterationCount, awaitingData } = useMemo(() => {
    const completed   = new Set();
    let active        = null;
    let critique      = null;
    let iterationCount = 0;
    let awaitingData  = null;

    for (const evt of events) {
      if (evt.type === "node_start")    active = evt.data.node;
      if (evt.type === "node_complete") {
        completed.add(evt.data.node);
        active = null;
        // Extract iteration from detail
        if (evt.data.node === "writer") {
          const m = (evt.data.detail || "").match(/iteration=(\d+)/);
          if (m) iterationCount = parseInt(m[1]);
        }
      }
      if (evt.type === "awaiting_validation") {
        completed.add("critic");
        awaitingData = evt.data;
        critique     = evt.data.critique;
      }
    }

    return { completedNodes: completed, activeNode: active, critique, iterationCount, awaitingData };
  }, [events]);

  return (
    <div className="glass rounded-2xl overflow-hidden animate-fade-in">
      <div className="px-6 py-4 border-b border-ink-800/60 flex items-center justify-between">
        <h2 className="text-sm font-semibold text-ink-100 tracking-tight">Pipeline</h2>
        <div className="flex items-center gap-2">
          {iterationCount > 1 && (
            <span className="stage-pill bg-signal-amber/10 text-signal-amber border border-signal-amber/20">
              Iteration {iterationCount}
            </span>
          )}
          <span className={`stage-pill ${
            phase === "running"  ? "bg-signal-blue/10 text-signal-blue border border-signal-blue/20" :
            phase === "awaiting" ? "bg-signal-amber/10 text-signal-amber border border-signal-amber/20" :
            phase === "done"     ? "bg-signal-green/10 text-signal-green border border-signal-green/20" :
            "bg-ink-700 text-ink-400"
          }`}>
            {phase === "running"  && "Running"}
            {phase === "awaiting" && "Awaiting Review"}
            {phase === "done"     && "Complete"}
            {phase === "error"    && "Error"}
          </span>
        </div>
      </div>

      {/* Stage list */}
      <div className="px-6 py-5">
        <div className="relative">
          {/* Vertical connector line */}
          <div className="absolute left-[11px] top-5 bottom-5 w-px bg-ink-800" />

          <div className="space-y-3">
            {STAGES.map((stage, i) => {
              const status = getStageStatus(stage.id, completedNodes, activeNode, phase);

              return (
                <div key={stage.id} className={`relative flex items-center gap-4 transition-all duration-300 ${
                  status === "pending" ? "opacity-30" : "opacity-100"
                }`}>
                  {/* Node dot */}
                  <div className={`relative z-10 w-6 h-6 rounded-full flex items-center justify-center text-xs flex-shrink-0 transition-all duration-500 ${
                    status === "done"    ? "bg-signal-green/20 text-signal-green border border-signal-green/30" :
                    status === "active"  ? "bg-signal-blue/20 text-signal-blue border border-signal-blue/40 shadow-[0_0_12px_rgba(59,130,246,0.3)]" :
                    status === "waiting" ? "bg-signal-amber/20 text-signal-amber border border-signal-amber/30" :
                    "bg-ink-800 text-ink-600 border border-ink-700"
                  }`}>
                    {status === "done"   ? "✓" :
                     status === "active" ? <span className="animate-pulse-dot">●</span> :
                     stage.icon}
                  </div>

                  {/* Label */}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className={`text-sm font-medium ${
                        status === "done"    ? "text-ink-200" :
                        status === "active"  ? "text-white" :
                        status === "waiting" ? "text-signal-amber" :
                        "text-ink-600"
                      }`}>
                        {stage.label}
                      </span>
                      {status === "active" && (
                        <span className="text-xs text-ink-500 animate-pulse">running…</span>
                      )}
                      {status === "waiting" && (
                        <span className="text-xs text-signal-amber/70">waiting for you</span>
                      )}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Critique scorecard — shown when awaiting */}
      {phase === "awaiting" && critique && (
        <div className="px-6 pb-5 border-t border-ink-800/60 pt-5 animate-slide-up space-y-4">
          <div>
            <p className="text-xs font-semibold text-ink-400 uppercase tracking-wide mb-3">Critic Scores</p>
            <div className="grid grid-cols-2 gap-2">
              {[
                { label: "Hook",    value: critique.hook_strength },
                { label: "Voice",   value: critique.voice_match },
                { label: "Density", value: critique.insight_density },
                { label: "Clarity", value: critique.clarity },
              ].map(({ label, value }) => (
                <div key={label} className="bg-ink-800/50 rounded-lg px-3 py-2">
                  <div className="flex justify-between items-center mb-1.5">
                    <span className="text-xs text-ink-400">{label}</span>
                    <span className="text-xs font-mono text-ink-100">{value?.toFixed(1)}</span>
                  </div>
                  <div className="h-1 bg-ink-700 rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all duration-700 ${
                        value >= 8 ? "bg-signal-green" :
                        value >= 6 ? "bg-signal-amber" :
                        "bg-signal-red"
                      }`}
                      style={{ width: `${((value - 1) / 9) * 100}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Flags */}
          {(critique.hallucination_flags?.length > 0 || critique.cliche_flags?.length > 0) && (
            <div className="space-y-1.5">
              {critique.hallucination_flags?.slice(0, 2).map((f, i) => (
                <p key={i} className="text-xs text-signal-red/80 flex gap-1.5">
                  <span>⚠</span><span className="truncate">{typeof f === "object" ? f.claim : f}</span>
                </p>
              ))}
              {critique.cliche_flags?.slice(0, 2).map((f, i) => (
                <p key={i} className="text-xs text-signal-amber/70 flex gap-1.5">
                  <span>◈</span><span className="truncate">{typeof f === "object" ? f.phrase : f}</span>
                </p>
              ))}
            </div>
          )}

          {/* CTA */}
          <Link
            href={`/validate/${runId}`}
            className="block w-full py-2.5 bg-signal-amber hover:bg-amber-400 text-ink-950 font-bold text-sm rounded-xl text-center transition-colors"
          >
            Review &amp; Approve Draft →
          </Link>
        </div>
      )}

      {/* Done state */}
      {phase === "done" && (
        <div className="px-6 pb-5 border-t border-ink-800/60 pt-5 animate-slide-up">
          <p className="text-signal-green text-sm font-medium">✓ Pipeline complete</p>
          <p className="text-ink-500 text-xs mt-1">Output formatted and saved.</p>
        </div>
      )}

      {/* Error state */}
      {phase === "error" && (
        <div className="px-6 pb-5 border-t border-ink-800/60 pt-5 animate-slide-up">
          {(() => {
            const errEvt = [...events].reverse().find(e => e.type === "error");
            const msg = errEvt?.data?.error || "An unexpected error occurred.";
            return (
              <>
                <p className="text-signal-red text-sm font-medium">✗ Pipeline error</p>
                <p className="text-ink-400 text-xs mt-1 font-mono break-all">{msg}</p>
              </>
            );
          })()}
        </div>
      )}
    </div>
  );
}
