"use client";
import { useState } from "react";
import GeneratePanel from "../components/GeneratePanel";
import StageTracker from "../components/StageTracker";
import Header from "../components/Header";

export default function Home() {
  const [runId, setRunId]       = useState(null);
  const [events, setEvents]     = useState([]);
  const [phase, setPhase]       = useState("idle"); // idle | running | awaiting | done | error

  function handleEvent(evt) {
    // ALL raw events go to console — not shown in UI
    console.group(`[Voice Agent] ${evt.type}`);
    console.log(evt.data);
    console.groupEnd();

    // Update stage tracker state
    if (evt.type === "start") {
      setRunId(evt.data.run_id);
      setPhase("running");
    }
    if (evt.type === "awaiting_validation") setPhase("awaiting");
    if (evt.type === "complete")            setPhase("done");
    if (evt.type === "error")               setPhase("error");

    setEvents(prev => [...prev, evt]);
  }

  return (
    <div className="min-h-screen flex flex-col">
      <Header />
      <main className="flex-1 max-w-5xl mx-auto w-full px-6 py-10 flex flex-col gap-8">
        <GeneratePanel
          onEvent={handleEvent}
          phase={phase}
          runId={runId}
          onReset={() => { setPhase("idle"); setRunId(null); setEvents([]); }}
        />
        {(phase !== "idle") && (
          <StageTracker events={events} phase={phase} runId={runId} />
        )}
      </main>
    </div>
  );
}
