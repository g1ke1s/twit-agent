"use client";
import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import Header from "../../../components/Header";
import ValidatorPanel from "../../../components/ValidatorPanel";
import { apiJson } from "../../../lib/api";

export default function SharePage() {
  const params = useParams();
  const { runId } = params;
  const [runData, setRunData] = useState(null);
  const [error, setError]     = useState(null);

  useEffect(() => {
    let cancelled = false;
    async function poll() {
      try {
        const data = await apiJson(`/api/runs/${runId}`);
        if (!cancelled) {
          setRunData(data);
          if (!data.awaiting_validation && data.status !== "complete") {
            setTimeout(poll, 2500);
          }
        }
      } catch (e) {
        if (!cancelled) setError(e.message);
      }
    }
    poll();
    return () => { cancelled = true; };
  }, [runId]);

  return (
    <div className="min-h-screen flex flex-col">
      <Header />
      <main className="flex-1 max-w-7xl mx-auto w-full px-6 py-10">
        {error ? (
          <div className="text-signal-red text-sm">{error}</div>
        ) : !runData ? (
          <div className="flex items-center gap-3 text-ink-400 text-sm animate-pulse">
            <span className="w-2 h-2 rounded-full bg-signal-blue animate-pulse-dot" />
            Loading run…
          </div>
        ) : (
          <ValidatorPanel runId={runId} runData={runData} />
        )}
      </main>
    </div>
  );
}
