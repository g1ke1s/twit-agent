"use client";

function CharMeter({ count }) {
  const pct   = Math.min((count / 280) * 100, 100);
  const color = count > 270 ? "bg-signal-red" : count > 240 ? "bg-signal-amber" : "bg-signal-blue";
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-0.5 bg-ink-700 rounded-full overflow-hidden">
        <div className={`h-full ${color} transition-all duration-500`} style={{ width: `${pct}%` }} />
      </div>
      <span className={`text-xs font-mono tabular-nums ${count > 270 ? "text-signal-red" : "text-ink-500"}`}>
        {count}/280
      </span>
    </div>
  );
}

function TweetCard({ tweet, index }) {
  return (
    <div className="relative animate-slide-up" style={{ animationDelay: `${index * 60}ms` }}>
      <div className="bg-ink-800/50 hover:bg-ink-800/70 border border-ink-700/50 rounded-xl p-4 transition-colors group">
        <div className="flex items-start gap-3">
          {/* Position indicator */}
          <div className="flex-shrink-0 w-7 h-7 rounded-full bg-ink-700/80 flex items-center justify-center">
            <span className="text-xs font-mono text-ink-400">{tweet.position}</span>
          </div>

          <div className="flex-1 min-w-0 space-y-2.5">
            {/* Badges */}
            {(tweet.is_hook || tweet.is_cta) && (
              <div className="flex gap-1.5">
                {tweet.is_hook && (
                  <span className="stage-pill bg-signal-blue/10 text-signal-blue border border-signal-blue/20 text-[9px]">
                    hook
                  </span>
                )}
                {tweet.is_cta && (
                  <span className="stage-pill bg-signal-purple/10 text-signal-purple border border-signal-purple/20 text-[9px]">
                    cta
                  </span>
                )}
              </div>
            )}

            {/* Tweet text */}
            <p className="text-ink-100 text-sm leading-relaxed whitespace-pre-wrap">{tweet.text}</p>

            {/* Char meter */}
            <CharMeter count={tweet.char_count} />
          </div>
        </div>
      </div>

      {/* Thread connector */}
      {tweet.position < 999 && (
        <div className="absolute left-[23px] -bottom-1.5 w-px h-3 bg-ink-700 z-10" />
      )}
    </div>
  );
}

function EssayPreview({ text }) {
  return (
    <div className="prose prose-sm prose-invert max-w-none">
      <div className="text-ink-200 text-sm leading-relaxed whitespace-pre-wrap font-sans">{text}</div>
    </div>
  );
}

export default function ThreadPreview({ draft }) {
  if (!draft) return null;

  if (Array.isArray(draft) && draft.length > 0) {
    return (
      <div className="space-y-1.5">
        {draft.map((tweet, i) => (
          <TweetCard key={tweet.position || i} tweet={tweet} index={i} />
        ))}
      </div>
    );
  }

  if (typeof draft === "string" && draft) {
    return (
      <div className="bg-ink-800/50 border border-ink-700/50 rounded-xl p-5">
        <EssayPreview text={draft} />
      </div>
    );
  }

  return null;
}
