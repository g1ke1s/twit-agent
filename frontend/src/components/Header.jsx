"use client";
import Link from "next/link";

export default function Header({ shareMode = false }) {
  return (
    <header className="border-b border-ink-800/60 backdrop-blur-md sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-6 h-14 flex items-center justify-between">
        <Link href="/" className="flex items-center gap-2.5 group">
          <span className="w-6 h-6 rounded-md bg-gradient-to-br from-signal-blue to-signal-teal flex items-center justify-center text-xs font-bold text-white">
            V
          </span>
          <span className="font-display font-semibold text-sm tracking-tight text-ink-100 group-hover:text-white transition-colors">
            Voice Agent
          </span>
        </Link>

        {shareMode && (
          <div className="stage-pill bg-ink-700 text-ink-300">
            Read-only view
          </div>
        )}

        {!shareMode && (
          <nav className="flex items-center gap-4 text-xs text-ink-400">
            <Link href="/" className="hover:text-ink-100 transition-colors">Generate</Link>
            <a
              href="https://github.com/yourname/voice-agent"
              target="_blank"
              rel="noreferrer"
              className="hover:text-ink-100 transition-colors"
            >
              GitHub
            </a>
          </nav>
        )}
      </div>
    </header>
  );
}
