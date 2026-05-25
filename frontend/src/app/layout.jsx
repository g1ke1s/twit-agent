import "./globals.css";
import { Sora, JetBrains_Mono } from "next/font/google";

const sora = Sora({
  subsets: ["latin"],
  variable: "--font-body",
  display: "swap",
});

const jetbrains = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
  display: "swap",
});

export const metadata = {
  title: "Voice Agent",
  description: "Agentic content generation in your authentic voice",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en" className={`${sora.variable} ${jetbrains.variable}`}>
      <body className="bg-ink-950 text-ink-100 min-h-screen antialiased font-sans">
        <div className="fixed inset-0 pointer-events-none">
          <div className="absolute top-0 left-1/4 w-96 h-96 bg-signal-blue/5 rounded-full blur-3xl" />
          <div className="absolute bottom-1/4 right-1/4 w-64 h-64 bg-signal-teal/4 rounded-full blur-3xl" />
        </div>
        <div className="relative z-10">
          {children}
        </div>
      </body>
    </html>
  );
}
