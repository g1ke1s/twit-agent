/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["var(--font-body)", "system-ui", "sans-serif"],
        display: ["var(--font-display)", "system-ui", "sans-serif"],
        mono: ["var(--font-mono)", "monospace"],
      },
      colors: {
        ink: {
          950: "#080A0F",
          900: "#0D1117",
          800: "#131920",
          700: "#1C2433",
          600: "#263040",
          500: "#3A4A5E",
          400: "#5A7089",
          300: "#8AA4BE",
          200: "#B8CDD9",
          100: "#E0EAF0",
          50:  "#F3F7FA",
        },
        signal: {
          blue:   "#3B82F6",
          teal:   "#14B8A6",
          amber:  "#F59E0B",
          red:    "#EF4444",
          green:  "#22C55E",
          purple: "#A855F7",
        },
      },
      animation: {
        "fade-in":       "fadeIn 0.4s ease forwards",
        "slide-up":      "slideUp 0.5s cubic-bezier(.16,1,.3,1) forwards",
        "pulse-dot":     "pulseDot 1.4s ease-in-out infinite",
        "shimmer":       "shimmer 2s linear infinite",
        "progress-fill": "progressFill 0.6s cubic-bezier(.16,1,.3,1) forwards",
      },
      keyframes: {
        fadeIn:       { from: { opacity: 0 }, to: { opacity: 1 } },
        slideUp:      { from: { opacity: 0, transform: "translateY(12px)" }, to: { opacity: 1, transform: "translateY(0)" } },
        pulseDot:     { "0%,100%": { opacity: 0.3, transform: "scale(0.8)" }, "50%": { opacity: 1, transform: "scale(1.2)" } },
        shimmer:      { from: { backgroundPosition: "-400px 0" }, to: { backgroundPosition: "400px 0" } },
        progressFill: { from: { width: "0%" }, to: { width: "var(--target-width)" } },
      },
    },
  },
  plugins: [],
};
