/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        syne: ["Syne", "sans-serif"],
        mono: ["JetBrains Mono", "monospace"],
      },
      colors: {
        // Engram design tokens
        bg: {
          primary: "#0d0d0f",    // Near-black canvas
          secondary: "#13131a",  // Panel background
          tertiary: "#1a1a24",   // Card background
          hover: "#1f1f2e",
        },
        accent: {
          yellow: "#d4f000",     // Acid yellow — primary action
          blue: "#3b82f6",
          red: "#ef4444",
          green: "#22c55e",
          amber: "#f59e0b",
        },
        memory: {
          active: "#22c55e",
          pending: "#f59e0b",
          deprecated: "#ef4444",
          archived: "#4b5563",
          procedural: "#a78bfa",
          episodic: "#60a5fa",
          semantic: "#34d399",
        },
        border: "#2a2a3a",
        text: {
          primary: "#f0f0f4",
          secondary: "#8888a0",
          muted: "#555567",
        },
      },
      backgroundImage: {
        "gradient-radial": "radial-gradient(var(--tw-gradient-stops))",
      },
    },
  },
  plugins: [],
};
