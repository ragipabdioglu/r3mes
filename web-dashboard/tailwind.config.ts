import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: 'class',
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "var(--bg-primary)",
        foreground: "var(--text-primary)",
        'bg-primary': "var(--bg-primary)",
        'bg-secondary': "var(--bg-secondary)",
        'bg-tertiary': "var(--bg-tertiary)",
        'text-primary': "var(--text-primary)",
        'text-secondary': "var(--text-secondary)",
        'text-muted': "var(--text-muted)",
        primary: {
          DEFAULT: "var(--accent-primary)",
          hover: "var(--accent-primary-hover)",
        },
        secondary: {
          DEFAULT: "var(--accent-secondary)",
        },
        success: "var(--success)",
        warning: "var(--warning)",
        error: "var(--error)",
        info: "var(--info)",
        border: "var(--border-color)",
      },
      keyframes: {
        scroll: {
          "0%": { transform: "translateX(0)" },
          "100%": { transform: "translateX(-50%)" },
        },
        fadeIn: {
          "0%": { opacity: "0", transform: "translateY(10px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
      },
      animation: {
        scroll: "scroll 30s linear infinite",
        fadeIn: "fadeIn 0.5s ease-out",
      },
    },
  },
  plugins: [],
};
export default config;

