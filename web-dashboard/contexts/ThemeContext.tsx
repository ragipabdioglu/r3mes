"use client";

import { createContext, useContext, useEffect, useState, ReactNode } from "react";

type Theme = "dark" | "light" | "system";

interface ThemeContextType {
  theme: Theme;
  resolvedTheme: "dark" | "light";
  setTheme: (theme: Theme) => void;
  toggleTheme: () => void;
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [theme, setThemeState] = useState<Theme>("light");
  const [resolvedTheme, setResolvedTheme] = useState<"dark" | "light">("light");

  useEffect(() => {
    // Load theme from localStorage, default to "light" if not set
    try {
      const savedTheme = localStorage.getItem("r3mes_theme") as Theme | null;
      if (savedTheme && (savedTheme === "light" || savedTheme === "dark" || savedTheme === "system")) {
        setThemeState(savedTheme);
      } else {
        // Default to light mode if no saved theme or invalid theme
        setThemeState("light");
        localStorage.setItem("r3mes_theme", "light");
      }
    } catch (e) {
      // If localStorage is not available, default to light mode
      setThemeState("light");
    }
  }, []);

  useEffect(() => {
    // Determine resolved theme
    let resolved: "dark" | "light" = "light"; // Default to light mode
    
    if (theme === "system") {
      if (typeof window !== "undefined") {
        resolved = window.matchMedia("(prefers-color-scheme: light)").matches
          ? "light"
          : "dark";
      }
    } else {
      resolved = theme;
    }

    setResolvedTheme(resolved);

    // Apply theme to document
    if (typeof window !== "undefined") {
      const root = document.documentElement;
      if (resolved === "light") {
        root.classList.remove("dark");
      } else {
        root.classList.add("dark");
      }
    }
  }, [theme]);

  useEffect(() => {
    // Listen for system theme changes
    if (theme === "system" && typeof window !== "undefined") {
      const mediaQuery = window.matchMedia("(prefers-color-scheme: light)");
      const handleChange = (e: MediaQueryListEvent) => {
        setResolvedTheme(e.matches ? "light" : "dark");
      };

      mediaQuery.addEventListener("change", handleChange);
      return () => mediaQuery.removeEventListener("change", handleChange);
    }
  }, [theme]);

  const setTheme = (newTheme: Theme) => {
    setThemeState(newTheme);
    localStorage.setItem("r3mes_theme", newTheme);
  };

  const toggleTheme = () => {
    if (theme === "dark") {
      setTheme("light");
    } else if (theme === "light") {
      setTheme("dark");
    } else {
      // If system, toggle to opposite of current resolved theme
      setTheme(resolvedTheme === "dark" ? "light" : "dark");
    }
  };

  return (
    <ThemeContext.Provider value={{ theme, resolvedTheme, setTheme, toggleTheme }}>
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme() {
  const context = useContext(ThemeContext);
  if (context === undefined) {
    throw new Error("useTheme must be used within a ThemeProvider");
  }
  return context;
}

