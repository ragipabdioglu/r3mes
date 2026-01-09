"use client";

import { LayoutDashboard, Globe, Settings, Vote, Coins } from "lucide-react";

interface SidebarProps {
  activeView: "console" | "explorer" | "governance" | "staking";
  setActiveView: (view: "console" | "explorer" | "governance" | "staking") => void;
}

export default function Sidebar({ activeView, setActiveView }: SidebarProps) {
  return (
    <aside className="w-72 bg-slate-950/80 border-r border-slate-800/80 backdrop-blur-xl">
      <div className="p-6">
        <h2 className="text-xs font-semibold tracking-[0.25em] text-slate-500 uppercase mb-6">
          Control Surface
        </h2>
        <nav className="space-y-2">
          <button
            onClick={() => setActiveView("console")}
            className={`w-full flex items-center space-x-3 px-4 py-3 rounded-xl transition-colors ${
              activeView === "console"
                ? "bg-gradient-to-r from-cyan-500 to-sky-500 text-slate-950 shadow-[0_0_20px_rgba(56,189,248,0.6)]"
                : "text-slate-300 hover:bg-slate-900/70"
            }`}
          >
            <LayoutDashboard className="w-5 h-5" />
            <span>Miner Console</span>
          </button>
          <button
            onClick={() => setActiveView("explorer")}
            className={`w-full flex items-center space-x-3 px-4 py-3 rounded-xl transition-colors ${
              activeView === "explorer"
                ? "bg-gradient-to-r from-purple-500 to-cyan-400 text-slate-950 shadow-[0_0_20px_rgba(168,85,247,0.55)]"
                : "text-slate-300 hover:bg-slate-900/70"
            }`}
          >
            <Globe className="w-5 h-5" />
            <span>Network Explorer</span>
          </button>
          <button
            onClick={() => setActiveView("governance")}
            className={`w-full flex items-center space-x-3 px-4 py-3 rounded-xl transition-colors ${
              activeView === "governance"
                ? "bg-gradient-to-r from-green-500 to-emerald-400 text-slate-950 shadow-[0_0_20px_rgba(34,197,94,0.55)]"
                : "text-slate-300 hover:bg-slate-900/70"
            }`}
          >
            <Vote className="w-5 h-5" />
            <span>Governance</span>
          </button>
          <button
            onClick={() => setActiveView("staking")}
            className={`w-full flex items-center space-x-3 px-4 py-3 rounded-xl transition-colors ${
              activeView === "staking"
                ? "bg-gradient-to-r from-yellow-500 to-orange-400 text-slate-950 shadow-[0_0_20px_rgba(234,179,8,0.55)]"
                : "text-slate-300 hover:bg-slate-900/70"
            }`}
          >
            <Coins className="w-5 h-5" />
            <span>Staking</span>
          </button>
          <button
            className="w-full flex items-center space-x-3 px-4 py-3 rounded-xl text-slate-400 hover:bg-slate-900/70 transition-colors"
          >
            <Settings className="w-5 h-5" />
            <span>Settings</span>
          </button>
        </nav>
      </div>
    </aside>
  );
}

