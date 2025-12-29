"use client";

import { useState, useEffect, useMemo } from "react";
import { Search, X, FileText } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";

interface DocSearchResult {
  path: string;
  title: string;
  snippet: string;
}

interface DocSearchProps {
  onResultClick?: () => void;
}

// Simple search index - in production, this could be pre-built or use a search service
const DOC_SEARCH_INDEX: Array<{ path: string; title: string; keywords: string[] }> = [
  { path: "", title: "Project Summary", keywords: ["introduction", "overview", "summary", "r3mes"] },
  { path: "getting-started/quickstart", title: "Quick Start", keywords: ["quickstart", "getting started", "begin", "start"] },
  { path: "getting-started/installation", title: "Installation", keywords: ["install", "setup", "configure", "installation"] },
  { path: "node-operators/running-a-node", title: "Running a Node", keywords: ["node", "blockchain", "validator", "operator"] },
  { path: "node-operators/running-a-validator", title: "Running a Validator", keywords: ["validator", "staking", "consensus"] },
  { path: "node-operators/miner-setup", title: "Miner Setup", keywords: ["miner", "mining", "gpu", "setup"] },
  { path: "miner-engine/overview", title: "Miner Engine Overview", keywords: ["miner", "engine", "training", "gradient"] },
  { path: "miner-engine/configuration", title: "Miner Configuration", keywords: ["config", "configuration", "settings"] },
  { path: "protocol-design/architecture", title: "Architecture", keywords: ["architecture", "design", "system", "structure"] },
  { path: "protocol-design/consensus", title: "Consensus", keywords: ["consensus", "blockchain", "proof", "verification"] },
  { path: "protocol-design/security", title: "Security", keywords: ["security", "verification", "trap", "sandbox"] },
  { path: "protocol-design/economics", title: "Economics", keywords: ["economics", "rewards", "token", "incentives"] },
  { path: "protocol-design/governance", title: "Governance", keywords: ["governance", "voting", "proposal", "dao"] },
  { path: "web-and-tools/dashboard", title: "Web Dashboard", keywords: ["dashboard", "web", "ui", "interface"] },
  { path: "web-and-tools/desktop-launcher", title: "Desktop Launcher", keywords: ["launcher", "desktop", "app", "tauri"] },
  { path: "web-and-tools/api", title: "API Reference", keywords: ["api", "endpoint", "rest", "reference"] },
];

export default function DocSearch({ onResultClick }: DocSearchProps) {
  const [query, setQuery] = useState("");
  const [isOpen, setIsOpen] = useState(false);
  const [results, setResults] = useState<DocSearchResult[]>([]);
  const router = useRouter();

  const searchResults = useMemo(() => {
    if (!query.trim()) {
      return [];
    }

    const lowerQuery = query.toLowerCase();
    const matched: DocSearchResult[] = [];

    for (const doc of DOC_SEARCH_INDEX) {
      const titleMatch = doc.title.toLowerCase().includes(lowerQuery);
      const keywordMatch = doc.keywords.some(kw => kw.toLowerCase().includes(lowerQuery));
      
      if (titleMatch || keywordMatch) {
        matched.push({
          path: doc.path,
          title: doc.title,
          snippet: doc.keywords.slice(0, 3).join(", "),
        });
      }
    }

    return matched.slice(0, 8); // Limit to 8 results
  }, [query]);

  useEffect(() => {
    setResults(searchResults);
  }, [searchResults]);

  const handleResultClick = (path: string) => {
    router.push(`/docs/${path}`);
    setQuery("");
    setIsOpen(false);
    onResultClick?.();
  };

  return (
    <div className="relative w-full max-w-md">
      <div className="relative">
        <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-slate-400" />
        <input
          type="text"
          value={query}
          onChange={(e) => {
            setQuery(e.target.value);
            setIsOpen(true);
          }}
          onFocus={() => setIsOpen(true)}
          placeholder="Search documentation..."
          className="w-full pl-10 pr-10 py-2 bg-slate-800 border border-slate-700 rounded-lg text-slate-100 placeholder-slate-500 focus:outline-none focus:border-green-500"
        />
        {query && (
          <button
            onClick={() => {
              setQuery("");
              setIsOpen(false);
            }}
            className="absolute right-3 top-1/2 transform -translate-y-1/2 text-slate-400 hover:text-slate-200"
          >
            <X className="w-4 h-4" />
          </button>
        )}
      </div>

      {isOpen && results.length > 0 && (
        <div className="absolute top-full left-0 right-0 mt-2 bg-slate-800 border border-slate-700 rounded-lg shadow-xl z-50 max-h-96 overflow-y-auto">
          {results.map((result, index) => (
            <button
              key={index}
              onClick={() => handleResultClick(result.path)}
              className="w-full text-left px-4 py-3 hover:bg-slate-700 transition-colors border-b border-slate-700 last:border-b-0"
            >
              <div className="flex items-start gap-3">
                <FileText className="w-4 h-4 text-green-400 mt-0.5 flex-shrink-0" />
                <div className="flex-1 min-w-0">
                  <div className="text-slate-100 font-medium">{result.title}</div>
                  <div className="text-sm text-slate-400 mt-1 truncate">{result.snippet}</div>
                </div>
              </div>
            </button>
          ))}
        </div>
      )}

      {isOpen && query && results.length === 0 && (
        <div className="absolute top-full left-0 right-0 mt-2 bg-slate-800 border border-slate-700 rounded-lg shadow-xl z-50 p-4 text-center text-slate-400">
          No results found
        </div>
      )}
    </div>
  );
}

