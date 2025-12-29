import { useState, useEffect } from "react";
import { invoke } from "@tauri-apps/api/tauri";
import { FileText, Search, Filter, Download, RefreshCw, X } from "lucide-react";

interface LogEntry {
  timestamp: string;
  level: string;
  message: string;
  process: string;
}

interface LogViewerProps {
  process: string;
}

export default function LogViewer({ process }: LogViewerProps) {
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [filteredLogs, setFilteredLogs] = useState<LogEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [levelFilter, setLevelFilter] = useState<string>("all");
  const [autoRefresh, setAutoRefresh] = useState(true);

  const fetchLogs = async () => {
    try {
      setLoading(true);
      setError(null);
      const result = await invoke<string[]>("tail_log_file", {
        process,
        lines: 100,
      });
      
      // Parse log entries (simplified - would need proper parsing)
      const entries: LogEntry[] = result.map((line) => {
        // Try to parse timestamp and level from log line
        const timestampMatch = line.match(/\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]/);
        const levelMatch = line.match(/(ERROR|WARN|INFO|DEBUG)/i);
        
        return {
          timestamp: timestampMatch ? timestampMatch[1] : new Date().toISOString(),
          level: levelMatch ? levelMatch[1].toUpperCase() : "INFO",
          message: line,
          process,
        };
      });
      
      setLogs(entries);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to fetch logs");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchLogs();
    if (autoRefresh) {
      const interval = setInterval(fetchLogs, 2000); // Refresh every 2 seconds
      return () => clearInterval(interval);
    }
  }, [process, autoRefresh]);

  useEffect(() => {
    let filtered = logs;

    // Apply search filter
    if (searchQuery) {
      filtered = filtered.filter((log) =>
        log.message.toLowerCase().includes(searchQuery.toLowerCase())
      );
    }

    // Apply level filter
    if (levelFilter !== "all") {
      filtered = filtered.filter((log) => log.level === levelFilter);
    }

    setFilteredLogs(filtered);
  }, [logs, searchQuery, levelFilter]);

  const exportLogs = async () => {
    try {
      const logText = filteredLogs.map((log) => log.message).join("\n");
      // In production, use Tauri's save dialog
      const blob = new Blob([logText], { type: "text/plain" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${process}-logs-${new Date().toISOString()}.txt`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error("Failed to export logs:", err);
    }
  };

  const getLevelColor = (level: string) => {
    switch (level.toUpperCase()) {
      case "ERROR":
        return "text-red-400";
      case "WARN":
        return "text-yellow-400";
      case "INFO":
        return "text-blue-400";
      case "DEBUG":
        return "text-slate-400";
      default:
        return "text-slate-300";
    }
  };

  if (loading && logs.length === 0) {
    return (
      <div className="card p-6">
        <div className="animate-pulse">
          <div className="h-4 bg-slate-700 rounded w-1/4 mb-4"></div>
          <div className="h-4 bg-slate-700 rounded w-full mb-2"></div>
          <div className="h-4 bg-slate-700 rounded w-5/6"></div>
        </div>
      </div>
    );
  }

  return (
    <div className="card p-6">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-2xl font-bold flex items-center gap-2">
          <FileText className="w-6 h-6 text-green-400" />
          Logs: {process}
        </h2>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setAutoRefresh(!autoRefresh)}
            className={`p-2 rounded-lg transition-colors ${
              autoRefresh ? "bg-green-500/20 text-green-400" : "hover:bg-slate-800"
            }`}
            title={autoRefresh ? "Auto-refresh enabled" : "Auto-refresh disabled"}
          >
            <RefreshCw className={`w-5 h-5 ${autoRefresh ? "animate-spin" : ""}`} />
          </button>
          <button
            onClick={exportLogs}
            className="p-2 hover:bg-slate-800 rounded-lg transition-colors"
            title="Export logs"
          >
            <Download className="w-5 h-5" />
          </button>
        </div>
      </div>

      {/* Filters */}
      <div className="flex gap-4 mb-4">
        <div className="flex-1 relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-slate-400" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search logs..."
            className="w-full pl-10 pr-10 py-2 bg-slate-800 border border-slate-700 rounded-lg text-slate-100 placeholder-slate-500 focus:outline-none focus:border-green-500"
          />
          {searchQuery && (
            <button
              onClick={() => setSearchQuery("")}
              className="absolute right-3 top-1/2 transform -translate-y-1/2 text-slate-400 hover:text-slate-200"
            >
              <X className="w-4 h-4" />
            </button>
          )}
        </div>
        <select
          value={levelFilter}
          onChange={(e) => setLevelFilter(e.target.value)}
          className="px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-slate-100 focus:outline-none focus:border-green-500"
        >
          <option value="all">All Levels</option>
          <option value="ERROR">Error</option>
          <option value="WARN">Warning</option>
          <option value="INFO">Info</option>
          <option value="DEBUG">Debug</option>
        </select>
      </div>

      {/* Logs */}
      {error ? (
        <div className="text-red-400 p-4 bg-red-500/10 rounded-lg">
          {error}
        </div>
      ) : (
        <div className="bg-slate-900 rounded-lg p-4 max-h-96 overflow-y-auto font-mono text-sm">
          {filteredLogs.length === 0 ? (
            <div className="text-slate-400 text-center py-8">No logs found</div>
          ) : (
            filteredLogs.map((log, index) => (
              <div
                key={index}
                className="border-b border-slate-800 py-2 hover:bg-slate-800/50 transition-colors"
              >
                <div className="flex items-start gap-3">
                  <span className={`text-xs font-bold ${getLevelColor(log.level)} min-w-[60px]`}>
                    {log.level}
                  </span>
                  <span className="text-xs text-slate-500 min-w-[150px]">
                    {log.timestamp}
                  </span>
                  <span className="text-slate-300 flex-1 break-all">{log.message}</span>
                </div>
              </div>
            ))
          )}
        </div>
      )}

      {/* Log count */}
      <div className="mt-4 text-sm text-slate-400 text-center">
        Showing {filteredLogs.length} of {logs.length} log entries
      </div>
    </div>
  );
}
